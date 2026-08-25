#!/usr/bin/env bash
# Deploy the mose_replication_v2 drain in parallel: build the worklist, init the cursor, and
# submit CPU-only worker arrays across partitions (all drain the SAME queue). 2 cores/worker
# per the SLURM sizing rule -> 250 workers = 500 cores.
#
# Per HPC scheduling notes: a single big preempt array starves CPU tasks, so we spread across
# partitions that accept --gres=gpu:0 (eecs reliable; share large; dgx2/gpu CPU fan-out).
# Tune WORKERS via PARTS="partition:count ..." to match current cluster headroom.
set -euo pipefail

P=${P:-/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor}
V2=${V2:-$P/mose_replication_v2}
PARTS=${PARTS:-"eecs:40 share:150 dgx2:60"}     # partition:num_workers  (sum = total workers)
WALL_SECONDS=${WALL_SECONDS:-43000}
D=$V2/_deploy

mkdir -p "$D/logs"
# 1) worklist (idempotent — rebuild is cheap; completed shards are skipped by the worker)
python3 "$P/analysis/mose_replication_v2/make_worklist.py" --base "$P" --out "$D/worklist.tsv"
# 2) cursor: start at row 2 (row 1 is the header comment)
echo 2 > "$D/cursor"; : > "$D/lock"
echo "queued $(( $(wc -l < "$D/worklist.tsv") - 1 )) shards; submitting workers: $PARTS"

# 3) one array per partition, all draining $D/cursor
for pc in $PARTS; do
  part=${pc%%:*}; n=${pc##*:}
  sbatch --parsable -J mrv2 -p "$part" --gres=gpu:0 -c 2 -t 12:00:00 \
    --array=1-"$n"%"$n" -o "$D/logs/worker_${part}_%a.out" \
    --wrap "export P=$P V2=$V2 WALL_SECONDS=$WALL_SECONDS; bash $P/analysis/mose_replication_v2/eval_worker.sh" \
    | sed "s/^/submitted ${part} x${n} job=/"
done
echo "watch: grep -h '^RESULT' $D/logs/worker_*.out | sort | uniq -c ; tail cursor=$(cat $D/cursor)"
