#!/bin/bash
# hpc_deploy_launch.sh — spawn N shard workers as ONE SLURM array (default 300) that
# run hpc_deploy_worker.sh. Re-runnable: the driver skips completed setups, so
# re-submitting just fills gaps (preempted/failed shards). Per-shard failure logs land
# in $DEST/_failures/; SLURM stdout in $DEST/_slurmlogs/.
#
# Config (env with defaults). Roots default inside the worker; override here to change
# the run. To reuse for a different driver: SCRIPT=analysis/other.py N=... bash <this>.
#   N       number of shards / array tasks (= workers)          default 300
#   DEVICE  cpu | cuda                                          default cpu
#   SCRIPT  python driver relative to REPO                      default GT-ROC sweep
#   DEST    NEW output tree                                     default gtroc_nodedirect_v1
#   PART    SLURM partitions        GRES  --gres        TIME  walltime
set -uo pipefail
REPO="${REPO:-/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor}"
N="${N:-300}"
DEVICE="${DEVICE:-cpu}"
SCRIPT="${SCRIPT:-analysis/validate_posthoc_gtroc.py}"
DEST="${DEST:-$REPO/gtroc_nodedirect_v1}"
PART="${PART:-preempt,share}"
GRES="${GRES:-gpu:0}"
TIME="${TIME:-8:00:00}"
W="$REPO/hpc_deploy_worker.sh"
SLOG="$DEST/_slurmlogs"; mkdir -p "$SLOG" "$DEST/_failures"

# Submit the worker AS the batch script (+ --export), NOT via --wrap: this cluster's
# sbatch rejects script args with --wrap. The worker derives SHARD from
# SLURM_ARRAY_TASK_ID + SHARD_N at runtime.
sbatch --array=0-$((N-1)) --requeue -p "$PART" --gres="$GRES" -c 2 --mem=16G -t "$TIME" \
  -J gtroc -o "$SLOG/%A_%a.out" \
  --export=ALL,REPO="$REPO",DEST="$DEST",DEVICE="$DEVICE",SCRIPT="$SCRIPT",SHARD_N="$N" \
  "$W"

echo "submitted SLURM array 0-$((N-1))  (N=$N, device=$DEVICE, script=$SCRIPT)  -> $DEST"
echo "watch:    squeue -u \$USER -h -o '%t' | sort | uniq -c"
echo "failures: cat $DEST/_failures/*.log   |   done: find $DEST -name gtroc.json | wc -l"
