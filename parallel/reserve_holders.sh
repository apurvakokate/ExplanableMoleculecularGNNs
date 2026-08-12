#!/usr/bin/env bash
# reserve_holders.sh — reserve >=300 CPU + >=64 GPU across preempt/dgxh/dgx2/gpu as long-lived
# SLURM "holder" allocations, then print the HOLDERS string for queue_dispatch_v2.sh.
#
#   HOLDERS format: "jobid:cores:gpus jobid:cores:gpus ..."  (queue_node.sh runs `gpus` GPU
#   workers + `cores-2*gpus` CPU workers per holder).
#
# Usage:
#   bash parallel/reserve_holders.sh              # submit the holders, wait, print HOLDERS
#   export HOLDERS="$(bash parallel/reserve_holders.sh --print)"   # capture for the env
#
# Tune the four allocations below to your cluster's node shapes/limits. The default targets
# ~64 GPU + ~350 CPU (>= the 300/64 minimum). A holder is a `salloc`/`sbatch` job that just
# sleeps; queue_node.sh srun's workers into it. Preempt is CPU-heavy (post-hoc + reaggregate).
set -uo pipefail
HOLD_HOURS="${HOLD_HOURS:-24}"
SLEEP="sleep $((HOLD_HOURS*3600))"

# partition : nodes : gpus-per-holder : cores-per-holder : job-name
# Adjust --gres/-c to what each partition allows. Sums to 64 gpu / ~352 cpu.
SPECS=(
  "dgxh:2:16:64:ertl_dgxh"       # 2 holders x 16 gpu = 32 gpu, 128 cpu
  "dgx2:1:16:64:ertl_dgx2"       # 16 gpu, 64 cpu
  "gpu:2:8:48:ertl_gpu"          # 2 x 8 = 16 gpu, 96 cpu
  "preempt:1:0:64:ertl_preempt"  # CPU-only holder (post-hoc explainers + reaggregate), 64 cpu
)

PRINT_ONLY=0; [ "${1:-}" = "--print" ] && PRINT_ONLY=1
holders=""
for spec in "${SPECS[@]}"; do
  IFS=: read -r part nodes gpus cores name <<<"$spec"
  gres=""; [ "$gpus" -gt 0 ] && gres="--gres=gpu:$gpus"
  for _n in $(seq 1 "$nodes"); do
    jid=$(sbatch --parsable -p "$part" $gres -c "$cores" -t "${HOLD_HOURS}:00:00" \
                 --job-name "$name" --wrap "$SLEEP" 2>/dev/null) || {
      echo "[reserve] WARN: could not submit holder on partition '$part'" >&2; continue; }
    holders="$holders $jid:$cores:$gpus"
    [ "$PRINT_ONLY" = 0 ] && echo "[reserve] submitted holder $jid on $part ($gpus gpu / $cores cpu)" >&2
  done
done
holders="${holders# }"
if [ "$PRINT_ONLY" = 1 ]; then
  echo "$holders"
else
  echo "[reserve] wait for RUNNING, then:  export HOLDERS=\"$holders\"" >&2
  echo "$holders"
fi
