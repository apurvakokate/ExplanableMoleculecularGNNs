#!/usr/bin/env bash
# SLURM array dispatch: ONE GPU per cell, up to CONCURRENCY at a time (default 8 = your
# per-user gpu-QOS cap). Turns wall-clock from Sum(cells)*t into ceil(N/CONCURRENCY)*t.
# Run this AFTER exporting the run env (_setup: PROJECT/OUT_ROOT/VOCAB_ROOT/RULE_ENGINE/...);
# sbatch --export=ALL carries it to the tasks. SLURM_SETUP (conda activate etc.) is inlined.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${PROJECT:?}" "${OUT_ROOT:?}" "${VOCAB_ROOT:?}"
OUTDIR="${OUTDIR:-$PROJECT/logs/cells}"; mkdir -p "$OUTDIR"
CELLS="$OUTDIR/cells_$(date +%Y%m%d_%H%M%S).txt"
bash "$HERE/enumerate_cells.sh" > "$CELLS"
N=$(wc -l < "$CELLS"); [ "$N" -eq 0 ] && { echo "no cells enumerated"; exit 1; }
CONC="${CONCURRENCY:-8}"
echo "[array] $N cells; up to $CONC concurrent GPUs -> ~$(( (N + CONC - 1) / CONC )) waves"
echo "        cells list: $CELLS"

sbatch --export=ALL \
  --array=0-$((N-1))%"$CONC" \
  --partition="${SLURM_PARTITION:-gpu}" --gres=gpu:1 \
  --cpus-per-task="${CPUS:-8}" --mem="${MEM:-64G}" --time="${TIME:-24:00:00}" \
  --job-name="cell" \
  --output="$OUTDIR/cell_%A_%a.out" --error="$OUTDIR/cell_%A_%a.err" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd "$PROJECT"
${SLURM_SETUP:-}
spec=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$CELLS")
echo "[task \$SLURM_ARRAY_TASK_ID] cell: \$spec  (\$(hostname), gpu \$CUDA_VISIBLE_DEVICES)"
# one worker per GPU task -> give it the node's CPUs for its own threads
export OMP_NUM_THREADS="${CPUS:-8}" MKL_NUM_THREADS="${CPUS:-8}"
bash "$HERE/run_cell.sh" \$spec
EOF
