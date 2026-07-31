#!/usr/bin/env bash
# queue_worker_v2.sh <cuda_devices> <total> <claims> <done> <pdir> <cellfile1> [cellfile2 ...]
#
# One worker. CUDA_VISIBLE_DEVICES = <cuda_devices> ("" = CPU-only, hides all GPUs;
# "<n>" = pin GPU n). Drains the given cell files IN PRIORITY ORDER, re-scanning until
# the shared queue is fully claimed (claimed >= total). `mkdir` = atomic cross-node claim.
#
# posthoc cells are CHECKPOINT-GATED: only claimed once the vanilla best_model.pt exists,
# so post-hoc never runs before its Stage-1 training finished (produces empty results).
set -uo pipefail
export CUDA_VISIBLE_DEVICES="$1"; TOTAL="$2"; CLAIMS="$3"; DONE="$4"; PDIR="$5"; shift 5
CELLFILES=("$@")
# shellcheck disable=SC1091
source "$PDIR/final_v2_env.sh"
# GPU cells get a couple threads for dataloading; CPU-only cells pinned to 1.
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
else export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; fi

ckpt_ready() {   # ds variant bb fold tier
    # 'source' tier trains on ORIGINAL labels like 'real' (source-GT / *_Verified_GT datasets),
    # so its vtrain checkpoint lands under the _real suffix, NOT _gt_source. Gate posthoc on the
    # real ckpt for both, else source posthoc is never claimable.
    local suf; if [ "$5" = real ] || [ "$5" = source ]; then suf=real; else suf="gt_$5"; fi
    # Encoder must match how vtrain names the checkpoint: OGB datasets train with the
    # atom_encoder, everything else with onehot. Hardcoding onehot made every OGB posthoc
    # cell permanently un-claimable (checkpoint sought at the wrong path).
    local enc=onehot; case "$1" in ogbg-*) enc=atom_encoder ;; esac
    [ -f "$OUT_ROOT/vanilla/$1/fold$4/$2/bb-${3}_enc-${enc}_norm-none_${suf}/best_model.pt" ]
}

while true; do
    [ "$(ls "$CLAIMS" 2>/dev/null | wc -l)" -ge "$TOTAL" ] && break
    progress=0
    for cf in "${CELLFILES[@]}"; do
        [ -f "$cf" ] || continue
        while IFS=$'\t' read -r ds variant bb fold tier celltype device; do
            [ -z "${ds:-}" ] && continue
            key="${ds}__${variant}__${bb}__f${fold}__${tier}__${celltype}"
            [ -d "$CLAIMS/$key" ] && continue                       # already claimed
            if [ "$celltype" = posthoc ] && ! ckpt_ready "$ds" "$variant" "$bb" "$fold" "$tier"; then
                continue                                           # dep not ready → try next scan
            fi
            mkdir "$CLAIMS/$key" 2>/dev/null || continue           # lost the race
            if bash "$PDIR/run_cell.sh" "$ds" "$variant" "$bb" "$fold" "$tier" "$celltype" \
                    > "$FINAL_LOGS/cell_${key}.log" 2>&1; then
                touch "$DONE/$key"                                 # success → done
            else
                # Failure is usually a transient NFS processed-cache build race
                # (PytorchStreamReader / EOFError): the first cell for a
                # (variant,fold) is still writing the .pt while siblings read it.
                # Auto-retry by RELEASING the claim so another pass re-runs it —
                # by then the cache is warm. Attempts persist outside the claim;
                # after RETRY_CAP we give up and mark done (rc!=0 stays in the
                # timing log) so the queue can still terminate.
                _adir="$(dirname "$CLAIMS")/attempts"; mkdir -p "$_adir"
                _n=$(( $(cat "$_adir/$key" 2>/dev/null || echo 0) + 1 )); echo "$_n" > "$_adir/$key"
                if [ "$_n" -ge "${RETRY_CAP:-3}" ]; then
                    touch "$DONE/$key"                             # give up after cap
                else
                    rmdir "$CLAIMS/$key" 2>/dev/null               # release → retry next pass
                fi
            fi
            progress=1
        done < "$cf"
    done
    # nothing claimable this pass (e.g. posthoc waiting on checkpoints) → brief wait
    [ "$progress" = 0 ] && sleep 15
done
