#!/usr/bin/env bash
# planted_prep.sh — P1 of the MotifSAT planted-GT campaign (approved 2026-10-26). Run LOCALLY:
#     bash planted_prep.sh            # submits stage A (15 tasks) then stage B (140 tasks, afterok:A)
# Submits itself through sbatch stdin (no file copy). Nothing under planted_v2/ is written.
#
# For each (dataset, rule, fold) of the 31 selected rules x 5 folds:
#   1. planted_v3/vocab/<ds>/rbrics/ = symlinks to vocab_final_v3/<ds>/rbrics/* (rBRICS v3, with copies)
#      + rule_tiers.json COPIED from planted_v2/<ds>/_shared/vocab/<ds>/rbrics/ (the sampled rules).
#   2. SharedModules/data/apply_gt.py --tier <rule> -> planted_v3/gt_cache/<ds>/fold<f>/rbrics/relabel_<rule>/
#      {train,valid,test}_with_gt.pt (fired-clause node_label, per-motif node_label_spurious, v3 motif + copy ids).
#   3. CHECK vs the planted_v2 cache of the same rule/fold, molecules matched by SMILES:
#      same SMILES set per split; y identical on every molecule (else FAIL, exit 3); count of molecules whose
#      ground-truth atom mask differs (v3 atom-tracking fix); every graph has nodes_to_copy.
#      One JSON line -> planted_v3/_prep/<ds>__<rule>__f<f>.json
# Stage A runs the FIRST rule of every (dataset, fold) so the processed cache is built once; stage B the rest.
set -euo pipefail
if [ "${1:-}" != "--task" ]; then
    LOG=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/motifsat_paper_deliverables_v1/planted_v3/_prep/logs
    SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    hdr(){ printf '#!/usr/bin/env bash\n#SBATCH -J planted_prep_%s\n#SBATCH -p share,preempt,dgx2,gpu\n#SBATCH --gres=gpu:0\n#SBATCH -c 2\n#SBATCH --mem=16G\n#SBATCH -t 02:00:00\n#SBATCH --requeue\n#SBATCH --array=%s\n#SBATCH -o %s/%s_%%A_%%a.out\n' "$1" "$2" "$LOG" "$1"; }
    A=$( { hdr A 0-14; sed -n '/^# ===BODY===/,$p' "$SELF" | sed 's/^__STAGE__=.*/__STAGE__=A/'; } | ssh osuhpc "mkdir -p $LOG && sbatch --parsable")
    echo "stage A: $A"
    B=$( { hdr B 0-139; sed -n '/^# ===BODY===/,$p' "$SELF" | sed 's/^__STAGE__=.*/__STAGE__=B/'; } | ssh osuhpc "sbatch --parsable --dependency=afterok:$A")
    echo "stage B: $B (afterok:$A)"
    exit 0
fi
# ===BODY===
__STAGE__=A
set -uo pipefail
R=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor; O=$R/motifsat_paper_deliverables_v1/planted_v3
FOLDS_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS/
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
cd "$R"; export PYTHONPATH="$R" OMP_NUM_THREADS=2
# The 31 rules (stratified, sampling order; see EXPERIMENT_LOG §12). First rule of each dataset = stage A.
declare -A RULES=(
  [BBBP]="dnf_k1_r1 dnf_k1_r2 dnf_k3_r8 dnf_k2_r6 dnf_k2_r7 dnf_k2_r10 dnf_k3_r1 dnf_k3_r2 dnf_k3_r3 dnf_k3_r4"
  [hERG]="dnf_k1_r1 dnf_k1_r2 dnf_k2_r2 dnf_k2_r6 dnf_k3_r12 dnf_k2_r9 dnf_k2_r1 dnf_k3_r1 dnf_k3_r2 dnf_k3_r3"
  [Mutagenicity]="dnf_k1_r1 dnf_k1_r2 dnf_k1_r3 dnf_k3_r1 dnf_k3_r2 dnf_k3_r3 dnf_k3_r4 dnf_k3_r6 dnf_k3_r7 dnf_k3_r8 dnf_k2_r4"
)
TASKS=()
for ds in BBBP hERG Mutagenicity; do
  read -ra rs <<< "${RULES[$ds]}"
  for f in 0 1 2 3 4; do
    for i in "${!rs[@]}"; do
      if [ "$__STAGE__" = A ] && [ "$i" -eq 0 ]; then TASKS+=("$ds ${rs[$i]} $f"); fi
      if [ "$__STAGE__" = B ] && [ "$i" -gt 0 ]; then TASKS+=("$ds ${rs[$i]} $f"); fi
    done
  done
done
read -r DS RULE F <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "stage $__STAGE__ task $SLURM_ARRAY_TASK_ID: $DS $RULE fold $F"

# 1. vocab dir (idempotent; safe under concurrent tasks)
V=$O/vocab/$DS/rbrics; mkdir -p "$V" "$O/_prep"
for src in "$R/vocab_final_v3/$DS/rbrics"/*; do
  b=$(basename "$src"); [ "$b" = rule_tiers.json ] && continue
  [ -e "$V/$b" ] || ln -s "$src" "$V/$b" 2>/dev/null || true
done
tmp=$(mktemp "$V/.rt.XXXX"); cp "$R/planted_v2/$DS/_shared/vocab/$DS/rbrics/rule_tiers.json" "$tmp"; mv -f "$tmp" "$V/rule_tiers.json"

# 2. relabel caches with the v3 vocabulary
python3 SharedModules/data/apply_gt.py --dataset "$DS" --fold "$F" --vocab_root "$O/vocab" --variant rbrics \
    --tier "$RULE" --data_root "$FOLDS_ROOT" --out_dir "$O/gt_cache" --processed_root "$O/processed" \
    > "$O/_prep/${DS}__${RULE}__f${F}.apply_gt.log" 2>&1 || { echo "apply_gt FAILED"; tail -20 "$O/_prep/${DS}__${RULE}__f${F}.apply_gt.log"; exit 2; }

# 3. check against planted_v2
python3 - "$DS" "$RULE" "$F" <<'PY'
import json, sys, torch
ds, rule, f = sys.argv[1], sys.argv[2], int(sys.argv[3])
R = '/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor'
new = f'{R}/motifsat_paper_deliverables_v1/planted_v3/gt_cache/{ds}/fold{f}/rbrics/relabel_{rule}'
old = f'{R}/planted_v2/{ds}/{rule}/gt_cache/{ds}/fold{f}/rbrics/relabel_{rule}'
out = dict(dataset=ds, rule=rule, fold=f, ok=True)
for split in ('train', 'valid', 'test'):
    a = torch.load(f'{new}/{split}_with_gt.pt', weights_only=False)
    b = torch.load(f'{old}/{split}_with_gt.pt', weights_only=False)
    A = {g.smiles: g for g in a}; B = {g.smiles: g for g in b}
    same_set = set(A) == set(B)
    y_mis = sum(1 for s in set(A) & set(B) if float(A[s].y.view(-1)[0]) != float(B[s].y.view(-1)[0]))
    nl_mis = sum(1 for s in set(A) & set(B)
                 if A[s].node_label.numel() != B[s].node_label.numel()
                 or not torch.equal(A[s].node_label.view(-1).float(), B[s].node_label.view(-1).float()))
    no_copy = sum(1 for g in a if getattr(g, 'nodes_to_copy', None) is None)
    out[split] = dict(n_new=len(a), n_old=len(b), same_smiles=same_set, y_mismatch=y_mis,
                      gt_mask_changed=nl_mis, no_copy=no_copy, n_pos=int(sum(float(g.y.view(-1)[0]) for g in a)))
    if not same_set or y_mis or no_copy:
        out['ok'] = False
json.dump(out, open(f'{R}/motifsat_paper_deliverables_v1/planted_v3/_prep/{ds}__{rule}__f{f}.json', 'w'))
print(json.dumps(out))
sys.exit(0 if out['ok'] else 3)
PY
