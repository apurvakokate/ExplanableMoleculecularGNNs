#!/usr/bin/env bash
# ablation_ertl_delete.sh — reclaim space by removing ertl_first ABLATION runs.
#
# TARGETS ONLY ertl_first* vocab dirs under the two ABLATION trees:
#     ablated_completely_v1/<family>/ertl_first*          (real + relabelled_dnf planted)
#     ablation_v2/**/ertl_first*
# KEEPS (never matched): final_v2 (base config), antehoc_v1 / antehoc_recompute_v1 (base
# re-eval), and EVERY non-ertl_first vocab (rbrics_filter, fg_first, rdkit_fg_first) — so the
# running rbrics factorial in ablated_completely_v1/mose/rbrics_filter is untouched.
#
# SAFE BY DEFAULT: lists each target dir + total size and deletes NOTHING. Set APPLY=1 to delete.
# Guards are re-asserted immediately before every rm: a path that is not inside an ablation tree,
# or that contains rbrics/fg_first/rdkit/final_v2/antehoc, is REFUSED even if it was listed.
#
# Usage:
#   bash ablation_ertl_delete.sh            # dry run: list targets + reclaim
#   APPLY=1 bash ablation_ertl_delete.sh    # delete the listed dirs
set -uo pipefail
REPO=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor
APPLY="${APPLY:-0}"
TREES=("$REPO/ablated_completely_v1" "$REPO/ablation_v2")
ALLOW_RE='/(ablated_completely_v1|ablation_v2)/'
DENY_RE='rbrics|fg_first|rdkit|/final_v2/|/antehoc'

_ok(){  # path -> 0 if safe to delete
  local d="$1"
  [[ "$d" =~ $ALLOW_RE ]] || return 1
  [[ "$d" =~ $DENY_RE  ]] && return 1
  [[ "$(basename "$d")" == ertl_first* ]] || return 1
  return 0
}

total=0; n=0; list=()
for T in "${TREES[@]}"; do
  [ -d "$T" ] || continue
  while IFS= read -r d; do
    if ! _ok "$d"; then echo "GUARD skip: ${d#$REPO/}"; continue; fi
    sz=$(du -sb "$d" 2>/dev/null | awk '{print $1}'); sz=${sz:-0}
    total=$((total+sz)); n=$((n+1)); list+=("$d")
    printf "%8s  %s\n" "$(du -sh "$d" 2>/dev/null | cut -f1)" "${d#$REPO/}"
  done < <(find "$T" -type d -name 'ertl_first*' -prune 2>/dev/null)
done
echo "---"
echo "targets: $n    total: $(numfmt --to=iec "$total" 2>/dev/null || echo "${total}B")"

if [ "$APPLY" = 1 ]; then
  echo "=== APPLY: deleting ==="
  for d in "${list[@]}"; do
    if ! _ok "$d"; then echo "REFUSE ${d#$REPO/}"; continue; fi
    rm -rf -- "$d" && echo "deleted ${d#$REPO/}"
  done
  echo "done. reclaimed ~$(numfmt --to=iec "$total" 2>/dev/null || echo "${total}B"). Verify: lfs quota -p 28412 ."
else
  echo "DRY RUN — nothing deleted. Re-run with APPLY=1 to remove the above."
fi
