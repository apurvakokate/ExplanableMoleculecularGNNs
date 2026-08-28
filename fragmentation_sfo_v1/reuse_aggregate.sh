#!/usr/bin/env bash
P=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor; EXP=$P/fragmentation_v2
OUT=$EXP/eval_reuse; TRK=$EXP/_tracking_reuse
DS='Mutagenicity BBBP hERG esol Lipophilicity Benzene_Verified_GT Alkane_Carbonyl_Verified_GT Fluoride_Carbonyl_Verified_GT'
for V in sfo rbrics; do for M in gnnexplainer pgexplainer motif_occlusion gsat; do for D in $DS; do
  files=$(ls $OUT/$V/${M}_${D}_f*_*.csv 2>/dev/null); [ -z "$files" ] && continue
  first=$(echo $files | awk '{print $1}'); head -1 $first > $OUT/$V/${M}_${D}_allfolds.csv
  for f in $files; do tail -n +2 $f >> $OUT/$V/${M}_${D}_allfolds.csv; done
done; done; done
echo "===== TRACKING ROSTER ====="
tot=$(ls $TRK/*.status 2>/dev/null | wc -l)
okc=$(grep -l 'rc=0 ' $TRK/*.status 2>/dev/null | wc -l)
echo "cells with status: $tot / 400   OK(rc=0): $okc   FAIL: $((tot-okc))"
echo "--- failures ---"; grep -h 'rc=[1-9]' $TRK/*.status 2>/dev/null | head -40
echo "--- missing (never ran) ---"; echo "expected 400, have $tot"
