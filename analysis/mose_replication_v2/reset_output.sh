#!/usr/bin/env bash
# Wipe the mose_replication_v2 OUTPUT tree (rollups / artifacts / probe / _deploy state) so the
# full run starts clean — run this AFTER the smoke passes, BEFORE submit_workers.sh, so smoke
# shards don't seed the full run. Does NOT touch the code dir ($P/analysis/mose_replication_v2).
set -euo pipefail
P=${P:-/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor}
V2=${V2:-$P/mose_replication_v2}

# safety: only ever wipe a dir literally named mose_replication_v2, and never the code dir
case "$(basename "$V2")" in mose_replication_v2) ;; *) echo "refusing: V2=$V2 not named mose_replication_v2"; exit 1;; esac
[ "$V2" = "$P/analysis/mose_replication_v2" ] && { echo "refusing: \$V2 is the CODE dir"; exit 1; }

echo "about to wipe OUTPUT under: $V2"
for d in rollups artifacts probe _deploy; do
  [ -e "$V2/$d" ] && echo "   rm -rf $V2/$d"
done
rm -rf "$V2/rollups" "$V2/artifacts" "$V2/probe" "$V2/_deploy"
echo "cleared — output tree is empty; code in \$P/analysis/mose_replication_v2 untouched."
