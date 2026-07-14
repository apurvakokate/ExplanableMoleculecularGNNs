#!/usr/bin/env python3
"""merge_explainer_results.py — splice one explainer's artifacts from a side run
tree back into the main results tree.

Motivation
----------
``run_vanilla.py`` rewrites ``summary.json`` wholesale, so a partial baselines run
(e.g. ``MAGE_ONLY=1``) must be written to a separate ``BASELINE_OUT_ROOT`` to avoid
dropping the GNNExplainer/PGExplainer/Motif-Occlusion metrics already recorded in
the main tree. This script merges that side tree back in:

  * copies ``{explainer}_*`` artifacts (motif-score CSVs, score_vs_impact CSVs)
  * splices ``{explainer}_*`` keys from the side summary.json into the main one

Every other key/file in the destination is left byte-for-byte untouched, so the
existing explainers' metrics (including their unrecoverable per-instance numbers)
survive. Re-running is idempotent: the explainer's keys are simply overwritten.

Usage
-----
    # dry run first — shows exactly what would change
    python3 analysis/merge_explainer_results.py \
        --src-root results_mage --dst-root results --dry-run

    # then apply
    python3 analysis/merge_explainer_results.py \
        --src-root results_mage --dst-root results

Run dirs are matched by their path RELATIVE to the root, so the two trees must
share the same <family>/<dataset>/fold<k>/<variant>/<cfg_slug> layout (they do —
both are produced by run_experiments.sh with the same naming helpers).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List


def _load_json(p: Path) -> Dict:
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def merge_run(
    src_dir: Path,
    dst_dir: Path,
    prefix: str,
    dry_run: bool,
    backup: bool,
) -> Dict[str, int]:
    """Merge one run dir. Returns counts for the report."""
    stats = {'keys': 0, 'files': 0}

    src_summary_p = src_dir / 'summary.json'
    dst_summary_p = dst_dir / 'summary.json'
    src_summary = _load_json(src_summary_p)

    # Keys to splice: exactly this explainer's namespace.
    new_keys = {k: v for k, v in src_summary.items() if k.startswith(prefix + '_')}
    if not new_keys:
        return stats

    dst_summary = _load_json(dst_summary_p)
    # Preserve everything already in dst; overwrite only this explainer's keys.
    merged = dict(dst_summary)
    merged.update(new_keys)
    stats['keys'] = len(new_keys)

    # Artifacts belonging to this explainer (motif score CSVs, score_vs_impact CSVs).
    art: List[Path] = sorted(
        p for p in src_dir.iterdir()
        if p.is_file() and p.name.startswith(prefix + '_') and p.name != 'summary.json'
    )
    stats['files'] = len(art)

    if dry_run:
        return stats

    if backup and not (dst_dir / 'summary.json.premerge').exists():
        shutil.copy2(dst_summary_p, dst_dir / 'summary.json.premerge')
    for p in art:
        shutil.copy2(p, dst_dir / p.name)
    with open(dst_summary_p, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2)

    return stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge one explainer's results from a side tree into the main results tree.")
    ap.add_argument('--src-root', required=True,
                    help='Side tree written by the partial run (e.g. results_mage).')
    ap.add_argument('--dst-root', required=True,
                    help='Main results tree to merge into (e.g. results).')
    ap.add_argument('--explainer', default='mage_official',
                    help='Explainer key namespace to move (default: mage_official).')
    ap.add_argument('--dry-run', action='store_true',
                    help='Report what would change without writing anything.')
    ap.add_argument('--no-backup', action='store_true',
                    help='Do not write summary.json.premerge backups.')
    args = ap.parse_args()

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    prefix = args.explainer
    if not src_root.is_dir():
        print(f'ERROR: --src-root not a directory: {src_root}', file=sys.stderr)
        return 2
    if not dst_root.is_dir():
        print(f'ERROR: --dst-root not a directory: {dst_root}', file=sys.stderr)
        return 2
    if src_root == dst_root:
        print('ERROR: --src-root and --dst-root are the same tree.', file=sys.stderr)
        return 2

    merged = skipped_no_dst = skipped_no_keys = 0
    tot_keys = tot_files = 0
    missing: List[str] = []

    for src_summary in sorted(src_root.rglob('summary.json')):
        src_dir = src_summary.parent
        rel = src_dir.relative_to(src_root)
        dst_dir = dst_root / rel

        if not (dst_dir / 'summary.json').exists():
            skipped_no_dst += 1
            missing.append(str(rel))
            continue

        try:
            st = merge_run(src_dir, dst_dir, prefix, args.dry_run, not args.no_backup)
        except Exception as e:
            print(f'  [warn] {rel}: {e}')
            continue

        if st['keys'] == 0:
            skipped_no_keys += 1
            continue

        merged += 1
        tot_keys += st['keys']
        tot_files += st['files']
        print(f"  {'[dry] ' if args.dry_run else ''}{rel}  "
              f"+{st['keys']} {prefix}_* keys, {st['files']} file(s)")

    print('\n' + '=' * 62)
    print(f'  {"DRY RUN — nothing written" if args.dry_run else "MERGE COMPLETE"}')
    print(f'  explainer namespace : {prefix}_*')
    print(f'  run dirs merged     : {merged}')
    print(f'  summary keys spliced: {tot_keys}')
    print(f'  artifact files copied: {tot_files}')
    if skipped_no_keys:
        print(f'  skipped (no {prefix}_* keys in src): {skipped_no_keys}')
    if skipped_no_dst:
        print(f'  skipped (NO MATCHING DST run dir)  : {skipped_no_dst}')
        for m in missing[:10]:
            print(f'      missing in dst: {m}')
        if len(missing) > 10:
            print(f'      ... and {len(missing) - 10} more')
        print('  ^ these src runs have no counterpart in --dst-root. If that is')
        print('    unexpected, the two trees disagree on backbone/fold/variant.')
    if not args.dry_run and not args.no_backup and merged:
        print('  originals backed up as summary.json.premerge in each dst run dir')
    print('=' * 62)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
