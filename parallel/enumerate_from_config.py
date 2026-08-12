#!/usr/bin/env python3
"""enumerate_from_config.py — expand config/experiment_matrix.yaml into the full
cell list the dispatcher runs. One TAB-separated line per cell:

    dataset<TAB>variant<TAB>backbone<TAB>fold<TAB>tier<TAB>celltype<TAB>device

  variant  : BASE fragmentation (fg_first / rbrics / ertl_first / rdkit_fg_first).
             The run_experiments.sh phases DERIVE the right vocab from it:
               vtrain/posthoc(full)   -> the base vocab
               posthoc(filter)/mose   -> the *_filter vocab (_vocab_focus_filtered_variants)
               gsat/motifsat          -> the base (full) vocab
  tier     : 'real' (original-labels regime) | 'dnf_*' (relabelled/synthetic-GT regime).
             source_gt datasets use 'source' (trained on original labels; SMARTS GT).
  celltype : vtrain | posthoc | mose | gsat | motifsat
  device   : cpu  (post-hoc explainers — GPU useless)
             gpu  (vanilla-train + ante-hoc — GPU-preferred, CPU-eligible in the mixed pool)

Usage:
    enumerate_from_config.py config/experiment_matrix.yaml [--only-dataset BBBP]
"""
import sys, argparse
try:
    import yaml
except ImportError:
    sys.exit("pyyaml required: pip install pyyaml")

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--only-dataset", default=None, help="restrict to one display dataset name (e.g. BBBP)")
args = ap.parse_args()

c = yaml.safe_load(open(args.config))
bbs = c["backbones"]
base_frags = [f["variant"] for f in c["fragmentations"] if not f["filtered"]]  # 4 base
ds_meta = c["datasets"]
r = c["regimes"]

# --- phantom-tier guard ------------------------------------------------------
# Only emit a synthetic-GT (dnf_*) cell for a (dataset, variant) whose rule_tiers.json
# actually contains that tier. The DNF engine does NOT guarantee every arity/rank
# exists per fragmentation (e.g. rbrics has no dnf_k3_r2), so the uniform config tier
# list would otherwise emit orphaned cells that soft-skip and get marked done.
# Conservative: if rule_tiers.json is absent/unreadable (vocab not built yet), do NOT
# filter — emit exactly as before (never drop a legitimate cell on a missing file).
import os, re
_VOCAB_ROOT = os.environ.get("VOCAB_ROOT", "vocab_final_v2")
_tier_cache = {}
def _accepted_dnf_tiers(code, base):
    key = (code, base)
    if key not in _tier_cache:
        p = os.path.join(_VOCAB_ROOT, code, base, "rule_tiers.json")
        try:
            _tier_cache[key] = set(re.findall(r"dnf_k\d+_r\d+", open(p).read()))
        except OSError:
            _tier_cache[key] = None   # missing/unreadable → do not filter
    return _tier_cache[key]

# (display_dataset, tier) pairs across both regimes -----------------------------
# regime groups are OPTIONAL — a scoped config may defer synthetic_gt or omit source_gt.
runs = []
for disp in r.get("original_labels", {}).get("datasets", []):
    runs.append((disp, "real"))
_groups = r.get("relabelled", {}).get("groups", {}) or {}
for _gkey in ("synthetic_gt", "source_gt"):
    _g = _groups.get(_gkey)
    if not _g:
        continue
    for disp in _g["datasets"]:
        for t in _g["tiers"]:
            runs.append((disp, t))

# every cell is keyed by a BASE variant; the phase derives full-vs-filter per celltype.
# vtrain+posthoc (Vanilla) always; ante-hoc celltypes are derived from the config's models,
# so a config that omits MotifSAT (or MOSE/GSAT) simply won't emit those cells.
_MODEL_CELLTYPE = {"MOSE": ("mose", "gpu"), "GSAT": ("gsat", "gpu"), "MotifSAT": ("motifsat", "gpu")}
CELLTYPES = [("vtrain", "gpu"), ("posthoc", "cpu")]
for _m in c.get("models", {}):
    if _m in _MODEL_CELLTYPE:
        CELLTYPES.append(_MODEL_CELLTYPE[_m])

out = []
for disp, tier in runs:
    if args.only_dataset and disp != args.only_dataset:
        continue
    if disp not in ds_meta:
        sys.exit(f"dataset {disp!r} in regimes but not in datasets: block")
    code = ds_meta[disp]["code"]
    folds = list(range(ds_meta[disp]["folds"]))
    # Per-dataset fragmentation restriction (default: all 4 base variants). e.g. molhiv
    # is rbrics-only because the FG fragmenters fail on its organometallic molecules.
    _frags = ds_meta[disp].get("frag_variants", base_frags)
    for bb in bbs:
        for fold in folds:
            for base in _frags:
                if tier.startswith("dnf_"):
                    _acc = _accepted_dnf_tiers(code, base)
                    if _acc is not None and tier not in _acc:
                        continue   # phantom tier — rule not planted for this variant
                for celltype, device in CELLTYPES:
                    out.append((code, base, bb, str(fold), tier, celltype, device))

# stable de-dup (a dataset could appear in >1 regime with the same tier only if mis-specified)
seen = set(); lines = []
for row in out:
    if row in seen:
        continue
    seen.add(row); lines.append("\t".join(row))
sys.stdout.write("\n".join(lines) + ("\n" if lines else ""))
