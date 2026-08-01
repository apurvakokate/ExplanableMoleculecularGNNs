#!/usr/bin/env python3
"""ablation_worker.py — read ONE row of run_configs.csv and route it to run_cell.sh with
every ablation knob set EXPLICITLY as an env var. Regime is isolated via a per-regime
OUT_ROOT so Normal/M1 never overwrite; readout/norm/node/unk/filter are isolated in the
run-dir name (see config.variant_tag). Dry-run prints the resolved env + command.

  python3 analysis/ablation_worker.py --csv <csv> --run_id r000002 [--dry-run]
  python3 analysis/ablation_worker.py --csv <csv> --run_id r000002   # execute (on HPC)
"""
import argparse, csv, os, subprocess, sys

def load_row(csv_path, run_id):
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["run_id"] == run_id:
                return r
    sys.exit(f"run_id {run_id} not found in {csv_path}")

def resolve(row, study_root):
    regime = row["regime"]                       # normal | m1
    norm   = row["norm"]                         # none | l2 | layernorm
    readout = row["readout"]                     # add | mean
    enc    = row["node_encoder"]                 # onehot | linear
    frag   = row["fragmentation"]                # rbrics | rdkit_fg_first | NA
    filt   = row["filter"]                       # filtered | unfiltered | NA
    unk    = row["unk"]                          # fixed | learn | NA
    ct     = row["celltype"]                     # vtrain | mose | motifsat | gsat | posthoc

    # --- env knobs (explicit for every run) ---
    env = {
        "OUT_ROOT_SUFFIX": regime,               # OUT_ROOT = <study_root>/<regime>
        "CONV_NORMALIZE": norm,                  # vanilla/motifsat/gsat
        "MOSE_CONV_NORMALIZE": norm,             # mose
        "GRAPH_POOL": readout,                   # all trainers
        "NODE_ENCODER": enc,                     # all trainers
        "REAL_ONLY": "1",                        # source-GT: original-label training
    }
    # M1 = class-balanced sampler (train loader only); Normal leaves it unset.
    # M1 also raises early-stop patience to 100 (balanced resampling converges
    # more slowly/noisily); normal runs leave PATIENCE unset so every trainer
    # keeps its config default (MoSE 30, MotifSAT/GSAT 20, vanilla 20) and stays
    # comparable to the already-finished experiments.
    if regime == "m1":
        env["BALANCED_SAMPLER"] = "1"
        env["PATIENCE"] = "100"
    # MoSE-only knobs
    if ct == "mose":
        env["MOSE_UNK_MODE"] = "learnable_shared" if unk == "learn" else "fixed"
        env["MOSE_FILTER_MODE"] = "filtered" if filt == "filtered" else "base"

    # --- variant (VOCAB_FOCUS) passed to run_cell ---
    #   mose/motifsat/posthoc -> the row's fragmentation (base token; MOSE_FILTER_MODE
    #       decides filtered vs base). vtrain/gsat are fragmentation-agnostic in training
    #       but the vanilla/gsat checkpoint is stored per-variant, so train under BOTH
    #       base frags in one cell so post-hoc (per frag) finds a checkpoint.
    if ct in ("mose", "motifsat", "posthoc"):
        variant = frag
    else:  # vtrain, gsat
        variant = "rbrics,rdkit_fg_first"

    tier = "source"                              # source-GT + mutag route via the real-label phases
    out_root = f"{study_root.rstrip('/')}/{regime}"
    return env, out_root, dict(ds=row["dataset"], variant=variant, bb=row["backbone"],
                               fold=row["fold"], tier=tier, celltype=ct)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--study-root", default="final_v2/ablation_srcgt")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    row = load_row(a.csv, a.run_id)
    env, out_root, cell = resolve(row, a.study_root)
    env["OUT_ROOT"] = out_root
    cmd = ["bash", "parallel/run_cell.sh", cell["ds"], cell["variant"],
           cell["bb"], cell["fold"], cell["tier"], cell["celltype"]]

    if a.dry_run:
        print(f"# run_id={a.run_id}  celltype={cell['celltype']}  regime={row['regime']}  arch_id={row['arch_id']}")
        print("ENV:")
        for k in ("OUT_ROOT", "CONV_NORMALIZE", "MOSE_CONV_NORMALIZE", "GRAPH_POOL",
                  "NODE_ENCODER", "MOSE_UNK_MODE", "MOSE_FILTER_MODE", "BALANCED_SAMPLER",
                  "PATIENCE", "REAL_ONLY"):
            if k in env:
                print(f"    {k}={env[k]}")
        print("CMD: " + " ".join(cmd))
        return
    full = dict(os.environ); full.update(env)
    sys.exit(subprocess.call(cmd, env=full))

if __name__ == "__main__":
    main()
