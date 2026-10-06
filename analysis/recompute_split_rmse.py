#!/usr/bin/env python3
"""recompute_split_rmse.py — APPEND train/val RMSE to base final_v2 regression summary.json.

The base final_v2 / final_v2_gath1 regression runs stored only the TEST metric (bare
`rmse`/`rmse_orig`); they predate `--per_split_eval`, so train/val RMSE are absent. This computes
them by re-running each run's OWN trainer in EVAL-ONLY mode (loads best_model.pt, skips training)
into a SCRATCH side directory with `--per_split_eval`, which produces summary_splits.json with
train/valid/test rmse/rmse_orig via the exact tested eval + per-fold de-standardization.

APPEND-ONLY SAFETY (this is the whole point):
- The eval-only run writes to a SCRATCH dir (<DISP>/side/<cell>), NEVER the original run dir.
- The ORIGINAL run dir is touched in exactly ONE way: we load its summary.json, ADD four keys
  (train_rmse, val_rmse, train_rmse_orig, val_rmse_orig) if absent, and write it back. Every
  existing key is preserved byte-for-identical (we never rebuild summary.json). best_model.pt and
  all other files in the run dir are untouched; no new CSVs are added there.
- CONFIRM-TEST guard: the scratch test rmse_orig must equal the original stored value (|Δ|<=TOL)
  before we append; otherwise the run is logged to failures.tsv and NOT modified.
- Idempotent: runs that already have train_rmse are skipped. Claim-based (atomic mkdir).
- DRY_RUN=1 (default): print the plan, change nothing.

ENV: DRY_RUN (default 1), POOL_VOCABS (default all 8), TOL (default 1e-3), KEEP_SIDE=1 to keep scratch.
Run on HPC (needs the model filesystem + l2xgnn env).
"""
import glob, json, os, shutil, subprocess, sys

REPO = "/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor"
VOCAB_ROOT = REPO + "/vocab_final_v2"
FOLDS = "/nfs/hpc/share/kokatea/ChemIntuit/MotifBreakdown/datasets/FOLDS"
REG_DS = ("esol", "Lipophilicity")
VOCABS_ALL = ["rbrics", "rbrics_filter", "ertl_first", "ertl_first_filter",
              "rdkit_fg_first", "rdkit_fg_first_filter", "fg_first", "fg_first_filter"]
TREES = ["final_v2", "final_v2_gath1"]
NEW_KEYS = ("train_rmse", "val_rmse", "train_rmse_orig", "val_rmse_orig")
DRY = os.environ.get("DRY_RUN", "1") == "1"
TOL = float(os.environ.get("TOL", "1e-3"))
KEEP_SIDE = os.environ.get("KEEP_SIDE", "0") == "1"
VOCABS = os.environ.get("POOL_VOCABS", " ".join(VOCABS_ALL)).split()
DISP = REPO + "/_dispatch_recompute_rmse"
CLAIMS, SIDE, FAIL = DISP + "/claims", DISP + "/side", DISP + "/failures.tsv"
os.makedirs(CLAIMS, exist_ok=True); os.makedirs(SIDE, exist_ok=True)
if not os.path.exists(FAIL):
    open(FAIL, "w").write("ts\tcell\treason\n")


def trainer_cmd(run_dir, s, side_out):
    """Eval-only command writing to side_out (scratch), loading weights from run_dir."""
    fam = s.get("family"); ds = s["dataset"]; fold = s["fold"]; bb = s["backbone"]
    vv = s.get("vocab_variant"); enc = s.get("node_encoder", "onehot")
    conv = s.get("conv_normalize", "none"); pool = s.get("graph_pool", "add")
    L = s.get("num_layers", 3); H = s.get("hidden_dim", 64)
    proc = s.get("processed_root") or (REPO + "/processed_final_v2")
    common = ["--dataset", ds, "--fold", str(fold), "--backbone", bb,
              "--node_encoder", enc, "--conv_normalize", conv, "--graph_pool", pool,
              "--num_layers", str(L), "--hidden_dim", str(H),
              "--data_root", FOLDS, "--vocab_root", VOCAB_ROOT, "--vocab_variant", vv,
              "--processed_root", proc, "--out_dir", side_out,
              "--load_weights_from", run_dir, "--final_out_dir"]
    if fam == "vanilla":
        return [sys.executable, "SharedModules/baselines/run_vanilla.py", "--epochs", "0"] + common
    if fam == "mose":
        return ([sys.executable, "MOSE-GNN/run.py", "--eval_only", "--per_split_eval",
                 "--w_feat", "--w_readout", "--unk_mode", s.get("unk_mode", "fixed")] + common)
    if fam in ("gsat", "motifsat"):
        extra = ["--eval_only", "--per_split_eval",
                 "--motif_method", str(s.get("motif_method", "none")),
                 "--noise", str(s.get("noise", "none")),
                 "--info_loss_level", str(s.get("info_loss_level", "node"))]
        for flag, key in (("--w_feat", "w_feat"), ("--w_message", "w_message"), ("--w_readout", "w_readout")):
            if s.get(key): extra.append(flag)
        return [sys.executable, "MotifSAT/run.py"] + extra + common
    return None


def iter_runs():
    for tree in TREES:
        for vv in VOCABS:
            for ds in REG_DS:
                yield from glob.glob(f"{REPO}/{tree}/*/{vv}/{ds}/fold*/*/summary.json")       # mose/gsat/motifsat
                yield from glob.glob(f"{REPO}/{tree}/vanilla/{ds}/fold*/{vv}/*/summary.json")  # vanilla


def main():
    seen = done = skip = 0
    for sj in iter_runs():
        run_dir = os.path.dirname(sj)
        try:
            s = json.load(open(sj))
        except Exception:
            continue
        if s.get("dataset") not in REG_DS:
            continue
        if all(k in s and isinstance(s[k], (int, float)) for k in ("train_rmse", "val_rmse")):
            skip += 1; continue                     # already appended
        seen += 1
        cell = run_dir.replace(REPO + "/", "").replace("/", "__")
        side_out = os.path.join(SIDE, cell)
        cmd = trainer_cmd(run_dir, s, side_out)
        if cmd is None:
            continue
        stored = s.get("rmse_orig")
        if DRY:
            print("[dry]", cell, "::", " ".join(cmd)); done += 1; continue
        cdir = os.path.join(CLAIMS, cell)
        try:
            os.mkdir(cdir)
        except FileExistsError:
            skip += 1; continue
        os.makedirs(side_out, exist_ok=True)
        env = dict(os.environ, PYTHONPATH=REPO, WANDB_MODE="disabled")
        if s.get("backbone") == "GAT":
            env["MOSE_GAT_HEADS"] = "1"
        rc = subprocess.run(cmd, cwd=REPO, env=env).returncode
        # read per-split metrics from the SCRATCH summary_splits.json
        reason = None; add = {}
        try:
            sp = json.load(open(os.path.join(side_out, "summary_splits.json")))
            stem = next(iter(sp))                   # family stem key
            d = sp[stem]
            te = float(d["test"]["rmse_orig"])
            if stored is None or abs(te - float(stored)) > TOL:
                reason = f"test_rmse_drift(side={te:.5f},stored={stored})"
            else:
                add = {"train_rmse": d["train"]["rmse"], "val_rmse": d["valid"]["rmse"],
                       "train_rmse_orig": d["train"]["rmse_orig"], "val_rmse_orig": d["valid"]["rmse_orig"]}
        except Exception as e:
            reason = f"rc={rc}_or_no_splits({type(e).__name__})"
        if reason:
            open(FAIL, "a").write(f"-\t{cell}\t{reason}\n"); open(cdir + "/.failed", "w").write("")
        else:
            # APPEND-ONLY: load original, add 4 keys, write back; everything else preserved.
            cur = json.load(open(sj))
            cur.update(add)
            with open(sj, "w") as f:
                json.dump(cur, f, indent=2)
            done += 1
        if not KEEP_SIDE:
            shutil.rmtree(side_out, ignore_errors=True)
    print(f"DONE seen={seen} appended={done} skip={skip} dry={DRY} vocabs={VOCABS}")


if __name__ == "__main__":
    main()
