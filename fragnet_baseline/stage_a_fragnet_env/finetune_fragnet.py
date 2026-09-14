"""Stage A (2/3) — runs in the FragNet env. Generate a finetune YAML (mirroring
exps/ft/esol/e1pt4.yaml, with a classification head) and invoke FragNet's OWN
finetune_gat2.py from the pretrained pt.pt. FragNet stays unmodified; we only supply config.

Verified (fragnet/train/utils.py TrainerFineTune): target_type 'clsf' -> BCEWithLogitsLoss
(train_clsf_bce); 'regr' -> MSELoss. Benzene is binary -> 'clsf'.
Model dims MUST match pt.pt: atom/frag_features 167, edge_features 17, emb_dim 128,
num_layer 4, num_heads 4, fthead FTHead3 (from the esol config).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def build_config(work: Path, pt_ckpt: str, target_type: str, n_classes: int,
                 epochs: int, es_patience: int, lr: float, batch_size: int, seed: int = 42) -> dict:
    return {
        "exp_dir": str(work),
        "seed": seed,            # REQUIRED: finetune_gat2.py calls seed_everything(args.seed) (top-level key)
        "model_version": "gat2",
        "device": "gpu",
        "atom_features": 167, "frag_features": 167, "edge_features": 17,
        "fedge_in": 6, "fbond_edge_in": 6,
        "pretrain": {
            "model_version": "gat2", "num_layer": 4, "drop_ratio": 0.2, "num_heads": 4,
            "emb_dim": 128, "chkpoint_name": str(pt_ckpt), "loss": "mse",
            "batch_size": 128, "es_patience": 500, "lr": 1e-4, "n_epochs": 20000,
            "n_classes": n_classes,
        },
        "finetune": {
            "n_multi_task_heads": 0, "batch_size": batch_size, "lr": lr,
            "model": {"n_classes": n_classes, "num_layer": 4, "drop_ratio": 0.1,
                      "num_heads": 4, "emb_dim": 128, "h1": 128, "h2": 1024, "h3": 1024,
                      "h4": 512, "act": "relu", "fthead": "FTHead3"},
            "n_epochs": epochs, "target_type": target_type,
            "loss": ("bce" if target_type == "clsf" else "mse"),
            "use_schedular": False, "es_patience": es_patience,
            "chkpoint_name": str(work / "ft.pt"),
            "train": {"path": str(work / "train.pkl")},
            "val": {"path": str(work / "val.pkl")},
            "test": {"path": str(work / "test.pkl")},
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Stage A/2: finetune FragNet from pt.pt")
    ap.add_argument("--work", required=True, help="fold work dir (holds the pkls; ft.pt written here)")
    ap.add_argument("--pt_ckpt", required=True, help="path to pretrained pt.pt")
    ap.add_argument("--vendor", required=True, help="vendored pnnl/FragNet repo")
    ap.add_argument("--task", choices=["clf", "regr"], default="clf",
                    help="clf -> BCE head/sigmoid; regr -> MSE head/raw output (esol, Lipophilicity)")
    ap.add_argument("--n_classes", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--es_patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    work = Path(args.work)
    target_type = "clsf" if args.task == "clf" else "regr"
    for s in ("train", "val", "test"):
        if not (work / f"{s}.pkl").exists():
            raise FileNotFoundError(f"missing {work / f'{s}.pkl'} — run prep_data.py first")
    cfg = build_config(work, args.pt_ckpt, target_type, args.n_classes,
                       args.epochs, args.es_patience, args.lr, args.batch_size, seed=args.seed)
    cfg_path = work / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[finetune] config -> {cfg_path} (target_type={target_type})")

    cmd = [sys.executable, "-m", "fragnet.train.finetune.finetune_gat2", "--config", str(cfg_path)]
    print("[finetune] " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=args.vendor)
    if r.returncode != 0:
        raise RuntimeError(f"FragNet finetune_gat2 failed (rc={r.returncode})")
    if not (work / "ft.pt").exists():
        raise FileNotFoundError(
            f"finetune did not produce {work / 'ft.pt'} — check finetune.chkpoint_name handling.")
    print(f"[finetune] OK — {work / 'ft.pt'}")


if __name__ == "__main__":
    main()
