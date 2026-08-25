#!/usr/bin/env python3
"""Does the mask make the (already weak) self-identity signal WEAKER than no mask?
Paired MoSE-vs-Vanilla. For the SAME atoms (grouped LOW/HIGH by MoSE's gate
quartiles), probe self-identity in each model's OWN L2-normalized embedding
(decoder trained on middle atoms). Difference-in-differences:
  DiD = (MoSE_high - MoSE_low) - (Van_high - Van_low)  > 0  => mask suppresses LOW.
Eval only.

SELF-CONTAINED: the model/data-loading helpers (``_load_model_and_data``,
``_load_vanilla_and_data``, ``_node_emb_and_att``, ``_atom_class`` and their transitive
closure) are inlined below, so this is the single file for the probe work. This file
must live in ``<repo>/analysis/``
because ``_REPO`` is resolved as ``__file__.parent.parent`` (the repo root), which the
trainer-path juggling and ``SharedModules`` imports depend on.
"""
from __future__ import annotations

import os, glob, argparse
import json
import sys
from pathlib import Path

import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

# Make the repo root importable when run from elsewhere. We deliberately do NOT
# add MOSE-GNN/ and MotifSAT/ to sys.path here: both define a top-level
# `model.py`, so adding both would shadow each other; the loader below prepends
# the right trainer dir per run.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

BASE = Path("/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers inlined VERBATIM from analysis/probe_masked_nodes.py (self-contained).
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _node_emb_and_att(model, data, device, gated: bool):
    """Return (node_emb [N,H], node_att [N], x [N,F]) for one graph.

    gated=True uses the model's own attention injection; gated=False forces a
    plain (un-gated) embedding as a control.
    """
    model.eval()
    data = data.to(device)
    batch = getattr(data, 'batch', None)
    if batch is None:
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    # Obtain node_att from the model where available.
    node_att = None
    try:
        out = model(data.x, data.edge_index, batch,
                    getattr(data, 'nodes_to_motifs', None),
                    getattr(data, 'edge_attr', None))
        # MotifSAT returns (logits, node_att, aux); MOSE returns (logits, node_att)
        if isinstance(out, (tuple, list)) and len(out) >= 2 and out[1] is not None:
            node_att = out[1].view(-1)
            if len(out) >= 3 and isinstance(out[2], dict) and \
                    out[2].get('node_att_soft') is not None:
                node_att = out[2]['node_att_soft'].view(-1)
    except Exception:
        pass

    # Embedding backbone: MOSE exposes it at .backbone; GSAT/MotifSAT (GSAT class)
    # expose the classifier GNN at .clf. Both are BaseGNN with get_embedding.
    backbone = model
    for _attr in ('backbone', 'clf'):
        _cand = getattr(model, _attr, None)
        if _cand is not None and hasattr(_cand, 'get_embedding'):
            backbone = _cand
            break
    if not hasattr(backbone, 'get_embedding'):
        return None, None, None

    if gated and node_att is not None:
        w_feat = bool(getattr(model, 'w_feat', False))
        w_readout = bool(getattr(model, 'w_readout', False))
        w_message = bool(getattr(model, 'w_message', False))
        _, node_emb = backbone.get_embedding(
            data.x, data.edge_index, node_att=node_att.unsqueeze(-1),
            w_feat=w_feat, w_message=w_message, w_readout=w_readout, batch=batch)
    else:
        _, node_emb = backbone.get_embedding(
            data.x, data.edge_index, batch=batch)

    if node_att is None:
        node_att = torch.ones(data.x.size(0), device=device)
    return node_emb.cpu().numpy(), node_att.cpu().numpy(), data.x.cpu().numpy()


def _atom_class(x_rows: np.ndarray) -> np.ndarray:
    """Map one-hot (or multi-hot) node features to an integer class = argmax."""
    return x_rows.argmax(axis=1)


# Run-discovery helper (used by run_multi_explanation.py; inlined here so that file
# can import it from this single probe module instead of probe_masked_nodes).
_PROBE_PATH_MARKERS = ('mose', 'motifsat', 'gsat', 'base_gsat')


def _is_probeable_run(summary_path) -> bool:
    s = str(summary_path).lower()
    return any(m in s for m in _PROBE_PATH_MARKERS)


def _purge_trainer_modules() -> None:
    """Drop cached MOSE-GNN / MotifSAT top-level modules (both use ``run``/``config``)."""
    for name in ('run', 'config', 'model', 'train', 'reg_config', 'losses',
                 'motif_modules'):
        sys.modules.pop(name, None)


def _prepend_trainer_path(trainer_dir: Path) -> None:
    for p in (str(_REPO), str(trainer_dir)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def _meta_float(meta: dict, key: str, default: float = 0.0) -> float:
    """Read a float from summary.json; treat explicit JSON null as missing."""
    val = meta.get(key)
    if val is None:
        return default
    return float(val)


def _infer_hidden_dim(meta: dict, ckpt: Path) -> int:
    hidden_dim = meta.get('hidden_dim')
    if hidden_dim is None:
        try:
            _sd = torch.load(ckpt, map_location='cpu', weights_only=False)
            _sd = _sd.get('model_state_dict', _sd) if isinstance(_sd, dict) else _sd
            for _k in ('backbone.lin2.weight', 'backbone.lin1.weight',
                       'gnn.lin2.weight', 'gnn.lin1.weight'):
                if _k in _sd:
                    return int(_sd[_k].shape[1])
        except Exception:
            pass
    return int(hidden_dim or 32)


def _resolve_probe_family(meta: dict, run_dir: Path) -> str | None:
    """Return ``mose`` | ``motifsat`` or None if this run is not probeable."""
    model_type = (meta.get('model_type') or '').lower()
    motif_method = (meta.get('motif_method') or '').lower()
    run_s = str(run_dir).lower()
    if 'mose' in model_type or motif_method == 'mose':
        return 'mose'
    if 'motifsat' in model_type or motif_method in ('readout', 'loss'):
        return 'motifsat'
    if motif_method == 'none' or 'gsat' in run_s or 'base_gsat' in run_s:
        # Base GSAT uses edge attention, not node-att masking — skip below.
        if meta.get('learn_edge_att'):
            return None
        return 'motifsat'
    return None


def _common_cfg_kwargs(meta: dict, data_root: str, vocab_root: str,
                       hidden_dim: int) -> dict:
    return dict(
        dataset=meta['dataset'], fold=int(meta.get('fold', 0)),
        backbone=meta.get('backbone', 'GIN'),
        node_encoder=meta.get('node_encoder', 'onehot'),
        hidden_dim=hidden_dim,
        num_layers=int(meta.get('num_layers', 3)),
        vocab_variant=meta.get('vocab_variant', 'all_fallback_bpe'),
        conv_normalize=meta.get('conv_normalize', 'none'),
        gin_inner_bn=bool(meta.get('gin_inner_bn', True)),
        apply_layer_norm=bool(meta.get('apply_layer_norm', False)),
        data_root=meta.get('data_root', data_root),
        vocab_root=vocab_root,
        processed_root=meta.get('processed_root'),
        w_feat=bool(meta.get('w_feat', False)),
        w_message=bool(meta.get('w_message', False)),
        w_readout=bool(meta.get('w_readout', False)),
        mutag_index_maps_path=meta.get('mutag_index_maps_path'),
        mutag_smiles_csv_path=meta.get('mutag_smiles_csv_path'),
        mutag_splits_path=meta.get('mutag_splits_path'),
        mutag_seed=int(meta.get('mutag_seed') or 42),
        gnn_lr=_meta_float(meta, 'gnn_lr', 0.001),
        explainer_lr=_meta_float(meta, 'explainer_lr', 0.01),
        ent_reg=_meta_float(meta, 'ent_reg', 0.01),
        size_reg=_meta_float(meta, 'size_reg', 0.0),
        unk_mode=meta.get('unk_mode') or 'fixed',
    )


def _load_test_list(cfg, vocab_root: str):
    from SharedModules.data.dataset_routing import (
        default_processed_base,
        variant_processed_root,
    )

    from SharedModules.data.vocab import load_vocab
    from SharedModules.data.loader import get_loaders, TASK_TYPE

    vocab = load_vocab(cfg.vocab_root, cfg.dataset, cfg.vocab_variant)
    task_type = TASK_TYPE.get(cfg.dataset, 'BinaryClass')
    proc_root = cfg.processed_root or variant_processed_root(
        default_processed_base(cfg.data_root, None), cfg.vocab_variant)
    loaders, test_ds, dmeta = get_loaders(
        dataset=cfg.dataset, data_root=cfg.data_root, fold=cfg.fold,
        vocab=vocab, processed_root=proc_root,
        batch_size=cfg.batch_size,
        normalize=(task_type == 'Regression'),
        mutag_index_maps_path=getattr(cfg, 'mutag_index_maps_path', None),
        mutag_smiles_csv_path=getattr(cfg, 'mutag_smiles_csv_path', None),
        mutag_splits_path=getattr(cfg, 'mutag_splits_path', None),
        mutag_seed=getattr(cfg, 'mutag_seed', 42),
    )
    test_list = [g for g in loaders['test'].dataset]
    return test_list, task_type, dmeta, vocab


def _apply_injection_flags(model, meta: dict) -> None:
    for attr in ('w_feat', 'w_message', 'w_readout'):
        if hasattr(model, attr):
            setattr(model, attr, bool(meta.get(attr, getattr(model, attr, False))))


def _load_model_and_data(run_dir: Path, data_root: str, vocab_root: str,
                         device):
    """Load a trained MOSE or MotifSAT model + test data for a run directory."""
    sj = run_dir / 'summary.json'
    if not sj.exists():
        return None, None, 'no summary.json'
    with open(sj, encoding='utf-8') as f:
        meta = json.load(f)
    ckpt = run_dir / 'best_model.pt'
    if not ckpt.exists():
        cands = list(run_dir.glob('*.pt'))
        if not cands:
            return None, None, 'no checkpoint .pt'
        ckpt = cands[0]

    family = _resolve_probe_family(meta, run_dir)
    if family is None:
        mm = meta.get('motif_method') or meta.get('model_type') or '?'
        if meta.get('learn_edge_att'):
            return None, None, 'learn_edge_att GSAT has no node attention to probe'
        return None, None, f'not a probeable MOSE/MotifSAT run (got {mm})'

    hidden_dim = _infer_hidden_dim(meta, ckpt)
    cfg_kwargs = _common_cfg_kwargs(meta, data_root, vocab_root, hidden_dim)

    try:
        import importlib

        if family == 'mose':
            _purge_trainer_modules()
            _prepend_trainer_path(_REPO / 'MOSE-GNN')
            from config import MOSEConfig
            mose_run = importlib.import_module('run')
            cfg = MOSEConfig(**{k: v for k, v in cfg_kwargs.items()
                                if k in MOSEConfig.__dataclass_fields__
                                and v is not None})
            test_list, task_type, dmeta, vocab = _load_test_list(cfg, vocab_root)
            # Mirror MOSE run.py:134-137 — the model's motif_params is sized by the
            # FOLD's kept_motif_ids (filtered vocab), not the total motif count. Without
            # this a *_filter checkpoint (kept<<total) fails to load with a size mismatch.
            _fold_kept = getattr(dmeta, 'kept_motif_ids', None)
            if _fold_kept is None:
                raise ValueError("No fallback permitted in loading vocab")
            _kept_ids = _fold_kept
            model = mose_run.build_model(cfg, vocab.num_motifs, task_type, dmeta,
                                         kept_motif_ids=_kept_ids)
        else:
            _purge_trainer_modules()
            _prepend_trainer_path(_REPO / 'MotifSAT')
            from config import MotifSATConfig
            motifsat_run = importlib.import_module('run')
            ms_kwargs = {
                **cfg_kwargs,
                'motif_method': meta.get('motif_method') or 'readout',
                'noise': meta.get('noise') or 'none',
                'info_loss_level': meta.get('info_loss_level') or 'none',
                'info_loss_coef': _meta_float(meta, 'info_loss_coef', 0.0),
                'motif_loss_coef': _meta_float(meta, 'motif_loss_coef', 0.0),
                'within_node_coef': _meta_float(meta, 'within_node_coef', 0.0),
                'between_motif_coef': _meta_float(meta, 'between_motif_coef', 0.0),
                'init_r': meta.get('init_r'),
                'final_r': meta.get('final_r'),
                'decay_interval': meta.get('decay_interval'),
                'decay_r': meta.get('decay_r'),
                'learn_edge_att': bool(meta.get('learn_edge_att', False)),
            }
            cfg = MotifSATConfig(**{k: v for k, v in ms_kwargs.items()
                                    if k in MotifSATConfig.__dataclass_fields__
                                    and v is not None})
            test_list, task_type, dmeta, _vocab = _load_test_list(cfg, vocab_root)
            model = motifsat_run.build_model(cfg, task_type, dmeta)

        _apply_injection_flags(model, meta)
    except Exception as e:  # pragma: no cover - environment-specific
        return None, None, f'rebuild failed: {e}'

    try:
        state = torch.load(ckpt, map_location=device, weights_only=False)
        state = state.get('model_state_dict', state) if isinstance(state, dict) else state
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        if not test_list:
            return None, None, 'no test data'
        return model, test_list, 'ok'
    except Exception as e:  # pragma: no cover
        return None, None, f'load/data failed: {e}'


def _load_vanilla_and_data(run_dir: Path, data_root: str, vocab_root: str, device):
    """Build a VanillaGNN from summary.json, load its checkpoint + test data.

    Used for the post-hoc node-mask probe. Uses dmeta.deg / dmeta.x_dim so PNA
    (which needs the degree histogram) loads correctly alongside GIN/GCN/SAGE/GAT.
    """
    from types import SimpleNamespace
    sj = Path(run_dir) / 'summary.json'
    if not sj.exists():
        return None, None, 'no summary.json'
    meta = json.load(open(sj, encoding='utf-8'))
    ckpt = Path(run_dir) / 'best_model.pt'
    if not ckpt.exists():
        cands = list(Path(run_dir).glob('*.pt'))
        if not cands:
            return None, None, 'no checkpoint .pt'
        ckpt = cands[0]

    cfg = SimpleNamespace(
        dataset=meta['dataset'], fold=int(meta.get('fold', 0)),
        vocab_variant=meta.get('vocab_variant', 'all_fallback_bpe'),
        vocab_root=vocab_root, data_root=meta.get('data_root', data_root),
        processed_root=meta.get('processed_root'), batch_size=128,
        mutag_index_maps_path=meta.get('mutag_index_maps_path'),
        mutag_smiles_csv_path=meta.get('mutag_smiles_csv_path'),
        mutag_splits_path=meta.get('mutag_splits_path'),
        mutag_seed=int(meta.get('mutag_seed') or 42),
    )
    try:
        test_list, _task_type, dmeta, _vocab = _load_test_list(cfg, vocab_root)
    except Exception as e:  # pragma: no cover - environment-specific
        return None, None, f'data load failed: {e}'
    if not test_list:
        return None, None, 'no test data'
    try:
        from SharedModules.baselines.vanilla_gnn import VanillaGNN
        from SharedModules.data.loader import NUM_CLASSES, resolve_node_encoder
        nenc = resolve_node_encoder(meta.get('node_encoder'),
                                    getattr(dmeta, 'node_encoder', 'onehot'))
        model = VanillaGNN(
            x_dim=int(getattr(dmeta, 'x_dim', test_list[0].x.shape[1])),
            hidden_dim=int(meta.get('hidden_dim', 64)),
            num_layers=int(meta.get('num_layers', 3)),
            backbone=meta.get('backbone', 'GIN'), node_encoder=nenc,
            apply_layer_norm=bool(meta.get('apply_layer_norm', False)),
            dropout=float(meta.get('dropout', 0.5)),
            conv_normalize=meta.get('conv_normalize', 'none'),
            gin_inner_bn=bool(meta.get('gin_inner_bn', True)),
            num_classes=NUM_CLASSES.get(meta['dataset'], 1),
            deg=getattr(dmeta, 'deg', None))
        state = torch.load(ckpt, map_location=device, weights_only=False)
        state = state.get('model_state_dict', state) if isinstance(state, dict) else state
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
    except Exception as e:  # pragma: no cover
        return None, None, f'vanilla rebuild failed: {e}'
    return model, test_list, 'ok'


# ─────────────────────────────────────────────────────────────────────────────
# probe_vs_vanilla logic
# ─────────────────────────────────────────────────────────────────────────────
def run_dirs(ds):
    return sorted(Path(r) for r in glob.glob(str(BASE / "final_v2/mose/rbrics_filter" / ds / "fold*" / "*"))
                  if "_real" in os.path.basename(r) and (Path(r) / "summary.json").exists())


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X / n


@torch.no_grad()
def vanilla_emb(van, d, device):
    dd = d.to(device)
    batch = torch.zeros(dd.x.size(0), dtype=torch.long, device=device)
    return van.get_emb(dd.x, dd.edge_index, batch=batch,
                       edge_attr=getattr(dd, "edge_attr", None)).cpu().numpy()


def probe_hilo(E, Y, low, high, mid):
    EN = l2(E)
    clf = LogisticRegression(max_iter=300).fit(EN[mid], Y[mid])
    tc = set(np.unique(Y[mid]).tolist())
    def ev(msk):
        m = msk & np.isin(Y, list(tc))
        return balanced_accuracy_score(Y[m], clf.predict(EN[m])) if m.sum() >= 30 else np.nan
    return ev(high), ev(low)


def process(mose_dir, data_root, vocab_root, device, max_graphs=800):
    ds = mose_dir.parts[mose_dir.parts.index("rbrics_filter") + 1]
    fold = mose_dir.parent.name
    bb = mose_dir.name.split("_")[0]
    mose, tl, st = _load_model_and_data(mose_dir, data_root, vocab_root, device)
    if st != "ok":
        return None, "mose " + st
    vdir = BASE / "final_v2/vanilla" / ds / fold / "rbrics" / f"bb-{bb}_enc-onehot_norm-none_real"
    van, _tl2, st2 = _load_vanilla_and_data(vdir, data_root, vocab_root, device)
    if st2 != "ok":
        return None, "vanilla " + st2
    Emo, Eva, GA, Y = [], [], [], []
    for d in tl[:max_graphs]:
        ne, na, x = _node_emb_and_att(mose, d, device, gated=True)
        if ne is None:
            continue
        try:
            ve = vanilla_emb(van, d, device)
        except Exception:
            continue
        if len(ve) != len(ne):
            continue
        Emo.append(ne * na[:, None]); Eva.append(ve); GA.append(na); Y.append(_atom_class(x))
    if not Emo:
        return None, "no aligned reps"
    Emo = np.concatenate(Emo); Eva = np.concatenate(Eva); GA = np.concatenate(GA); Y = np.concatenate(Y)
    # ABSOLUTE gate thresholds: LOW = genuinely masked (<0.1), HIGH = genuinely
    # important (>0.9). Skip a run if either extreme is (near-)empty -- i.e. the
    # backbone never gates that far (PNA/GIN). That skip is itself informative.
    low = GA < 0.1; high = GA > 0.9; mid = (~low) & (~high)
    if low.sum() < 30 or high.sum() < 30 or mid.sum() < 100 or len(np.unique(Y[mid])) < 2:
        return None, f"no extremes (n_low={int(low.sum())} n_high={int(high.sum())})"
    mo_h, mo_l = probe_hilo(Emo, Y, low, high, mid)
    va_h, va_l = probe_hilo(Eva, Y, low, high, mid)
    n = len(GA)
    return dict(mose_high=mo_h, mose_low=mo_l, van_high=va_h, van_low=va_l,
                mose_gap=mo_h - mo_l, van_gap=va_h - va_l,
                did=(mo_h - mo_l) - (va_h - va_l),
                n_atoms=int(n), n_low=int(low.sum()), n_high=int(high.sum()),
                frac_low=float(low.mean()), frac_high=float(high.mean()),
                gate_min=float(GA.min()), gate_med=float(np.median(GA))), "ok"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--data_root", required=True)
    ap.add_argument("--vocab_root", required=True); ap.add_argument("--max_graphs", type=int, default=800)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for rd in run_dirs(a.dataset):
        row, st = process(rd, a.data_root, a.vocab_root, device, a.max_graphs)
        bb = rd.name.split("_")[0]; fold = rd.parent.name
        if row is None:
            print(f"  [skip] {bb} {fold}: {st}"); continue
        row.update(dataset=a.dataset, backbone=bb, fold=fold)
        rows.append(row)
        print(f"  [{bb} {fold}] MoSE h/l={row['mose_high']:.2f}/{row['mose_low']:.2f} "
              f"Van h/l={row['van_high']:.2f}/{row['van_low']:.2f} | DiD={row['did']:+.2f}")
    if rows:
        out = Path(a.save) if a.save else BASE / "final_v2" / f"probe_vsvanilla_{a.dataset}.csv"
        pd.DataFrame(rows).to_csv(out, index=False); print("wrote", out, len(rows))
    else:
        print("no runs")
