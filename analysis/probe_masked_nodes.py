#!/usr/bin/env python3
"""probe_masked_nodes.py — do masked nodes hide their input features?

Post-hoc probing experiment. Motif-aware models (MOSE / MotifSAT) down-weight
unimportant nodes via a soft attention gate (node_att). If that gating is doing
its job, the *embedding* of a heavily-masked node should retain little
recoverable information about that node's input features — a probe trained to
predict node features from embeddings should do markedly WORSE on masked nodes
than on unmasked ones.

Method
------
1. Load a trained model + its dataset (same loaders as training).
2. Run the model's get_embedding to obtain, for every node:
     - node_emb   [N, H]   the post-conv node embedding (what a readout sees)
     - node_att   [N]      the soft attention/gate weight in (0,1)
     - x          [N, F]   the input node features (one-hot atom type here)
3. Split nodes into MASKED (low att) vs UNMASKED (high att) by the att median
   (or a fixed threshold via --att_threshold).
4. Train a linear probe (logistic regression over atom-type classes) to predict
   the node's input feature/class from node_emb, fit on a train split of nodes
   and evaluated on a held-out split — separately for masked vs unmasked nodes.
5. Report probe accuracy on masked vs unmasked. A large gap
   (unmasked >> masked) is evidence the mask removes recoverable feature info.

Two embedding sources are compared:
   - 'gated'   : node_emb with the model's attention injection applied
                 (w_feat/w_readout as the model was trained) — the realistic case.
   - 'raw'     : node_emb with NO attention injection — control; both groups
                 should then be similarly recoverable (gap ~ 0). The contrast
                 between 'gated' and 'raw' isolates the masking effect.

Usage
-----
    python analysis/probe_masked_nodes.py \
        --run_dir results/mose/all_fallback_bpe/Mutagenicity/fold0/<tag> \
        --data_root $DATA_ROOT --vocab_root $VOCAB_ROOT \
        [--att_threshold 0.5] [--max_graphs 500]

Or point --out_root at a tree and it probes MOSE / MotifSAT readout runs:
    python analysis/probe_masked_nodes.py --out_root results --data_root ... --vocab_root ...

Writes a CSV summary (one row per run) to --save (default: masked_node_probe.csv).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Make the repo root importable when run from elsewhere. We deliberately do NOT
# add MOSE-GNN/ and MotifSAT/ to sys.path here: both define a top-level
# `model.py`, so adding both would shadow each other. The caller (notebook or
# run script) imports the right model module.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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


def _probe_accuracy(emb: np.ndarray, y: np.ndarray, seed: int = 0):
    """Fit a linear probe emb->y with a train/test node split; return test acc."""
    if len(np.unique(y)) < 2 or len(y) < 20:
        return float('nan'), len(y)
    Xtr, Xte, ytr, yte = train_test_split(
        emb, y, test_size=0.3, random_state=seed, stratify=None)
    if len(np.unique(ytr)) < 2:
        return float('nan'), len(y)
    clf = LogisticRegression(max_iter=200)
    try:
        clf.fit(Xtr, ytr)
        return float(accuracy_score(yte, clf.predict(Xte))), len(y)
    except Exception:
        return float('nan'), len(y)


def probe_run(model, data_list, device, att_threshold=None, max_graphs=500,
              seed=0):
    """Collect node embeddings/atts over graphs and probe masked vs unmasked."""
    results = {}
    for gated in (True, False):
        embs, atts, ys = [], [], []
        for d in data_list[:max_graphs]:
            ne, na, x = _node_emb_and_att(model, d, device, gated=gated)
            if ne is None:
                continue
            embs.append(ne); atts.append(na); ys.append(_atom_class(x))
        if not embs:
            continue
        E = np.concatenate(embs); A = np.concatenate(atts); Y = np.concatenate(ys)
        thr = att_threshold if att_threshold is not None else float(np.median(A))
        masked = A <= thr
        unmasked = ~masked
        acc_m, n_m = _probe_accuracy(E[masked], Y[masked], seed)
        acc_u, n_u = _probe_accuracy(E[unmasked], Y[unmasked], seed)
        tag = 'gated' if gated else 'raw'
        results[f'{tag}_acc_masked'] = acc_m
        results[f'{tag}_acc_unmasked'] = acc_u
        results[f'{tag}_gap_unmasked_minus_masked'] = (
            acc_u - acc_m if (acc_u == acc_u and acc_m == acc_m) else float('nan'))
        results[f'{tag}_n_masked'] = n_m
        results[f'{tag}_n_unmasked'] = n_u
        results[f'{tag}_att_threshold'] = thr
    return results


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
        conv_normalize=meta.get('conv_normalize', 'l2'),
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
            model = mose_run.build_model(cfg, vocab.num_motifs, task_type, dmeta)
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


_PROBE_PATH_MARKERS = ('mose', 'motifsat', 'gsat', 'base_gsat')


def _is_probeable_run(summary_path: Path) -> bool:
    s = str(summary_path).lower()
    return any(m in s for m in _PROBE_PATH_MARKERS)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', default=None,
                    help='A single run directory (with summary.json + .pt).')
    ap.add_argument('--out_root', default=None,
                    help='Probe MOSE / MotifSAT readout runs under this tree.')
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--att_threshold', type=float, default=None,
                    help='Fixed att cutoff for masked vs unmasked (default: per-run median).')
    ap.add_argument('--max_graphs', type=int, default=500)
    ap.add_argument('--save', default='masked_node_probe.csv')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dataset', nargs='*', default=None,
                    help='only probe runs for these dataset(s), e.g. --dataset mutag')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = set(args.dataset) if args.dataset else None

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    elif args.out_root:
        # iter_summaries skips _archive/_trash/_old and applies the dataset filter.
        from analysis.aggregate_experiments import iter_summaries
        run_dirs = [p.parent for p in iter_summaries(Path(args.out_root), datasets=datasets)
                    if _is_probeable_run(p)]
    else:
        raise SystemExit('provide --run_dir or --out_root')

    rows = []
    for rd in run_dirs:
        model, test_list, status = _load_model_and_data(
            rd, args.data_root, args.vocab_root, device)
        if status != 'ok':
            print(f'  [skip] {rd}: {status}')
            continue
        res = probe_run(model, test_list, device,
                        att_threshold=args.att_threshold,
                        max_graphs=args.max_graphs, seed=args.seed)
        res['run_dir'] = str(rd)
        gated_gap = res.get('gated_gap_unmasked_minus_masked')
        raw_gap = res.get('raw_gap_unmasked_minus_masked')
        if gated_gap is None and raw_gap is None:
            # No probe data extracted (e.g. no node embeddings for this model /
            # split). Report honestly instead of crashing on a None format.
            print(f'  [probe] {rd.name}: no probe data extracted (skipped)')
            continue
        rows.append(res)
        _fmt = lambda v: f'{v:.4f}' if isinstance(v, (int, float)) else 'n/a'
        print(f'  [probe] {rd.name}: '
              f'gated_gap={_fmt(gated_gap)} raw_gap={_fmt(raw_gap)}')

    if rows:
        import pandas as pd
        out = (Path(args.out_root) / args.save if args.out_root
               else Path(args.run_dir) / args.save)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f'wrote {out} ({len(rows)} runs)')
        print('Interpretation: gated_gap > raw_gap means the attention gate '
              'removes recoverable input-feature info from masked nodes — '
              'i.e. masking genuinely makes node features harder to recover.')
    else:
        print('No MOSE/MotifSAT runs successfully probed.')


if __name__ == '__main__':
    main()
