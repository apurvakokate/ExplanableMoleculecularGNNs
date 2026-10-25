#!/usr/bin/env bash
#SBATCH -J paper_vocab_check
#SBATCH -p share,preempt
#SBATCH --gres=gpu:0
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH --array=0-19
#SBATCH -o /nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/vocab_final_v3/_logs/paper_check_%A_%a.out
# paper_vocab_check.sh — READ-ONLY vocabulary checks + processed-cache build for the GSAT paper datasets.
# Task = (dataset, vocab, fold): mutag x {rbrics_v3, fmbpe_v4} x folds 0-4 (10) +
#        ogbg-{molhiv,molbace,molbbbp,moltox21,molclintox} x {rbrics_v3, fmbpe_v4} x fold 0 (10).
# Vocab checks (unfiltered variant used for MotifSAT):
#   sidecar  : flattened _mol_fragment_copies == lookup_all for every molecule
#   copies   : every copy's atoms reproduce its key (heavy-atom graph monomorphism; H and [*] ignored,
#              because mutag SMILES carry explicit mapped H atoms). Mutag copy atoms are numbered in
#              the H-removed molecule (RemoveHs, unsanitized), as AttachCopies reads them.
#   keys     : no key carries atom-map numbers; no key is the whole-molecule SMILES of a molecule
#              that has more than the copy's atoms (detached salt ion / loose H keyed as the molecule)
#   coverage : every heavy atom of every molecule is in a recorded copy
#   mutag rbrics v3 vs v2 per-molecule key multisets (fold-0 vocab only)
# Loader: get_loaders with the vocab -> processed_paper_v1/<vocab>/...; iterate every split; each graph
#   must carry nodes_to_copy, with copy >= 0 exactly where nodes_to_motifs >= 0.
# Submitted via sbatch stdin; this file is the record.
set -uo pipefail
source /nfs/stak/users/kokatea/hpc-share/anaconda3/etc/profile.d/conda.sh; conda activate l2xgnn
R=/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor; cd $R
export PYTHONPATH=$R WANDB_MODE=disabled OMP_NUM_THREADS=2 OGB_DATA_ROOT=/nfs/hpc/share/kokatea/ChemIntuit/GSAT_upstream_6812f5c/data
python3 - ${SLURM_ARRAY_TASK_ID} <<'PY'
import pickle, sys, json, re
from collections import Counter
import networkx as nx, torch
from networkx.algorithms import isomorphism as iso
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
R = '/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor'
OGB = '/nfs/hpc/share/kokatea/ChemIntuit/GSAT_upstream_6812f5c/data'
i = int(sys.argv[1])
if i < 10:
    ds, vocab, fold = 'mutag', ('rbrics_v3', 'fmbpe_v4')[i // 5], i % 5
else:
    j = i - 10
    ds, vocab, fold = ['ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-moltox21', 'ogbg-molclintox'][j // 2], ('rbrics_v3', 'fmbpe_v4')[j % 2], 0
vroot, var = ((f'{R}/vocab_final_v3', 'rbrics') if vocab == 'rbrics_v3'
              else (f'{R}/fmbpe_v4/vocab/fold{fold}', 'fm_bpe_full_union'))
base = f'{vroot}/{ds}/{var}/{ds}_{var}_'
L = lambda s: pickle.load(open(base + s, 'rb'))
out = {'dataset': ds, 'vocab': vocab, 'fold': fold}
fails = []
check_vocab = vocab == 'fmbpe_v4' or fold == 0          # rBRICS vocab is fold-0 only
if check_vocab:
    cp, lk, ml = L('mol_fragment_copies.pickle'), L('lookup_all.pickle'), L('motif_list.pickle')
    side_bad = sum({a: (k, m) for k, m, at in cp[s] for a in at} != dict(lk[s]) for s in cp)

    def heavy(m, atoms):
        G = nx.Graph()
        for a in atoms:
            n = m.GetAtomWithIdx(a).GetAtomicNum()
            if n > 1: G.add_node(a, el=n)
        for b in m.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if u in G and v in G: G.add_edge(u, v)
        return G
    nm = iso.categorical_node_match('el', None)
    wrong = ncp = uncovered = rep = 0
    mapped_keys = {k for k in ml if re.search(r':[0-9]+\]', k)}
    whole_mol_keys = set()
    for s, copies in cp.items():
        if ds == 'mutag':
            mol = Chem.RemoveHs(Chem.MolFromSmiles(s, sanitize=False), sanitize=False)
        else:
            mol = Chem.MolFromSmiles(s)
            if mol is None: mol = Chem.MolFromSmiles(s, sanitize=False)
        covered = set()
        keys = Counter(k for k, _, _ in copies); rep += any(c > 1 for c in keys.values())
        for k, _, at in copies:
            ncp += 1; covered |= set(at)
            if '.' in k and len(at) < mol.GetNumAtoms(): whole_mol_keys.add(k)
            km = Chem.MolFromSmiles(k, sanitize=False)
            K, A = (heavy(km, range(km.GetNumAtoms())) if km is not None else None), heavy(mol, at)
            ok = (K is not None and A.number_of_nodes() == K.number_of_nodes()
                  and (A.number_of_nodes() == 0 or nx.is_connected(A))
                  and (nx.is_isomorphic(A, K, node_match=nm) or iso.GraphMatcher(A, K, node_match=nm).subgraph_is_monomorphic()))
            wrong += not ok
        uncovered += sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1 and a.GetIdx() not in covered)
    out.update(motifs=len(ml), molecules=len(cp), copies=ncp, wrong_copies=wrong, sidecar_mismatch=side_bad,
               uncovered_heavy_atoms=uncovered, mols_with_repeated_motif=rep,
               mapped_keys=len(mapped_keys), whole_molecule_keys=len(whole_mol_keys))
    if side_bad: fails.append('sidecar')
    if wrong: fails.append(f'wrong_copies={wrong}')
    if uncovered: fails.append(f'uncovered={uncovered}')
    if mapped_keys: fails.append(f'mapped_keys={len(mapped_keys)}')
    if whole_mol_keys: fails.append(f'whole_molecule_keys={len(whole_mol_keys)}')
    if ds == 'mutag' and vocab == 'rbrics_v3':
        o = pickle.load(open(f'{R}/vocab_final_v2/mutag/rbrics/mutag_rbrics_mol_fragment_smarts.pickle', 'rb'))
        n = L('mol_fragment_smarts.pickle')
        out['mutag_v2_vs_v3_key_diff_mols'] = sum(Counter(o[s]) != Counter(n.get(s, [])) for s in o)
# loader + processed cache + copy ids on real graphs
from SharedModules.data.vocab import load_vocab
from SharedModules.data.loader import get_loaders, TASK_TYPE
from SharedModules.data.dataset_routing import variant_processed_root
voc = load_vocab(vroot, ds, var)
proc = variant_processed_root(f'{R}/processed_paper_v1/{vocab}', var)
root = f'{R}/data' if ds == 'mutag' else OGB
loaders, _, meta = get_loaders(dataset=ds, data_root=root, fold=fold, vocab=voc, processed_root=proc, batch_size=256,
                               normalize=(TASK_TYPE.get(ds) == 'Regression'))
for split, ld in loaders.items():
    g = bad = unk = 0
    for b in ld:
        n2m, c = b.nodes_to_motifs.view(-1), b.nodes_to_copy.view(-1)
        bad += int(((n2m >= 0) != (c >= 0)).sum()); unk += int((n2m < 0).sum()); g += b.num_graphs
    out[f'{split}_graphs'] = g; out[f'{split}_copy_mismatch'] = bad; out[f'{split}_unk_atoms'] = unk
    if bad: fails.append(f'{split} copy/motif mismatch {bad}')
out['task_type'] = meta.task_type; out['num_classes'] = meta.num_classes
print(('OK   ' if not fails else 'FAIL ') + json.dumps(out) + ('' if not fails else '  ' + '; '.join(fails)))
PY
