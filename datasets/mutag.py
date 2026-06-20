# Vendored from Graph-COM/GSAT (ICML 2022), src/datasets/mutag.py
# https://github.com/Graph-COM/GSAT
#
# Raw files (PGExplainer Mutagenicity bundle):
#   https://github.com/flyingdoog/PGExplainer/tree/master/dataset
# Unzip Mutagenicity.zip + Mutagenicity.pkl.zip into:
#   {data_root}/mutag/raw/
#
# Each processed Data carries source ground-truth explanations:
#   node_label [N], edge_label [E]  (from Mutagenicity_edge_gt.txt)
#
# Label encoding (Mutagenicity_label_readme.txt):
#   y = 0  → mutagen
#   y = 1  → nonmutagen
# edge_gt: 1 = edge inside NO2/NH2 motif, 0 = outside.
#
# GT is kept only on mutagen graphs (y==0) with annotated motifs; non-mutagen
# graphs have edge_label zeroed.  Mutagen graphs with y==0 but no motif edges
# are dropped at process time.

from __future__ import annotations

import os
import pickle as pkl

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

# Atom-type one-hot width when building features from node_labels (no .pkl).
MUTAG_NUM_ATOM_TYPES = 14


def _onehot_from_node_types(node_types) -> torch.Tensor:
    """14-dim one-hot node features from Mutagenicity_node_labels.txt indices."""
    x = np.zeros((len(node_types), MUTAG_NUM_ATOM_TYPES), dtype=np.float32)
    for j, t in enumerate(node_types):
        ti = int(t)
        if 0 <= ti < MUTAG_NUM_ATOM_TYPES:
            x[j, ti] = 1.0
    return torch.from_numpy(x)


class Mutag(InMemoryDataset):
    """Mutagenicity (MUTAG) with PGExplainer / GSAT preprocessing.

    Class labels (``Mutagenicity_label_readme.txt``): ``y=0`` mutagen,
    ``y=1`` nonmutagen.  Source explanation GT (NO2/NH2 motif edges) is
    attached only for mutagen graphs.

    Node features: uses ``Mutagenicity.pkl`` (PGExplainer pre-baked 14-dim
    features) when present; otherwise builds 14-dim one-hot from
    ``Mutagenicity_node_labels.txt`` (same atom-type vocabulary).
    """

    def __init__(self, root: str, force_reload: bool = False):
        super().__init__(root=root, force_reload=force_reload)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # Text files are vendored in-repo; Mutagenicity.pkl (~6 GB) is optional.
        return [
            'Mutagenicity_A.txt',
            'Mutagenicity_edge_gt.txt',
            'Mutagenicity_edge_labels.txt',
            'Mutagenicity_graph_indicator.txt',
            'Mutagenicity_graph_labels.txt',
            'Mutagenicity_label_readme.txt',
            'Mutagenicity_node_labels.txt',
        ]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise NotImplementedError(
            "Download Mutagenicity raw files from PGExplainer and place them "
            "under {root}/raw/. See datasets/README.md.")

    def process(self):
        pkl_path = os.path.join(self.raw_dir, 'Mutagenicity.pkl')
        use_pkl = os.path.isfile(pkl_path)
        if use_pkl:
            with open(pkl_path, 'rb') as fin:
                _, original_features, original_labels = pkl.load(fin)
            n_graphs = original_labels.shape[0]
        else:
            original_features = None
            print('[Mutag] Mutagenicity.pkl not found — building 14-dim one-hot '
                  'node features from Mutagenicity_node_labels.txt.')

        edge_lists, graph_labels, edge_label_lists, node_type_lists = (
            self.get_graph_data())
        if not use_pkl:
            n_graphs = len(graph_labels)

        data_list = []
        for i in range(n_graphs):
            num_nodes = len(node_type_lists[i])
            edge_index = torch.tensor(edge_lists[i], dtype=torch.long).T

            y = torch.tensor(graph_labels[i]).float().reshape(-1, 1)
            if use_pkl:
                x = torch.tensor(original_features[i][:num_nodes]).float()
                assert original_features[i][num_nodes:].sum() == 0
            else:
                x = _onehot_from_node_types(node_type_lists[i])
            edge_label = torch.tensor(edge_label_lists[i]).float()
            if y.item() != 0:
                edge_label = torch.zeros_like(edge_label).float()

            node_label = torch.zeros(x.shape[0])
            signal_nodes = list(set(
                edge_index[:, edge_label.bool()].reshape(-1).tolist()))
            if y.item() == 0:
                node_label[signal_nodes] = 1

            if len(signal_nodes) != 0:
                node_type = torch.tensor(node_type_lists[i])
                node_type = set(node_type[signal_nodes].tolist())
                assert node_type in ({4, 1}, {4, 3}, {4, 1, 3})  # NO or NH

            if y.item() == 0 and len(signal_nodes) == 0:
                continue

            data_list.append(Data(
                x=x,
                y=y,
                edge_index=edge_index,
                node_label=node_label,
                edge_label=edge_label,
                node_type=torch.tensor(node_type_lists[i]),
            ))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_graph_data(self):
        pri = self.raw_dir + '/Mutagenicity_'

        file_edges = pri + 'A.txt'
        file_edge_labels = pri + 'edge_gt.txt'
        file_graph_indicator = pri + 'graph_indicator.txt'
        file_graph_labels = pri + 'graph_labels.txt'
        file_node_labels = pri + 'node_labels.txt'

        edges = np.loadtxt(file_edges, delimiter=',').astype(np.int32)
        try:
            edge_labels = np.loadtxt(
                file_edge_labels, delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use edge label 0')
            edge_labels = np.zeros(edges.shape[0]).astype(np.int32)

        graph_indicator = np.loadtxt(
            file_graph_indicator, delimiter=',').astype(np.int32)
        graph_labels = np.loadtxt(
            file_graph_labels, delimiter=',').astype(np.int32)

        try:
            node_labels = np.loadtxt(
                file_node_labels, delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use node label 0')
            node_labels = np.zeros(
                graph_indicator.shape[0]).astype(np.int32)

        graph_id = 1
        starts = [1]
        node2graph = {}
        for i in range(len(graph_indicator)):
            if graph_indicator[i] != graph_id:
                graph_id = graph_indicator[i]
                starts.append(i + 1)
            node2graph[i + 1] = len(starts) - 1

        graphid = 0
        edge_lists = []
        edge_label_lists = []
        edge_list = []
        edge_label_list = []
        for (s, t), l in list(zip(edges, edge_labels)):
            sgid = node2graph[s]
            tgid = node2graph[t]
            if sgid != tgid:
                print('edges connecting different graphs, error here, '
                      'please check.')
                print(s, t, 'graph id', sgid, tgid)
                raise RuntimeError('edge spans multiple graphs')
            gid = sgid
            if gid != graphid:
                edge_lists.append(edge_list)
                edge_label_lists.append(edge_label_list)
                edge_list = []
                edge_label_list = []
                graphid = gid
            start = starts[gid]
            edge_list.append((s - start, t - start))
            edge_label_list.append(l)

        edge_lists.append(edge_list)
        edge_label_lists.append(edge_label_list)

        node_label_lists = []
        graphid = 0
        node_label_list = []
        for i in range(len(node_labels)):
            nid = i + 1
            gid = node2graph[nid]
            if gid != graphid:
                node_label_lists.append(node_label_list)
                graphid = gid
                node_label_list = []
            node_label_list.append(node_labels[i])
        node_label_lists.append(node_label_list)

        return edge_lists, graph_labels, edge_label_lists, node_label_lists
