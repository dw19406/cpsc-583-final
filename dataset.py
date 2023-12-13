from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T

from data_utils import rand_train_test_idx, even_quantile_labels, to_sparse_tensor, dataset_drive_url, class_rand_splits

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
import os

import scipy.sparse as sp

from ogb.nodeproppred import NodePropPredDataset


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):
        """
        split_type: 'random' for random splitting, 'class' for splitting with equal node num per class
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        label_num_per_class: num of nodes per class
        """

        if split_type == 'random':
            ignore_negative = True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_dataset(dataname):
    transform = T.NormalizeFeatures()
    torch_dataset = Planetoid(root=f'Planetoid', name=dataname, transform=transform)
    data = torch_dataset[0]
    dataset = NCDataset(dataname)
    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]
    dataset.graph = {'edge_index': data.edge_index,
                     'node_feat': data.x,
                     'edge_feat': None,
                     'num_nodes': data.num_nodes}
    dataset.label = data.y
    return dataset