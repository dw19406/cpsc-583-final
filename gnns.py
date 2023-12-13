import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,dropout=0.3, use_bn=True):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout
        self.activation = F.relu
        self.bns=None
        self.use_bn = use_bn
        if self.use_bn:
            self.bns = nn.ModuleList()
            for _ in range(num_layers-1):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'])
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        return x

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,dropout=0.3, use_bn=False, heads=2, out_heads=1):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, dropout=dropout, heads=heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, dropout=dropout, heads=heads, concat=True))
        self.convs.append(
            GATConv(hidden_channels*heads, out_channels, dropout=dropout, heads=out_heads, concat=False))
        self.bns=None
        self.use_bn = use_bn
        if self.use_bn:
            self.bns = nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
            for _ in range(num_layers-2):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = dropout
        self.activation = F.elu 

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'])
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        return x