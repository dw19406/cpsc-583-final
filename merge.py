import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from sklearn.neighbors import kneighbors_graph

from logger import Logger
from dataset import load_dataset
from data_utils import load_fixed_splits, adj_mul, get_gpu_memory_map
from parse import parse_method, parser_add_main_args
import time
from gnns import *
from nodeformer import *
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    for i in range(y_true.shape[1]):
        is_labeled = (y_true[:, i] == y_true[:, i])
        correct = (y_true[is_labeled, i] == y_pred[is_labeled, i])
        acc_list.append(float(np.sum(correct)) / len(correct))
    return sum(acc_list) / len(acc_list)

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0")

### Load and preprocess data ###
dataset = load_dataset(args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

# get the splits for all runs
split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop) for _ in range(args.runs)]

#

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

model =NodeFormer(d, args.hidden_channels, c, num_layers=args.num_layers, num_heads=args.num_heads, nb_random_features=args.M, 
                         use_residual=args.use_residual, use_act=args.use_act, use_jk=args.use_jk,nb_gumbel_sample=args.K, rb_order=args.rb_order).to(device)
model2=GAT(d, args.hidden_channels, c, num_layers=args.num_layers, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads).to(device)
criterion = nn.NLLLoss()

logger = Logger(args.runs, args)

model.train()
model2.train()
print('MODEL:', model)
print('Model 2:',model2)

### Adj storage for relational bias ###
adjs = []
adj, _ = remove_self_loops(dataset.graph['edge_index'])
adj, _ = add_self_loops(adj, num_nodes=n)
adjs.append(adj)
for i in range(args.rb_order - 1): # edge_index of high order adjacency
    adj = adj_mul(adj, adj, n)
    adjs.append(adj)
dataset.graph['adjs'] = adjs

### Training loop ###
for run in range(args.runs):
    loss_arr=[]
    split_idx = split_idx_lst[0]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    model2.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    optimizer2=torch.optim.Adam(model2.parameters(),weight_decay=args.weight_decay,lr=args.lr)
    best_val = float('-inf')
    print(torch.cuda.memory_allocated())
    for epoch in range(args.epochs):
        model.train()
        model2.train()
        optimizer.zero_grad()
        optimizer2.zero_grad()
        out, link_loss_ = model(dataset.graph['node_feat'], dataset.graph['adjs'], args.tau)
        out2=model2(dataset)
        out=out/2+out2/2
        out = F.log_softmax(out, dim=1)
        loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])
        torch.cuda.empty_cache()
        loss.backward()
        optimizer.step()
        optimizer2.step()
        if epoch % args.eval_step == 0:
            model.eval()
            model2.eval()
            with torch.no_grad():
                out,_=model(dataset.graph['node_feat'], dataset.graph['adjs'], args.tau)
                out2=model2(dataset)
                out=out+out2
                out = F.log_softmax(out, dim=1)
                train_acc = eval_acc(
                    dataset.label[split_idx['train']], out[split_idx['train']])
                valid_acc = eval_acc(
                    dataset.label[split_idx['valid']], out[split_idx['valid']])
                test_acc = eval_acc(
                    dataset.label[split_idx['test']], out[split_idx['test']])
                valid_loss = criterion(
                    out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])
                result = (train_acc, valid_acc, test_acc, valid_loss, out)
                logger.add_result(run, result[:-1])
                torch.cuda.empty_cache()
                if result[1] > best_val:
                    best_val = result[1]
                loss_arr.append(loss.to('cpu'))
                #print(f'Epoch: {epoch:02d}, 'f'Loss: {loss:.4f}, 'f'Train: {100 * result[0]:.2f}%, 'f'Valid: {100 * result[1]:.2f}%, 'f'Test: {100 * result[2]:.2f}%')
    logger.print_statistics(run)


results = logger.print_statistics()
#print(logger.results)
epochs=[i[3].to('cpu') for i in logger.results[0]]
#print(epochs)
figure, ax=plt.subplots(1,2)
ax[0].plot(loss_arr)
ax[0].set_title('Training loss function plot')
ax[1].plot(epochs)
ax[1].set_title('Validation loss function plot')
ax[0].set(xlabel='epochs',ylabel='training loss')
ax[1].set(xlabel='epochs',ylabel='validation loss')
plt.show()