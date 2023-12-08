import argparse
import pdb
import numpy as np
from numpy.lib.type_check import real
import scipy.sparse as sp
import json
import torch
import networkx as nx
import dgl
from dgl.data import *
import os
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import math
import random

#---------------------------------------------------------------------#
# g: dgl.Graph, on device
# feat: torch.FloatTensor, on device
# train_edges, val_edges, test_edges: np.array
# train_labels, val_labels, test_labels: np.array
# taxo_cats: torch.LongTensor, on device
# taxo2nodes: {str: np.array}
# taxo_p2c: {str: {str: np.array}}
# taxo_c2p_prob: torch.FloatTensor, on device
# num_taxo: int
# leaf_taxos, inner_taxos: [int]
# idx_adj, idx_adj_norm: torch.sparse_coo_tensor, on device
# tn_adjlist, nn_adjlist, nt_adjlist: [[int]]
#---------------------------------------------------------------------#
def load_data_lp(path="../data", dataset="dblp", device="cpu", split_idx = 0):
    edges = np.loadtxt(os.path.join(path, dataset, "edges.txt"), dtype=int)
    fake_edges = np.loadtxt(os.path.join(path, dataset, "fake_edges.txt"), dtype=int)

    # for lp task
    idx_train = np.loadtxt(os.path.join(path, dataset, f"splits/train_idx_lp_{split_idx}.txt"), dtype=int)
    idx_val = np.loadtxt(os.path.join(path, dataset, f"splits/val_idx_lp_{split_idx}.txt"), dtype=int)
    idx_test = np.loadtxt(os.path.join(path, dataset, f"splits/test_idx_lp_{split_idx}.txt"), dtype=int)

    train_edges = np.concatenate((edges[idx_train], fake_edges[idx_train]), axis=0)
    val_edges = np.concatenate((edges[idx_val], fake_edges[idx_val]), axis=0)
    test_edges = np.concatenate((edges[idx_test], fake_edges[idx_test]), axis=0)

    train_labels = [1] * len(idx_train) + [0] * len(idx_train)
    val_labels = [1] * len(idx_val) + [0] * len(idx_val)
    test_labels = [1] * len(idx_test) + [0] * len(idx_test)
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)

    # load features
    if dataset in ['retail', 'rocket', 'new-rocket']:
        feat = np.load(os.path.join(path, dataset, "features.npy"), allow_pickle=True)[()].toarray().astype(np.float32)
        feat = normalize(feat)
    else:
        feat = np.load(os.path.join(path, dataset, "features.npy")).astype(np.float32)
    feat = StandardScaler().fit_transform(feat)
    feat = torch.FloatTensor(feat).to(device)

    # load graph
    edges = edges[idx_train]

    g = nx.Graph()
    g.add_nodes_from([i for i in range(feat.shape[0])])
    g.add_edges_from(edges)
    adj = nx.to_scipy_sparse_matrix(g, [i for i in range(len(g.nodes))], format='coo')
    g = dgl.from_networkx(g)
    g = dgl.to_bidirected(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)

    nn_adjlist = [[i] for i in range(g.num_nodes())]
    for e in edges:
        nn_adjlist[e[0]].append(e[1])
        nn_adjlist[e[1]].append(e[0])

    # structure of taxo tree
    with open(os.path.join(path, dataset, "taxo-tree-p2c.json"), 'r') as f:
        taxo_p2c = json.load(f)

    taxo_c2p_prob = np.load(os.path.join(path, dataset, "taxo-c2p-prob.npy"))
    taxo_c2p_prob = torch.FloatTensor(taxo_c2p_prob).to(device)

    leaf_taxos = np.loadtxt(os.path.join(path, dataset, "leafs.txt"), dtype=int).tolist()
    inner_taxos = np.loadtxt(os.path.join(path, dataset, "inners.txt"), dtype=int).tolist()

    num_taxo = taxo_c2p_prob.shape[0]

    # taxo-path of nodes (one node may be related to multiple paths)
    taxo_cats = np.loadtxt(os.path.join(path, dataset, "taxo-path.txt"), dtype=int)
    idx_adj = sp.coo_matrix((np.ones(taxo_cats.shape[0]), (taxo_cats[:,0], taxo_cats[:,1])), 
                             shape=(taxo_cats.shape[0], g.num_nodes()), 
                             dtype=np.float32) # row: pathidx, column: nodeidx
    idx_adj_norm = normalize(idx_adj.T)
    idx_adj = sparse_mx_to_torch_sparse_tensor(idx_adj).to(device)
    idx_adj_norm = sparse_mx_to_torch_sparse_tensor(idx_adj_norm).to(device)

    tn_adjlist = [[] for i in range(num_taxo)]
    nt_adjlist = [[] for i in range(g.num_nodes())]
    tax2layer = dict()
    for i in range(taxo_cats.shape[0]):
        for j in range(2, taxo_cats.shape[1]):
            node = taxo_cats[i][1]
            tag = taxo_cats[i][j]
            tax2layer[tag] = j - 2
            tn_adjlist[tag].append(node)
            nt_adjlist[node].append(tag)
    layer2tax = [[] for i in range(taxo_cats.shape[1] - 2)]
    for k, v in tax2layer.items():
        layer2tax[v].append(k)

    taxo_cats = torch.LongTensor(taxo_cats[:, 2:]).to(device)

    # leaf cats contain nodes
    with open(os.path.join(path, dataset, "taxo-leaf-to-nodes.json"), 'r') as f:
        taxo2nodes = json.load(f)
    for k in taxo2nodes:
        taxo2nodes[k] = np.array(taxo2nodes[k])

    # sampling weights for NCE
    edge_ret = []
    edge_weight = []
    node_weight = [0.0 for i in range(0, g.num_nodes())]
    adj_pres = adj
    for i in range(0, len(adj.data)):
        edge_ret.append((adj_pres.row[i], adj_pres.col[i]))
        edge_weight.append(float(adj_pres.data[i]))
        node_weight[adj.row[i]] += adj.data[i]
    for i in range(0, len(node_weight)):
        node_weight[i] = math.pow(node_weight[i],0.75)
    # if dataset == 'news':
    #     node_weight = [1.0 for i in range(0, feat.shape[0])]

    return g, feat, train_edges, val_edges, test_edges, train_labels, val_labels, test_labels, \
        taxo_cats, taxo2nodes, taxo_p2c, taxo_c2p_prob, num_taxo, leaf_taxos, \
        inner_taxos, idx_adj, idx_adj_norm, edge_ret, torch.tensor(edge_weight).to(device), \
        torch.tensor(node_weight).to(device), tn_adjlist, nn_adjlist, nt_adjlist, tax2layer, layer2tax


def load_data_nc(path="../data", dataset="dblp", device="cpu", split_idx = 0):
    edges = np.loadtxt(os.path.join(path, dataset, "edges.txt"), dtype=int)

    # for nc task
    idx_train = np.loadtxt(os.path.join(path, dataset, f"splits/train_idx_nc_{split_idx}.txt"), dtype=int)
    idx_val = np.loadtxt(os.path.join(path, dataset, f"splits/val_idx_nc_{split_idx}.txt"), dtype=int)
    idx_test = np.loadtxt(os.path.join(path, dataset, f"splits/test_idx_nc_{split_idx}.txt"), dtype=int)

    # load features
    if dataset in ['retail', 'rocket', 'new-rocket']:
        feat = np.load(os.path.join(path, dataset, "features.npy"), allow_pickle=True)[()].toarray().astype(np.float32)
        feat = normalize(feat)
    else:
        feat = np.load(os.path.join(path, dataset, "features.npy")).astype(np.float32)
    feat = StandardScaler().fit_transform(feat)
    feat = torch.FloatTensor(feat).to(device)

    # load graph
    g = nx.Graph()
    g.add_nodes_from([i for i in range(feat.shape[0])])
    g.add_edges_from(edges)
    adj = nx.to_scipy_sparse_matrix(g, [i for i in range(len(g.nodes))], format='coo')
    g = dgl.from_networkx(g)
    g = dgl.to_bidirected(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)

    nn_adjlist = [[i] for i in range(g.num_nodes())]
    for e in edges:
        nn_adjlist[e[0]].append(e[1])
        nn_adjlist[e[1]].append(e[0])

    # structure of taxo tree
    with open(os.path.join(path, dataset, "taxo-tree-p2c.json"), 'r') as f:
        taxo_p2c = json.load(f)

    taxo_c2p_prob = np.load(os.path.join(path, dataset, "taxo-c2p-prob.npy"))
    taxo_c2p_prob = torch.FloatTensor(taxo_c2p_prob).to(device)

    leaf_taxos = np.loadtxt(os.path.join(path, dataset, "leafs.txt"), dtype=int).tolist()
    inner_taxos = np.loadtxt(os.path.join(path, dataset, "inners.txt"), dtype=int).tolist()

    num_taxo = taxo_c2p_prob.shape[0]

    # taxo-path of nodes (one node may be related to multiple paths)
    taxo_cats = np.loadtxt(os.path.join(path, dataset, "taxo-path.txt"), dtype=int)
    idx_adj = sp.coo_matrix((np.ones(taxo_cats.shape[0]), (taxo_cats[:,0], taxo_cats[:,1])), 
                             shape=(taxo_cats.shape[0], g.num_nodes()), 
                             dtype=np.float32)
    idx_adj_norm = normalize(idx_adj.T)
    idx_adj = sparse_mx_to_torch_sparse_tensor(idx_adj).to(device)
    idx_adj_norm = sparse_mx_to_torch_sparse_tensor(idx_adj_norm).to(device)

    tn_adjlist = [[] for i in range(num_taxo)]
    nt_adjlist = [[] for i in range(g.num_nodes())]
    tax2layer = dict()
    for i in range(taxo_cats.shape[0]):
        for j in range(2, taxo_cats.shape[1]):
            node = taxo_cats[i][1]
            tag = taxo_cats[i][j]
            tax2layer[tag] = j - 2
            tn_adjlist[tag].append(node)
            nt_adjlist[node].append(tag)
    layer2tax = [[] for i in range(taxo_cats.shape[1] - 2)]
    for k, v in tax2layer.items():
        layer2tax[v].append(k)

    taxo_cats = torch.LongTensor(taxo_cats[:, 2:]).to(device)

    # leaf cats contain nodes
    with open(os.path.join(path, dataset, "taxo-leaf-to-nodes.json"), 'r') as f:
        taxo2nodes = json.load(f)
    for k in taxo2nodes:
        taxo2nodes[k] = np.array(taxo2nodes[k])

    # sampling weights for NCE
    edge_ret = []
    edge_weight = []
    node_weight = [0.0 for i in range(0, g.num_nodes())]
    adj_pres = adj
    for i in range(0, len(adj.data)):
        edge_ret.append((adj_pres.row[i], adj_pres.col[i]))
        edge_weight.append(float(adj_pres.data[i]))
        node_weight[adj.row[i]] += adj.data[i]
    for i in range(0, len(node_weight)):
        node_weight[i] = math.pow(node_weight[i],0.75)
    
    if dataset == 'Aminer':
        labels = np.loadtxt(os.path.join(path, dataset, "node-labels.txt"))[:,1:]
    else:
        y_dict = dict()
        for line in open(os.path.join(path, dataset, "node-labels.txt")):
            line = line.strip().strip('\n').split(' ')
            y_dict[int(line[0])] = int(line[1])
        labels = list()
        num_of_nodes = max(y_dict.keys()) + 1
        for i in range(num_of_nodes):
            if i in y_dict:
                labels.append(y_dict[i])
            else:
                labels.append(-1)
        labels = np.array(labels)
        # labels = encode_onehot(labels)
        # labels = np.where(labels)[1]

    return g, feat, labels, idx_train, idx_val, idx_test, \
        taxo_cats, taxo2nodes, taxo_p2c, taxo_c2p_prob, num_taxo, leaf_taxos, \
        inner_taxos, idx_adj, idx_adj_norm, edge_ret, torch.tensor(edge_weight).to(device), \
        torch.tensor(node_weight).to(device), tn_adjlist, nn_adjlist, nt_adjlist, tax2layer, layer2tax


def load_data_tp(path="../data", dataset="dblp", device="cpu", split_idx = 0):
    edges = np.loadtxt(os.path.join(path, dataset, "edges.txt"), dtype=int)

    # for tp task
    idx_train = np.loadtxt(os.path.join(path, dataset, f"splits/train_idx_tp_{split_idx}.txt"), dtype=int)
    idx_train = set(idx_train)

    # load features
    if dataset in ['retail', 'rocket', 'new-rocket']:
        feat = np.load(os.path.join(path, dataset, "features.npy"), allow_pickle=True)[()].toarray().astype(np.float32)
        feat = normalize(feat)
    else:
        feat = np.load(os.path.join(path, dataset, "features.npy")).astype(np.float32)
    feat = StandardScaler().fit_transform(feat)
    feat = torch.FloatTensor(feat).to(device)

    # load graph
    g = nx.Graph()
    g.add_nodes_from([i for i in range(feat.shape[0])])
    g.add_edges_from(edges)
    adj = nx.to_scipy_sparse_matrix(g, [i for i in range(len(g.nodes))], format='coo')
    g = dgl.from_networkx(g)
    g = dgl.to_bidirected(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)

    nn_adjlist = [[i] for i in range(g.num_nodes())]
    for e in edges:
        nn_adjlist[e[0]].append(e[1])
        nn_adjlist[e[1]].append(e[0])

    # structure of taxo tree
    with open(os.path.join(path, dataset, "taxo-tree-p2c.json"), 'r') as f:
        taxo_p2c = json.load(f)

    taxo_c2p_prob = np.load(os.path.join(path, dataset, "taxo-c2p-prob.npy"))
    taxo_c2p_prob = torch.FloatTensor(taxo_c2p_prob).to(device)

    leaf_taxos = np.loadtxt(os.path.join(path, dataset, "leafs.txt"), dtype=int).tolist()
    inner_taxos = np.loadtxt(os.path.join(path, dataset, "inners.txt"), dtype=int).tolist()

    num_taxo = taxo_c2p_prob.shape[0]

    # taxo-path of nodes (one node may be related to multiple paths)
    taxo_cats = np.loadtxt(os.path.join(path, dataset, "taxo-path.txt"), dtype=int)
    idx_adj = sp.coo_matrix((np.ones(taxo_cats.shape[0]), (taxo_cats[:,0], taxo_cats[:,1])), 
                             shape=(taxo_cats.shape[0], g.num_nodes()), 
                             dtype=np.float32)
    idx_adj_norm = normalize(idx_adj.T)
    idx_adj = sparse_mx_to_torch_sparse_tensor(idx_adj).to(device)
    idx_adj_norm = sparse_mx_to_torch_sparse_tensor(idx_adj_norm).to(device)

    tp_mask = np.zeros((taxo_cats.shape[0], 1), dtype=int)

    tn_adjlist = [[] for i in range(num_taxo)]
    nt_adjlist = [[] for i in range(g.num_nodes())]
    tax2layer = dict()
    for i in range(taxo_cats.shape[0]):
        node = taxo_cats[i][1]
        if node in idx_train:
            tp_mask[i][0] = 1
        for j in range(2, taxo_cats.shape[1]):
            tag = taxo_cats[i][j]
            tax2layer[tag] = j - 2
            if node in idx_train:
                tn_adjlist[tag].append(node)
                nt_adjlist[node].append(tag)
    layer2tax = [[] for i in range(taxo_cats.shape[1] - 2)]
    for k, v in tax2layer.items():
        layer2tax[v].append(k)

    train_i = taxo_cats[np.where(tp_mask == 1)[0], 1]
    train_l = taxo_cats[np.where(tp_mask == 1)[0], -1]
    test_i = taxo_cats[np.where(tp_mask == 0)[0], 1]
    test_l = taxo_cats[np.where(tp_mask == 0)[0], -1]

    taxo_cats = torch.LongTensor(taxo_cats[:, 2:]).to(device)
    tp_mask = torch.LongTensor(tp_mask).to(device)

    taxo_cats = tp_mask * taxo_cats + (1 - tp_mask) * torch.ones_like(taxo_cats) * (num_taxo - 1)

    # leaf cats contain nodes
    with open(os.path.join(path, dataset, "taxo-leaf-to-nodes.json"), 'r') as f:
        taxo2nodes = json.load(f)
    for k in taxo2nodes:
        t = set(taxo2nodes[k])
        t = list(t & idx_train)
        taxo2nodes[k] = np.array(t)

    # sampling weights for NCE
    edge_ret = []
    edge_weight = []
    node_weight = [0.0 for i in range(0, g.num_nodes())]
    adj_pres = adj
    for i in range(0, len(adj.data)):
        edge_ret.append((adj_pres.row[i], adj_pres.col[i]))
        edge_weight.append(float(adj_pres.data[i]))
        node_weight[adj.row[i]] += adj.data[i]
    for i in range(0, len(node_weight)):
        node_weight[i] = math.pow(node_weight[i],0.75)
    
    # edge_ret = np.loadtxt(os.path.join(path, dataset, "corpus.txt"), dtype=int)
    # edge_weight = np.ones(edge_ret.shape[0])

    return g, feat, tp_mask, train_i, train_l, test_i, test_l, \
        taxo_cats, taxo2nodes, taxo_p2c, taxo_c2p_prob, num_taxo, leaf_taxos, \
        inner_taxos, idx_adj, idx_adj_norm, edge_ret, torch.tensor(edge_weight).to(device), \
        torch.tensor(node_weight).to(device), tn_adjlist, nn_adjlist, nt_adjlist, tax2layer, layer2tax


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sample_vec(mean, std):
    eps = torch.normal(torch.zeros_like(mean), torch.ones_like(std))
    return mean + std * eps


def sample_nodes_for_leafs(taxo2node, sample_num, node_num):
    # res = dict()
    # for k, v in taxo2node.items():
    #     sam = np.random.choice(v, sample_num, replace=True)
    #     res[int(k)] = sam
    # return res
    leaf = []
    pos = []
    neg = []
    node_list = np.array([i for i in range(node_num)])
    for k, v in taxo2node.items():
        leaf.append(int(k))
        pos_sam = np.random.choice(v, sample_num, replace=True)
        pos.append(pos_sam)
        neg_sam = np.random.choice(node_list, sample_num, replace=True)
        neg.append(neg_sam)

    pos = np.stack(pos).reshape(-1)
    neg = np.stack(neg).reshape(-1)

    return leaf, pos, neg


def batch_for_NCE(edges_weight, edges, nodes_weight, emb, rbatch_size = 256, k = 1):
    sample_edge = torch.multinomial(edges_weight, rbatch_size, replacement=False)
    idx_u = [edges[i][0] for i in sample_edge]
    idx_v = [edges[i][1] for i in sample_edge]
    neg = [emb[torch.multinomial(nodes_weight, rbatch_size, replacement=True)] for i in range(0, k)]
    return emb[idx_u], emb[idx_v], neg


def batch_for_KL(num_taxo, tn_adjlist, nn_adjlist, nt_adjlist, device, c, max_len, tax2layer, layer2tax):
    tag_1 = random.sample(range(num_taxo-1), min(num_taxo-1, 128))
    tag_2 = []
    tag_neg = []
    for t in tag_1:
        dst, n_dst = hybrid_walk(t, tn_adjlist, nn_adjlist, nt_adjlist, c, max_len, tax2layer, layer2tax)
        tag_2.append(dst)
        tag_neg.append(n_dst)
    return torch.LongTensor(tag_1).to(device), torch.LongTensor(tag_2).to(device), \
            torch.LongTensor(tag_neg).to(device)


def hybrid_walk(src, tn_adjlist, nn_adjlist, nt_adjlist, c, max_len, tax2layer, layer2tax):
    cur = "tag"
    l = 0
    layer = tax2layer[src]
    while l <= max_len:
        if cur == "tag":
            dst = np.random.choice(tn_adjlist[src])
            l += 1
            cur = "node"
            src = dst
        else:
            t = random.random()
            if t <= c or l == max_len:
                dst = np.random.choice(nt_adjlist[src])
                while not tax2layer[dst] == layer:
                    dst = np.random.choice(nt_adjlist[src])
                break
            else:
                dst = np.random.choice(nn_adjlist[src])
                l += 1
    n_dst = np.random.choice(layer2tax[layer])
    return dst, n_dst


def pair_norm(x):
    col_mean = x.mean(dim=0)
    x = x - col_mean
    row_norm = (1e-6 + (x ** 2).sum(dim=1, keepdim=True)).sqrt()
    x = 50.0 * x / row_norm
    return x
    

def KLDivergence(mean_1, mean_2, var_1, var_2, d):
    trace = torch.sum(var_2 / var_1, dim=1)
    det = torch.sum(torch.log(var_2) - torch.log(var_1), dim=1)
    mid = torch.sum(((mean_1 - mean_2) ** 2) / var_1, dim = 1)
    return 0.5 * (trace + mid - d - det)


def WDistance(mean_1, mean_2, var_1, var_2):
    m = torch.sum((mean_1 - mean_2) ** 2, dim=1)
    std = torch.sum((torch.sqrt(var_1) - torch.sqrt(var_2)) ** 2, dim=1)
    return m + std


def evaluate_lp(y_pred, y_true):
    return roc_auc_score(y_true, y_pred)


def evaluate_nc(y_pred, y_true, multi_label = False):
    if multi_label:
        acc = 0.0
        for i in range(y_pred.shape[1]):
            acc += accuracy_score(y_true[:, i], y_pred[:,i])
            # print(confusion_matrix(y_true[:, i], y_pred[:,i]))
        acc /= y_pred.shape[1]
    else:
        acc =  accuracy_score(y_true, y_pred)
    return acc


def save_emb(emb, dataset):
    np.save(f"../emb/OURS_{dataset}_tp_emb_1.npy", emb)


def save_emb_(emb, dataset):
    np.save(f"../emb/OURS_{dataset}_emb_1.npy", emb)


if __name__ == "__main__":
    load_data_tp(dataset = 'Aminer')

