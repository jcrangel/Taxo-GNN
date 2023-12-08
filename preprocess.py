import argparse
import numpy as np
import scipy.sparse as sp
import torch
import random
import networkx as nx
import pandas as pd
import json
import os
import queue
import pdb


def statistics(dataset):
    print(f'---{dataset}---')
    g = nx.read_edgelist(f'../data/{dataset}/edges.txt')
    num_nodes = len(g.nodes())
    num_edges = len(g.edges())
    print(f'# nodes: {num_nodes}')
    print(f'# edges: {num_edges}')

    with open(f'../data/{dataset}/taxo-tree-c2p.json', 'r') as f:
        c2p = json.load(f)
    taxo_num = len(c2p.keys())
    print(f'# categories: {taxo_num}')

    with open(f'../data/{dataset}/taxo-tree-p2c.json', 'r') as f:
        p2c = json.load(f)
    for a in p2c:
        print(len(p2c[a]))
    num_layer = max([int(k) for k in p2c])
    print(f'# taxo-tree layers: {num_layer+1}')

    return num_nodes, num_edges


def process_taxo_tree(dataset, remove_roots):
    taxos = pd.read_csv(f'../data/{dataset}/taxo-tree.txt', sep='\t', dtype=str)
    childs = set(taxos['child'])
    taxo_c2p = dict()

    if remove_roots:
        roots = list(taxos[taxos['parent'] == '-1']['child'])
        for root in roots:
            childs.remove(root)
        taxo_idx = {n: i for i, n in enumerate(childs)}
        for index, row in taxos.iterrows():
            c = row['child']
            p = row['parent']
            if p == '-1':
                continue
            elif p in roots:
                taxo_c2p[taxo_idx[c]] = -1
            else:
                taxo_c2p[taxo_idx[c]] = taxo_idx[p]
    else:
        taxo_idx = {n: i for i, n in enumerate(childs)}
        for index, row in taxos.iterrows():
            c = row['child']
            p = row['parent']
            if p == '-1':
                taxo_c2p[taxo_idx[c]] = -1
            else:
                taxo_c2p[taxo_idx[c]] = taxo_idx[p]

    # print(sorted(taxo_c2p.items()))
    # print(len(taxo_c2p))
    with open(f'../data/{dataset}/taxo-tree-c2p.json', 'w') as f:
        json.dump(taxo_c2p, f)
    
    tmp_taxo_p2c = dict()
    for c, p in taxo_c2p.items():
        if p in tmp_taxo_p2c:
            tmp_taxo_p2c[p].append(c)
        else:
            tmp_taxo_p2c[p] = [c]
    # print(tmp_taxo_p2c)
    
    taxonomys = set(range(len(taxo_idx)))
    inner = set(tmp_taxo_p2c.keys())
    # inner.remove(-1)
    leaf = taxonomys - inner
    # print(inner)
    # print(leaf)

    level = dict()
    level[-1] = 0
    q = queue.Queue()
    q.put(-1)
    while not q.empty():
        cur = q.get()
        chs = tmp_taxo_p2c[cur]
        for c in chs:
            level[c] = level[cur] + 1
            if c in inner:
                q.put(c)

    layer_num = max([level[c] for c in level])
    taxo_p2c = {i: dict() for i in range(layer_num)}
    # print(layer_num)
    for i in inner:
        lev = level[i]
        taxo_p2c[lev][i] = tmp_taxo_p2c[i]
    # print(taxo_p2c)
    
    with open(f'../data/{dataset}/taxo-tree-p2c.json', 'w') as f:
        json.dump(taxo_p2c, f)
    
    inner_np = np.array(list(inner), dtype=int)
    leaf_np = np.array(list(leaf), dtype=int)
    np.savetxt(f'../data/{dataset}/leafs.txt', leaf_np, fmt='%d')
    np.savetxt(f'../data/{dataset}/inners.txt', inner_np, fmt='%d')

    return [taxo_c2p, taxo_idx, len(taxo_p2c), leaf]


def process_taxo_nodes(dataset, ret, task):
    taxo_c2p = ret[0]
    taxo_dict = ret[1]

    taxos = pd.read_csv(f'../data/{dataset}/taxo.txt', sep='\t')
    taxo2node = dict()
    node2taxo = []
    for index, row in taxos.iterrows():
        n = int(row['idx'])
        t = taxo_dict[str(row['taxo'])]
        if t in ret[3]:
            if t in taxo2node:
                taxo2node[t].append(n)
            else:
                taxo2node[t] = [n]
        path = [t]
        t = taxo_c2p[t]
        while t != -1:
            path.append(t)
            t = taxo_c2p[t]
        tmp = [index, n]
        tmp = tmp + ([path[0]] * (ret[2] - len(path)))
        path = tmp + path
        node2taxo.append(path)
    
    # pdb.set_trace()
    np.savetxt(f'../data/{dataset}/taxo-path.txt', np.array(node2taxo, dtype=int), fmt='%d')
    with open(f'../data/{dataset}/taxo-leaf-to-nodes.json', 'w') as f:
        json.dump(taxo2node, f)


def taxo_prob(dataset):
    with open(f'../data/{dataset}/taxo-leaf-to-nodes.json', 'r') as f:
        data = json.load(f)
    cnt = dict()
    for k in data:
        cnt[int(k)] = len(data[k])

    with open(f'../data/{dataset}/taxo-tree-p2c.json', 'r') as f:
        p2c = json.load(f)
    num_layer = max([int(k) for k in p2c])

    with open(f'../data/{dataset}/taxo-tree-c2p.json', 'r') as f:
        c2p = json.load(f)
    taxo_num = len(c2p.keys()) + 1
    taxo_c2p_prob = np.ones(taxo_num)

    for i in range(num_layer, -1, -1):
        cur_layer = p2c[str(i)]
        for p, cs in cur_layer.items():
            p = int(p)
            if p not in cnt:
                cnt[p] = 0
            for c in cs:
                # print(p, c)
                cnt[p] += cnt[c]
            for c in cs:
                taxo_c2p_prob[c] = cnt[c] / cnt[p]
    # print(cnt)

    # pdb.set_trace()
    np.save(f'../data/{dataset}/taxo-c2p-prob', taxo_c2p_prob)


def generate_lp_splits(dataset, n_total, n_train, n_test, split_idx):
    idx = [i for i in range(n_total)]
    random.shuffle(idx)
    train_idx = idx[:n_train]
    val_idx = idx[n_train: -n_test]
    test_idx = idx[-n_test:]
    train_idx = np.array(train_idx, dtype=int)
    val_idx = np.array(val_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)

    np.savetxt(f'../data/{dataset}/splits/train_idx_lp_{split_idx}.txt', train_idx, fmt='%d')
    np.savetxt(f'../data/{dataset}/splits/val_idx_lp_{split_idx}.txt', val_idx, fmt='%d')
    np.savetxt(f'../data/{dataset}/splits/test_idx_lp_{split_idx}.txt', test_idx, fmt='%d')


def generate_nc_splits(dataset, n_total, n_train, n_test, split_idx):
    # idx = [i for i in range(n_total)]
    idx = np.loadtxt(f'../data/{dataset}/node-labels.txt')[:,0]
    n_train = int(n_train * len(idx))
    n_test = int(n_test * len(idx))
    random.shuffle(idx)
    train_idx = idx[:n_train]
    val_idx = idx[n_train: -n_test]
    test_idx = idx[-n_test:]
    train_idx = np.array(train_idx, dtype=int)
    val_idx = np.array(val_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)

    np.savetxt(f'../data/{dataset}/splits/train_idx_nc_{split_idx}.txt', train_idx, fmt='%d')
    np.savetxt(f'../data/{dataset}/splits/val_idx_nc_{split_idx}.txt', val_idx, fmt='%d')
    np.savetxt(f'../data/{dataset}/splits/test_idx_nc_{split_idx}.txt', test_idx, fmt='%d')


def generate_tp_splits(dataset, n_total, n_train, split_idx):
    n_train = int(n_train * n_total)
    with open(f'../data/{dataset}/taxo-leaf-to-nodes.json', 'r') as f:
        data = json.load(f)
    must_in = set()
    for k, v in data.items():
        random.shuffle(v)
        must_in.add(v[0])
    taxo_path = np.loadtxt(f'../data/{dataset}/taxo-path.txt', dtype=int)
    uniq = set()
    for i in range(taxo_path.shape[0]):
        node = taxo_path[i][1]
        if node in uniq:
            must_in.add(node)
        else:
            uniq.add(node)
    
    print(len(must_in))

    idx = set(i for i in range(n_total))
    idx = idx - must_in
    idx = list(idx)
    n_train = n_train - len(must_in)
    random.shuffle(idx)
    train_idx = idx[:n_train]
    train_idx = train_idx + list(must_in)
    train_idx = np.array(train_idx, dtype=int)

    np.savetxt(f'../data/{dataset}/splits/train_idx_tp_{split_idx}.txt', train_idx, fmt='%d')


if __name__ == "__main__":
    dataset_name = 'new-rocket'
    task = 'tp'
    split_idx = 1
    if task == 'lp':
        split = [0.8, 0.1, 0.1]
    elif task == 'nc':
        split = [0.6, 0.2, 0.2]
    else:
        split = [0.8, 0.2]
    if dataset_name in ['Aminer']:
        remove_roots = True
    else:
        remove_roots = False
    
    # ret = process_taxo_tree(dataset_name, remove_roots)

    # process_taxo_nodes(dataset_name, ret, task)

    # taxo_prob(dataset_name)

    num_nodes, num_edges = statistics(dataset_name)

    if task == 'nc':
        generate_nc_splits(dataset_name, num_nodes, split[0], split[2], split_idx)
    elif task == 'lp':
        generate_lp_splits(dataset_name, num_edges, int(split[0] * num_edges), int(split[2] * num_edges), split_idx)
    else:
        generate_tp_splits(dataset_name, num_nodes, 0.8, split_idx)