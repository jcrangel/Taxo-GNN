from operator import ne
import os
import time
import argparse
import random
from dgl.data import gindt
from networkx.readwrite.edgelist import parse_edgelist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import networkx as nx
from dgl import DGLGraph
from dgl import function as fn
import dgl.ops as ops

from utils import evaluate_nc, load_data_nc, sample_nodes_for_leafs, batch_for_NCE, batch_for_KL, save_emb, save_emb_
from models import TaxoGNN

from torch.utils.tensorboard import SummaryWriter

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier

# torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='dblp')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=1000, help='Patience')

parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--out_dim', type=int, default=64, help='Number of output units.')
parser.add_argument('--num_layer', type=int, default=2, help='Number of layers')
parser.add_argument('--num_head', type=int, default=1, help='Number of attention heads')
parser.add_argument('--sample_num', type=int, default=32, help='Nodes sample for each leaf taxo')
parser.add_argument('--rbatch_size', type=int, default=256, help='Number of reconstruction edges.')  
parser.add_argument('--neg_k', type=int, default=5, help='Number of negative samples.') 

parser.add_argument('--min_mean', type=float, default=-2.0, help='Mininum mean of categories.')
parser.add_argument('--max_mean', type=float, default=2.0, help='Mininum mean of categories.')
parser.add_argument('--min_std', type=float, default=-6.0, help='Mininum std of categories.')
parser.add_argument('--max_std', type=float, default=-4.0, help='Mininum std of categories.')
parser.add_argument('--m', type=float, default=2.0, help='Margin.')
parser.add_argument('--c', type=float, default=0.5, help='Walking coefficient.')
parser.add_argument('--max_len', type=int, default=5, help='Maximum length of walks.')

parser.add_argument('--eta', type=float, default=1.0, help='Weight of attention')
parser.add_argument('--lamb', type=float, default=1.0, help='Weight of taxo-similarity loss.')
parser.add_argument('--theta', type=float, default=1.0, help='Weight of taxo-tree-leaf loss.')
parser.add_argument('--beta', type=float, default=1.0, help='Weight of taxo-tree-inner loss.')

parser.add_argument('--log_dir', type=str, default="", help='tensorboard log directory')
parser.add_argument('--trial', type=int, default=0)

args = parser.parse_args()

# trial_name = "{}_{}_lr{}_wd{}_dropout{}w{}_dim{}_img{}_layer{}_head{}_m{}_lamb{}_eta{}".format(args.dataset, args.struct, args.lr, args.weight_decay, 
#                             args.fea_drop, args.weight_drop, args.hidden_conv, args.img_neighbor_num, args.num_layer, args.num_head, args.m, args.lamb, args.eta)
if args.log_dir != '':
    writer = SummaryWriter(log_dir=args.log_dir+"/")

print(args)

device = args.device

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if "cuda" in device:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

g, features, labels, train_idx, val_idx, test_idx, taxo_cats, \
        taxo2nodes, taxo_p2c, taxo_c2p_prob, num_taxo, leaf_taxos, inner_taxos, \
        idx_adj, idx_adj_norm, edges, edges_weight, nodes_weight, \
        tn_adjlist, nn_adjlist, nt_adjlist, tax2layer, layer2tax = load_data_nc(dataset=args.dataset, device=args.device, split_idx=args.trial)

use_random = features.shape[0] if 'patent' in args.dataset else -1
model = TaxoGNN(features.shape[1], args.hidden, args.out_dim, args.dropout, args.num_layer, taxo_p2c,
                taxo_c2p_prob, num_taxo, leaf_taxos, inner_taxos, args.eta, use_random, args).to(device)
model.taxo_id = model.taxo_id.to(device)
if model.idx is not None:
    model.idx = model.idx.to(device)

# create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

# main loop
dur = []
los = []
counter = 0
max_auc = 0.0

g.ndata['h'] = features
multi_label = False

if args.dataset ==  'Aminer':
    clf = MultiOutputClassifier(LogisticRegression(random_state = args.seed, max_iter = 1000))
    multi_label = True
else:
    clf = LogisticRegression(random_state = args.seed, max_iter = 1000)

for epoch in range(args.epochs):
    t0 = time.time()

    # train
    model.train()

    raw, emb = model(g, features, taxo_cats, taxo2nodes, idx_adj, idx_adj_norm, device)

    emb_u, emb_v, neg = batch_for_NCE(edges_weight, edges, nodes_weight, emb, args.rbatch_size, args.neg_k)
    loss_train = model.loss_rec(emb_u, emb_v, neg)

    # sample_node = sample_nodes_for_leafs(taxo2nodes, args.sample_num)
    # loss_tree, loss_leaf_mean, loss_leaf_var, loss_mean, loss_var = model.cal_loss_tree(sample_node, raw)
    leafs, pos_node_samples, neg_node_samples = sample_nodes_for_leafs(taxo2nodes, args.sample_num, features.shape[0])
    loss_leaf = model.cal_loss_leaf(leafs, raw, pos_node_samples, neg_node_samples, args.sample_num)
    loss_inner = model.cal_loss_inner()

    tag_1, tag_2, neg = batch_for_KL(num_taxo, tn_adjlist, nn_adjlist, nt_adjlist, device, args.c, args.max_len, tax2layer, layer2tax)
    loss_sim = model.cal_loss_sim(tag_1, tag_2, neg, m = args.m)

    # print(loss_train, loss_sim)
    loss = loss_train + args.lamb * loss_sim + args.theta * loss_leaf + args.beta * loss_inner

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # evaluation
    model.eval()
    raw, emb = model(g, features, taxo_cats, taxo2nodes, idx_adj, idx_adj_norm, device)

    emb = emb.cpu().detach_()
    # print(model.taxo_std_log(model.taxo_id))
    # print(emb[0])

    FREQ = 1
    if epoch % FREQ == 0:
        clf.fit(emb[train_idx], labels[train_idx])
        pred = clf.predict(emb[train_idx])
        train_acc = evaluate_nc(pred, labels[train_idx], multi_label)

        pred = clf.predict(emb[val_idx])
        val_acc = evaluate_nc(pred, labels[val_idx], multi_label)

        pred = clf.predict(emb[test_idx])
        test_acc = evaluate_nc(pred, labels[test_idx], multi_label)
        
        THRES = 1
        if epoch > THRES:
            los.append([epoch, val_acc, test_acc, emb.cpu().detach_()])

            if max_auc < val_acc:
                max_auc = val_acc
                counter = 0
            else:
                counter += FREQ

        # print 
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss.item()),
            'acc_train: {:.4f}'.format(train_acc),
            'acc_val: {:.4f}'.format(val_acc),
            'acc_test: {:.4f}'.format(test_acc),
            'time: {:.4f}s'.format(time.time()-t0))
    
    
    if args.log_dir != '':
        # writer.add_scalar('valid/loss', loss_val, epoch)
        # writer.add_scalar('train/loss', loss_train, epoch)
        # writer.add_scalar('valid/micro_f1',val_f1[1], epoch)
        # writer.add_scalar('train/micro_f1', train_f1[1], epoch)
        # writer.add_scalar('test/micro_f1', test_f1[1], epoch)
        writer.add_scalar('loss_train', loss_train.item(), epoch)
        writer.add_scalar('loss_mean', loss_mean.item(), epoch)
        writer.add_scalar('loss_var', loss_var.item(), epoch)
        writer.add_scalar('loss_leaf_mean', loss_leaf_mean.item(), epoch)
        writer.add_scalar('loss_leaf_var', loss_leaf_var.item(), epoch)
        writer.add_scalar('loss_sim', loss_sim.item(), epoch)

    if counter >= args.patience:
        print('early stop')
        break

print(model.taxo_mean(model.taxo_id))
print(model.taxo_std_log(model.taxo_id))

# res at max val auc
los.sort(key=lambda x: -x[1])
f1 = los[0][2]
print(los[0][0] + 1, f1)
save_emb(los[0][-1], args.dataset)


# model.eval()
# raw, emb = model(g, features, taxo_cats, taxo2nodes, idx_adj, idx_adj_norm, device)
# emb = emb.cpu().detach_()
