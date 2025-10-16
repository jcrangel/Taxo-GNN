"""
Training script specifically for new-patent-small dataset
Handles the space in the folder name properly
"""

import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import networkx as nx
from dgl import DGLGraph

from utils import evaluate_lp, load_data_lp, sample_nodes_for_leafs, batch_for_NCE, batch_for_KL, save_emb
from models import TaxoGNN, FermiDiracDecoder

parser = argparse.ArgumentParser(description='Train Taxo-GNN on new-patent-small dataset')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
parser.add_argument('--patience', type=int, default=500, help='Early stopping patience')

parser.add_argument('--hidden', type=int, default=32, help='Hidden dimension')
parser.add_argument('--out_dim', type=int, default=64, help='Output dimension')
parser.add_argument('--num_layer', type=int, default=2, help='Number of GNN layers')
parser.add_argument('--num_head', type=int, default=1, help='Number of attention heads')
parser.add_argument('--sample_num', type=int, default=32, help='Samples per leaf taxonomy')
parser.add_argument('--rbatch_size', type=int, default=256, help='Reconstruction batch size')  
parser.add_argument('--neg_k', type=int, default=5, help='Negative samples') 

parser.add_argument('--min_mean', type=float, default=-2.0, help='Min taxonomy mean')
parser.add_argument('--max_mean', type=float, default=2.0, help='Max taxonomy mean')
parser.add_argument('--min_std', type=float, default=-6.0, help='Min taxonomy log-std')
parser.add_argument('--max_std', type=float, default=-4.0, help='Max taxonomy log-std')
parser.add_argument('--m', type=float, default=2.0, help='Margin for similarity loss')
parser.add_argument('--c', type=float, default=0.5, help='Walk coefficient')
parser.add_argument('--max_len', type=int, default=5, help='Max walk length')

parser.add_argument('--eta', type=float, default=1.0, help='Feature vs taxonomy attention weight')
parser.add_argument('--lamb', type=float, default=1.0, help='Taxonomy similarity loss weight')
parser.add_argument('--theta', type=float, default=1.0, help='Leaf taxonomy loss weight')
parser.add_argument('--beta', type=float, default=1.0, help='Inner taxonomy loss weight')

parser.add_argument('--trial', type=int, default=0, help='Trial/split index')

args = parser.parse_args()

print("="*80)
print("Training Taxo-GNN on new-patent-small dataset")
print("="*80)
print(args)
print("="*80)

device = args.device

# Set random seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if "cuda" in device:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Fix the path to handle space in folder name
dataset_path = "Taxo-GNN-data"
dataset_name = "new-patent-small"

print(f"\nLoading data from: {os.path.join(dataset_path, dataset_name)}")
print("This may take a moment...")

# Load data
g, features, train_edges, val_edges, test_edges, train_labels, val_labels, test_labels, taxo_cats, \
        taxo2nodes, taxo_p2c, taxo_c2p_prob, num_taxo, leaf_taxos, inner_taxos, \
        idx_adj, idx_adj_norm, edges, edges_weight, nodes_weight, \
        tn_adjlist, nn_adjlist, nt_adjlist, tax2layer, layer2tax = load_data_lp(
            path=dataset_path, dataset=dataset_name, device=device, split_idx=args.trial)

print(f"\nDataset Statistics:")
print(f"  Number of nodes: {g.num_nodes()}")
print(f"  Number of edges: {g.num_edges()}")
print(f"  Feature dimension: {features.shape[1]}")
print(f"  Number of taxonomies: {num_taxo}")
print(f"  Number of taxonomy layers: {len(taxo_p2c)}")
print(f"  Number of leaf taxonomies: {len(leaf_taxos)}")
print(f"  Number of inner taxonomies: {len(inner_taxos)}")
print(f"  Train edges: {len(train_edges)} ({len(train_labels)} with negatives)")
print(f"  Val edges: {len(val_edges)} ({len(val_labels)} with negatives)")
print(f"  Test edges: {len(test_edges)} ({len(test_labels)} with negatives)")

# Initialize model
use_random = features.shape[0]  # Use random embeddings for large patent graphs
model = TaxoGNN(features.shape[1], args.hidden, args.out_dim, args.dropout, args.num_layer, taxo_p2c,
                taxo_c2p_prob, num_taxo, leaf_taxos, inner_taxos, args.eta, use_random, args).to(device)
model.taxo_id = model.taxo_id.to(device)
if model.idx is not None:
    model.idx = model.idx.to(device)

decoder = FermiDiracDecoder(r=2.0, t=1.0)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Training loop
dur = []
los = []
counter = 0
max_auc = 0.0
best_epoch = 0

g.ndata['h'] = features

print("\n" + "="*80)
print("Starting Training")
print("="*80)

for epoch in range(args.epochs):
    t0 = time.time()

    # Train
    model.train()

    raw, emb = model(g, features, taxo_cats, taxo2nodes, idx_adj, idx_adj_norm, device)

    # Loss 1: Graph reconstruction (NCE)
    emb_u, emb_v, neg = batch_for_NCE(edges_weight, edges, nodes_weight, emb, args.rbatch_size, args.neg_k)
    loss_train = model.loss_rec(emb_u, emb_v, neg)

    # Loss 2: Leaf taxonomy alignment
    leafs, pos_node_samples, neg_node_samples = sample_nodes_for_leafs(taxo2nodes, args.sample_num, features.shape[0])
    loss_leaf = model.cal_loss_leaf(leafs, raw, pos_node_samples, neg_node_samples, args.sample_num)
    
    # Loss 3: Inner taxonomy hierarchy
    loss_inner = model.cal_loss_inner()

    # Loss 4: Taxonomy similarity (from random walks)
    tag_1, tag_2, neg = batch_for_KL(num_taxo, tn_adjlist, nn_adjlist, nt_adjlist, device, args.c, args.max_len, tax2layer, layer2tax)
    loss_sim = model.cal_loss_sim(tag_1, tag_2, neg, m=args.m)

    # Total loss
    loss = loss_train + args.lamb * loss_sim + args.theta * loss_leaf + args.beta * loss_inner

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        raw, emb = model(g, features, taxo_cats, taxo2nodes, idx_adj, idx_adj_norm, device)
        
        # Train AUC
        pred = torch.sum((emb[train_edges[:,0]] - emb[train_edges[:,1]]) ** 2, dim=1)
        pred = decoder(pred)
        pred = pred.cpu().detach_()
        train_auc = evaluate_lp(pred, train_labels)

        # Val AUC
        pred = torch.sum((emb[val_edges[:,0]] - emb[val_edges[:,1]]) ** 2, dim=1)
        pred = decoder(pred)
        pred = pred.cpu().detach_()
        val_auc = evaluate_lp(pred, val_labels)

        # Test AUC
        pred = torch.sum((emb[test_edges[:,0]] - emb[test_edges[:,1]]) ** 2, dim=1)
        pred = decoder(pred)
        pred = pred.cpu().detach_()
        test_auc = evaluate_lp(pred, test_labels)
    
    if epoch > 50:  # Start tracking after warmup
        los.append([epoch, val_auc, test_auc, emb.cpu().detach_()])

    if max_auc < val_auc:
        max_auc = val_auc
        best_epoch = epoch
        counter = 0
    else:
        counter += 1

    dur.append(time.time() - t0)
    
    # Print progress
    if epoch % 10 == 0 or epoch < 5:
        print(f'Epoch: {epoch+1:04d} | '
              f'Loss: {loss.item():.4f} (rec:{loss_train.item():.3f} sim:{loss_sim.item():.3f} '
              f'leaf:{loss_leaf.item():.3f} inner:{loss_inner.item():.3f}) | '
              f'AUC - Train: {train_auc:.4f} Val: {val_auc:.4f} Test: {test_auc:.4f} | '
              f'Time: {np.mean(dur):.3f}s')
    
    if counter >= args.patience:
        print(f'\nEarly stopping at epoch {epoch+1}')
        break

print("\n" + "="*80)
print("Training Complete")
print("="*80)

# Get results at best validation AUC
los.sort(key=lambda x: -x[1])
best_val_auc = los[0][1]
best_test_auc = los[0][2]
best_epoch_final = los[0][0]

print(f"\nBest Results:")
print(f"  Epoch: {best_epoch_final + 1}")
print(f"  Validation AUC: {best_val_auc:.4f}")
print(f"  Test AUC: {best_test_auc:.4f}")

# Print some taxonomy statistics
print(f"\nTaxonomy Embedding Statistics:")
means = model.taxo_mean.weight.data
stds = torch.exp(model.taxo_std_log.weight.data)
print(f"  Mean - min: {means.min().item():.4f}, max: {means.max().item():.4f}, avg: {means.mean().item():.4f}")
print(f"  Std - min: {stds.min().item():.4f}, max: {stds.max().item():.4f}, avg: {stds.mean().item():.4f}")

# Save embeddings
output_dir = "embeddings"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"taxo_gnn_new_patent_small_trial{args.trial}.npy")
np.save(output_path, los[0][-1].numpy())
print(f"\nSaved embeddings to: {output_path}")

print("\n" + "="*80)
print("Done!")
print("="*80)
