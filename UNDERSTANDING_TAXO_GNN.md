# Taxo-GNN Repository - Comprehensive Guide

## Table of Contents
1. [Repository Overview](#repository-overview)
2. [File Structure and Purpose](#file-structure-and-purpose)
3. [Understanding the Training Modules](#understanding-the-training-modules)
4. [Deep Dive: Main GNN Model (models.py)](#deep-dive-main-gnn-model)
5. [Training on new-patent-small Dataset](#training-on-new-patent-small-dataset)
6. [Data Flow Architecture](#data-flow-architecture)

---

## Repository Overview

This repository implements **Taxo-GNN** (Taxonomy-Enhanced Graph Neural Network) from CIKM 2022. The core idea is to enhance traditional GNN learning by incorporating taxonomical (hierarchical category) knowledge.

**Key Concept**: Each taxonomy category is represented as a **Gaussian distribution** (mean μ and variance σ²), capturing uncertainty. The model:
1. **Information Distillation**: Aggregates graph information bottom-up to refine taxonomy embeddings
2. **Knowledge Fusion**: Injects taxonomy knowledge into GNN message-passing to guide node embedding learning

---

## File Structure and Purpose

### Core Files

#### 1. **models.py** - Neural Network Architectures
Contains all model components:

- **`FermiDiracDecoder`**: Decoder for link prediction (converts distances to probabilities)
  - Uses physics-inspired Fermi-Dirac distribution: `1 / (exp((dist - r) / t) + 1)`
  
- **`MLP`**: Multi-layer perceptron with batch normalization
  - Used to transform embeddings to different dimensions
  
- **`ConvLayer`**: Custom GNN convolution layer (the heart of knowledge fusion)
  - Combines feature-based attention (`alpha_f`) and taxonomy-based attention (`alpha_t`)
  - Formula: `alpha = η * alpha_f + (1 - η) * alpha_t`
  - `eta` controls the balance between graph structure and taxonomy structure
  
- **`TaxoGNN`**: Main model (detailed analysis below)

#### 2. **utils.py** - Data Loading and Helper Functions

**Data Loading Functions** (returns preprocessed data for each task):
- `load_data_lp()`: Load data for **Link Prediction**
- `load_data_nc()`: Load data for **Node Classification** 
- `load_data_tp()`: Load data for **Taxonomy Prediction**

**Sampling Functions**:
- `sample_nodes_for_leafs()`: Sample positive/negative nodes for leaf taxonomies
- `batch_for_NCE()`: Create batches for Noise Contrastive Estimation (graph reconstruction)
- `batch_for_KL()`: Create batches for taxonomy similarity learning
- `hybrid_walk()`: Random walk that alternates between taxonomy and node space

**Loss Functions**:
- `KLDivergence()`: KL divergence between two Gaussian distributions
- `WDistance()`: Wasserstein distance between Gaussians (sum of mean distance and std distance)

**Evaluation**:
- `evaluate_lp()`: AUC-ROC for link prediction
- `evaluate_nc()`: Accuracy for node classification

#### 3. **preprocess.py** - Data Preprocessing

Processes raw taxonomy and graph data into model-ready format:

**Main Functions**:
- `process_taxo_tree()`: Converts taxonomy tree to parent-to-child (p2c) and child-to-parent (c2p) mappings
  - Identifies leaf and inner nodes
  - Assigns layer levels to taxonomy nodes
  
- `process_taxo_nodes()`: Maps nodes to their taxonomy paths
  - Creates `taxo-path.txt`: Each node's path through the taxonomy hierarchy
  - Creates `taxo-leaf-to-nodes.json`: Which nodes belong to each leaf category
  
- `taxo_prob()`: Computes child-to-parent probability weights
  - Used for hierarchical aggregation (parent = weighted sum of children)
  
- `generate_lp_splits()`: Generate train/val/test splits for link prediction
- `generate_nc_splits()`: Generate splits for node classification  
- `generate_tp_splits()`: Generate splits for taxonomy prediction (ensures coverage)

#### 4. **train.py** - Link Prediction Training

Trains model to predict whether edges exist between nodes.

**Loss Components**:
```python
loss = loss_train + λ*loss_sim + θ*loss_leaf + β*loss_inner
```
- `loss_train`: NCE reconstruction loss (predict node co-occurrence)
- `loss_sim`: Taxonomy similarity loss (closer taxonomies should be similar)
- `loss_leaf`: Leaf taxonomy loss (leaf categories should match their nodes)
- `loss_inner`: Inner taxonomy loss (parent should be mixture of children)

**Evaluation**: Uses Fermi-Dirac decoder to convert embedding distances to link probabilities

#### 5. **train_nc.py** - Node Classification Training

Trains model to predict node labels (uses learned embeddings as features for classifier).

**Key Differences from train.py**:
- Loads labeled nodes from `node-labels.txt`
- Uses Logistic Regression on top of learned embeddings
- Supports multi-label classification (Aminer dataset)
- Evaluation: Classification accuracy

#### 6. **train_tp.py** - Taxonomy Prediction Training

Trains model to predict which taxonomy category a node belongs to (semi-supervised).

**Key Features**:
- Uses `tp_mask` to hide some node-taxonomy associations during training
- Tests on masked associations
- Ensures training set has at least one node per taxonomy (coverage requirement)

---

## Understanding the Training Modules

### Why Three Training Files?

Each corresponds to a different **downstream task**:

| File | Task | Input | Output | Evaluation |
|------|------|-------|--------|------------|
| `train.py` | Link Prediction | Graph edges | Edge exists? | AUC-ROC |
| `train_nc.py` | Node Classification | Node features | Node class | Accuracy |
| `train_tp.py` | Taxonomy Prediction | Partial taxonomy | Node category | Accuracy |

**All three share the same core model** (`TaxoGNN`), but:
- Use different loss weightings
- Evaluate on different metrics
- Process data differently

**Note**: The model learns **joint representations** that are useful for all three tasks because the taxonomy provides universal structural guidance.

---

## Deep Dive: Main GNN Model (models.py)

### TaxoGNN Architecture

#### Initialization Parameters
```python
TaxoGNN(in_dim, hidden_dim, out_dim, dropout, num_layer, 
        taxo_p2c, taxo_c2p_prob, num_taxo, leaf_taxos,
        inner_taxos, eta, use_random, args)
```

**Key Components**:
1. **Taxonomy Embeddings** (Gaussian distributions):
   - `taxo_mean`: Mean vectors for each taxonomy (μ)
   - `taxo_std_log`: Log standard deviation (log σ) - ensures positivity via exp()
   
2. **GNN Layers**:
   - `W`: Input projection layer
   - `conv_layers`: Stack of `ConvLayer` modules (taxonomy-aware convolution)
   
3. **Readout MLPs**:
   - One MLP per taxonomy layer + one for node-specific features
   - Each produces part of the final embedding

### Forward Pass - Complete Data Flow

Let me break down the forward pass step-by-step:

```python
def forward(self, g, h, taxo_cats, taxo2nodes, idx_adj, idx_adj_norm, device, tp_mask=None):
```

#### **Phase 1: Feature Initialization**
```python
if self.emb is not None:
    h = self.emb(self.idx)  # For large graphs, use learned embeddings
h = self.W(h)              # Project to hidden_dim
h = self.activation(h)     # PReLU activation
h = F.dropout(h, self.dropout, training=self.training)
g.ndata['h'] = h          # Store in graph
```
- Input features `h`: shape `(num_nodes, in_dim)`
- After projection: `(num_nodes, hidden_dim)`

#### **Phase 2: Taxonomy Embedding Extraction** (Information Distillation Start)
```python
tax_mean_all = []
tax_std_all = []
for i in range(self.taxo_layers):
    taxo_index = taxo_cats[:,i]  # Get taxonomy ID for layer i
    means = self.taxo_mean(taxo_index)  # Lookup mean vectors
    stds = torch.exp(self.taxo_std_log(taxo_index))  # Lookup stds (exp for positivity)
    tax_mean_all.append(means)
    tax_std_all.append(stds)
```
- `taxo_cats`: shape `(num_paths, num_taxo_layers)` - each node's path through taxonomy
- For each layer, extract the Gaussian parameters (μ, σ) for that taxonomy node

#### **Phase 3: Create Taxonomy Context for Nodes**
```python
tax_mean_concat = tax_mean_all.transpose(1, 0)  # (num_paths, layer, dim)
tax_mean_concat = tax_mean_concat.reshape(tax_mean_concat.shape[0], -1)  # Flatten layers
tax_mean_concat = torch.spmm(idx_adj_norm, tax_mean_concat)  # Aggregate to nodes
g.ndata['tax'] = tax_mean_concat  # Store taxonomy context
```
- `idx_adj_norm`: Maps taxonomy paths to nodes (with normalization)
- Each node now has aggregated taxonomy information from all its paths
- Shape: `(num_nodes, taxo_layers * hidden_dim)`

**Key Insight**: `g.ndata['tax']` is the taxonomy-augmented context for each node!

#### **Phase 4: Taxonomy-Aware GNN Convolution** (Knowledge Fusion)
```python
for i in range(self.num_layer):
    h = self.conv_layers[i](g)  # Custom convolution
    if i < self.num_layer - 1:
        h = self.activation(h)
    g.ndata['h'] = h
```

**Inside ConvLayer.forward()**:
```python
# Compute feature-based attention
g.apply_edges(self.edge_applying_fea)  # alpha_f from node features
alpha_f = edge_softmax(g, g.edata['wf'])

# Compute taxonomy-based attention  
g.apply_edges(self.edge_applying_tax)  # alpha_t from taxonomy similarity
alpha_t = edge_softmax(g, g.edata['wt'])

# Combine attentions
g.edata['alpha'] = self.eta * alpha_f + (1 - self.eta) * alpha_t

# Message passing with combined attention
g.update_all(fn.u_mul_e('h', 'alpha', '_'), fn.sum('_', 'z'))
h = g.ndata['z']
h = self.W(h)  # Transform
```

**This is where magic happens**:
- Traditional GNN: attention based only on node features
- Taxo-GNN: combines feature attention with **taxonomy similarity**
- `eta=1.0`: pure feature attention (standard GNN)
- `eta=0.0`: pure taxonomy attention
- `eta=0.5`: balanced hybrid

After this phase:
- `raw = h`: Raw node embeddings after taxonomy-aware convolution
- Shape: `(num_nodes, hidden_dim)`

#### **Phase 5: Generate Final Embeddings** (Layer-Aware Readout)

**5a. Node-specific embedding**:
```python
emb_h = self.mlps[-1](h)  # MLP transform
emb_h = F.normalize(emb_h, dim=1)  # L2 normalize
```
- Dimension: `(num_nodes, item_dim)`

**5b. Taxonomy-layer-specific embeddings**:
```python
emb_list = []
for i in range(self.taxo_layers):
    h_rev = torch.spmm(idx_adj, h)  # Aggregate node embeddings to paths
    
    # Compute blending weight between node and taxonomy
    stpl = self.psi[i](h_rev, tax_mean_all[i])  # Bilinear scoring
    stpl = torch.sigmoid(stpl).reshape(-1, 1)
    tau = stpl * torch.exp(-tax_std_all[i])  # Higher uncertainty → more taxonomy
    
    if tp_mask is not None:
        tau = tp_mask * tau  # Mask out test taxonomies
    
    # Blend node embedding with taxonomy embedding
    cur_emb = (1 - tau) * h_rev + tau * tax_mean_all[i]
    cur_emb = torch.spmm(idx_adj_norm, cur_emb)  # Back to nodes
    
    emb = self.mlps[i](cur_emb)  # MLP transform
    emb = F.normalize(emb, dim=1)  # L2 normalize
    emb_list.append(emb)
```

**Key Insight on τ (tau)**:
- `tau` controls blending: `embedding = (1-τ)*node + τ*taxonomy`
- When taxonomy std is high (uncertain) → `tau` is small → trust node embedding more
- When taxonomy std is low (confident) → `tau` is large → trust taxonomy more
- This is **adaptive information fusion**!

**5c. Concatenate all embeddings**:
```python
emb = torch.stack(emb_list, dim=2)
emb = emb.reshape(emb.shape[0], -1)  # Flatten layers
emb = torch.cat((emb_h, emb), dim=1)  # Add node-specific part
```

**Final output**:
- `raw`: shape `(num_nodes, hidden_dim)` - for taxonomy losses
- `emb`: shape `(num_nodes, out_dim)` - for downstream tasks
  - `out_dim = item_dim + taxo_layers * taxo_dim`

### Loss Functions

#### 1. `loss_rec` - Graph Reconstruction (NCE)
```python
def loss_rec(self, embeds1, embeds2, neg_sap):
```
- Positive pairs: connected nodes should be close
- Negative pairs: random nodes should be far
- Uses squared distance + log-sigmoid

#### 2. `cal_loss_leaf` - Leaf Taxonomy Loss
```python
def cal_loss_leaf(self, leafs, raw, pos_node_samples, neg_node_samples, k):
```
- For each leaf taxonomy, sample from its Gaussian
- Positive: sampled taxonomy vector should match member nodes
- Negative: shouldn't match random nodes
- This aligns leaf categories with their actual members

#### 3. `cal_loss_inner` - Inner Taxonomy Loss
```python
def cal_loss_inner(self, sample_num=20):
```
- Parent taxonomy should be a **mixture** of child taxonomies
- Sample from each child's Gaussian, take weighted average (by `taxo_c2p_prob`)
- Parent samples should match this mixture
- Ensures hierarchical consistency (parent = aggregate of children)

#### 4. `cal_loss_sim` - Taxonomy Similarity Loss
```python
def cal_loss_sim(self, tag1, tag2, neg, m):
```
- Uses **hybrid random walks** to find similar taxonomies
- If two taxonomies are reachable via graph walks → should be similar
- Margin loss: `max(0, m + dist(similar) - dist(dissimilar))`
- Uses Wasserstein distance between Gaussians

---

## Training on new-patent-small Dataset

### Step 1: Understand the Data Structure

The `new-patent-small` dataset contains:
```
edges.txt              # Graph edges (node_id1 node_id2)
features.npy           # Node features (sparse matrix for patent data)
taxo-tree.txt          # Taxonomy hierarchy (child parent)
taxo.txt               # Node-to-taxonomy mapping (node_idx taxo_id)
leafs.txt              # Leaf taxonomy IDs
inners.txt             # Inner (non-leaf) taxonomy IDs
taxo-c2p-prob.npy      # Child-to-parent probabilities
taxo-leaf-to-nodes.json # Which nodes belong to each leaf category
taxo-path.txt          # Each node's path through taxonomy
splits/                # Train/val/test splits for different tasks
```

### Step 2: Verify Data is Preprocessed

Check if preprocessing files exist:
```bash
ls -la "Taxo-GNN data/new-patent-small/"
```

If missing `taxo-tree-c2p.json`, `taxo-tree-p2c.json`, etc., you need to run preprocessing:

```python
# In preprocess.py, set:
dataset_name = 'new-patent-small'
remove_roots = False
```

Then uncomment the preprocessing functions and run.

### Step 3: Training Commands

#### Option A: Link Prediction
```bash
python train.py \
    --dataset "Taxo-GNN data/new-patent-small" \
    --device cuda:0 \
    --seed 42 \
    --epochs 1000 \
    --patience 500 \
    --hidden 32 \
    --out_dim 64 \
    --num_layer 2 \
    --sample_num 32 \
    --lr 0.01 \
    --dropout 0.5 \
    --eta 1.0 \
    --lamb 1.0 \
    --theta 1.0 \
    --beta 1.0 \
    --trial 0
```

#### Option B: Node Classification (if node-labels.txt exists)
```bash
python train_nc.py \
    --dataset "Taxo-GNN data/new-patent-small" \
    --device cuda:0 \
    --epochs 1000 \
    --hidden 32 \
    --out_dim 64 \
    --trial 0
```

#### Option C: Taxonomy Prediction
```bash
python train_tp.py \
    --dataset "Taxo-GNN data/new-patent-small" \
    --device cuda:0 \
    --epochs 1000 \
    --hidden 32 \
    --out_dim 64 \
    --sample_num 20 \
    --trial 0
```

### Step 4: Key Hyperparameters

| Parameter | Meaning | Typical Value |
|-----------|---------|---------------|
| `--hidden` | Hidden dimension | 32-128 |
| `--out_dim` | Output embedding dimension | 64-256 |
| `--num_layer` | Number of GNN layers | 2-3 |
| `--eta` | Feature vs taxonomy weight | 0.5-1.0 |
| `--lamb` | Taxonomy similarity loss weight | 0.1-1.0 |
| `--theta` | Leaf taxonomy loss weight | 0.5-1.0 |
| `--beta` | Inner taxonomy loss weight | 0.5-1.0 |
| `--sample_num` | Nodes per taxonomy for loss | 20-50 |

### Step 5: Check if Training Works

Expected output:
```
Epoch: 0001 loss_train: 2.5432 auc_train: 0.6234 auc_val: 0.5987 auc_test: 0.6012 time: 0.234s
Epoch: 0002 loss_train: 2.3421 auc_train: 0.6543 auc_val: 0.6234 auc_test: 0.6187 time: 0.198s
...
```

### Common Issues and Solutions

**Issue 1**: `FileNotFoundError` for splits
```bash
# Generate splits first
cd "Taxo-GNN data/new-patent-small"
# Make sure splits/ directory exists with train/val/test files
```

**Issue 2**: CUDA out of memory
```bash
# Reduce batch size or hidden dimensions
--hidden 16 --out_dim 32 --rbatch_size 128
```

**Issue 3**: Path issues with dataset name
```python
# In utils.py, load_data functions expect path like:
path = "../data"  # or "Taxo-GNN data"
dataset = "new-patent-small"
# Full path: "../data/new-patent-small/edges.txt"
```

**Fix**: Either:
- Move data to `../data/new-patent-small/`
- Change `path` parameter in train scripts
- Modify load functions to use correct path

---

## Data Flow Architecture - Visual Summary

```
INPUT: Graph + Node Features + Taxonomy Tree
   │
   ├─ Node Features (N x d) ──────┐
   │                              │
   └─ Taxonomy Structure          │
      │                           │
      ├─ Gaussian Parameters      │
      │  (μ, σ for each category) │
      │                           │
      │                           ▼
      │                    ┌─────────────┐
      │                    │  Project to │
      │                    │  hidden_dim │
      │                    └──────┬──────┘
      │                           │
      │                           ▼
      │                    ┌─────────────────┐
      └──────────────────► │ Extract Taxonomy│
                           │ Context (μ, σ)  │
                           └────────┬─────────┘
                                   │
                                   ▼
                           ┌────────────────────┐
                           │ Compute Dual       │
                           │ Attention:         │
                           │ α = η·α_f +        │
                           │     (1-η)·α_t      │
                           └──────┬─────────────┘
                                 │
                                 ▼
                           ┌────────────────┐
                           │ GNN Convolution│ (repeated num_layer times)
                           │ with Taxonomy  │
                           │ Attention      │
                           └──────┬─────────┘
                                 │
                          ┌──────┴──────┐
                          │             │
                          ▼             ▼
                   ┌──────────┐   ┌─────────────┐
                   │   raw    │   │ Layer-wise  │
                   │ (N x h)  │   │ Readout:    │
                   └────┬─────┘   │ Blend node  │
                        │         │ + taxonomy  │
                        │         │ with τ      │
                        │         └──────┬──────┘
                        │                │
                        │                ▼
                        │         ┌─────────────┐
                        │         │ MLP + Norm  │
                        │         │ (per layer) │
                        │         └──────┬──────┘
                        │                │
                        │                ▼
                        │         ┌─────────────┐
                        │         │ Concatenate │
                        │         │ All Layers  │
                        │         └──────┬──────┘
                        │                │
                        ▼                ▼
                   ┌──────────────────────┐
                   │   OUTPUT:            │
                   │   raw (for losses)   │
                   │   emb (for tasks)    │
                   └──────────────────────┘
                           │
                ┌──────────┼─────────────┐
                │          │             │
                ▼          ▼             ▼
         ┌─────────┐  ┌────────┐  ┌──────────┐
         │ Link    │  │ Node   │  │ Taxonomy │
         │ Predict │  │ Class  │  │ Predict  │
         └─────────┘  └────────┘  └──────────┘
```

### Key Takeaways

1. **Dual Attention Mechanism**: Combines graph structure (feature-based) and taxonomy structure (similarity-based)

2. **Gaussian Taxonomy Embeddings**: Uncertainty-aware representations (mean + variance)

3. **Information Flow**: 
   - Bottom-up: Nodes inform taxonomy embeddings (distillation)
   - Top-down: Taxonomy guides node embedding learning (fusion)

4. **Adaptive Blending (τ)**: Learns when to trust node features vs taxonomy (based on taxonomy confidence)

5. **Multi-Task Learning**: Same model works for link prediction, classification, and taxonomy prediction

---

## Quick Start Summary

### To train on new-patent-small:

1. **Fix the data path**:
   ```python
   # Option 1: Rename folder (remove space)
   mv "Taxo-GNN data" data
   
   # Option 2: Update train scripts to use:
   --dataset "Taxo-GNN data/new-patent-small"
   ```

2. **Run Link Prediction**:
   ```bash
   python train.py --dataset new-patent-small --device cuda:0
   ```

3. **Monitor**:
   - Loss should decrease
   - AUC should increase (> 0.7 is good)
   - Early stopping based on validation AUC

4. **Experiment with hyperparameters**:
   - Increase `--eta` (0.5 → 1.0) for more graph focus
   - Increase `--lamb` for more taxonomy similarity learning
   - Adjust `--theta` and `--beta` for taxonomy loss balance

---

## Further Exploration

- **Visualize taxonomy embeddings**: Use t-SNE on `model.taxo_mean.weight`
- **Ablation studies**: Set `eta=1.0` (no taxonomy) vs `eta=0.5` (hybrid)
- **Taxonomy variance**: Check `torch.exp(model.taxo_std_log.weight)` to see uncertainty
- **Attention weights**: Inspect `g.edata['alpha']` to see which neighbors matter

Good luck with your Taxo-GNN experiments! 🚀
