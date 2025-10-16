# Taxo-GNN Quick Reference

## Quick Start Training

### Option 1: Using Custom Script (Recommended for new-patent-small)
```bash
# Simple command
python train_new_patent_small.py --device cuda:0 --epochs 500

# With custom hyperparameters
python train_new_patent_small.py \
    --device cuda:0 \
    --epochs 1000 \
    --hidden 64 \
    --out_dim 128 \
    --eta 0.7 \
    --lamb 0.5
```

### Option 2: Using Original Scripts
```bash
# Link Prediction
python train.py --dataset "Taxo-GNN data/new-patent-small" --device cuda:0

# Node Classification (if labels exist)
python train_nc.py --dataset "Taxo-GNN data/new-patent-small" --device cuda:0

# Taxonomy Prediction
python train_tp.py --dataset "Taxo-GNN data/new-patent-small" --device cuda:0
```

## Explore Dataset First
```bash
python explore_dataset.py
```

## Key Hyperparameters

| Parameter | What it does | Typical Range | Default |
|-----------|-------------|---------------|---------|
| `--eta` | Balance between graph and taxonomy attention<br>1.0 = pure graph, 0.0 = pure taxonomy | 0.3-1.0 | 1.0 |
| `--lamb` | Weight for taxonomy similarity loss | 0.1-2.0 | 1.0 |
| `--theta` | Weight for leaf taxonomy loss | 0.5-2.0 | 1.0 |
| `--beta` | Weight for inner taxonomy loss | 0.5-2.0 | 1.0 |
| `--hidden` | Hidden dimension size | 16-128 | 32 |
| `--out_dim` | Output embedding dimension | 32-256 | 64 |
| `--num_layer` | Number of GNN layers | 2-4 | 2 |
| `--dropout` | Dropout rate | 0.0-0.7 | 0.5 |

## Hyperparameter Tuning Tips

### More Graph-Focused
```bash
--eta 1.0 --lamb 0.1 --theta 0.5 --beta 0.5
```
Best for: Graphs with weak taxonomy signal

### More Taxonomy-Focused
```bash
--eta 0.5 --lamb 2.0 --theta 2.0 --beta 2.0
```
Best for: Clean, informative taxonomies

### Balanced (Recommended Start)
```bash
--eta 0.7 --lamb 1.0 --theta 1.0 --beta 1.0
```
Best for: General case, good starting point

### For Small Datasets
```bash
--hidden 16 --out_dim 32 --dropout 0.3 --num_layer 2
```

### For Large Datasets
```bash
--hidden 64 --out_dim 128 --dropout 0.5 --num_layer 3 --rbatch_size 512
```

## Understanding the Output

### During Training
```
Epoch: 0010 | Loss: 3.4521 (rec:2.1 sim:0.8 leaf:0.3 inner:0.2) | 
AUC - Train: 0.7234 Val: 0.6987 Test: 0.7012 | Time: 0.234s
```

- **rec**: Graph reconstruction loss (should decrease)
- **sim**: Taxonomy similarity loss (should decrease)
- **leaf**: Leaf category alignment (should decrease)
- **inner**: Hierarchical consistency (should decrease)
- **AUC**: Link prediction performance (should increase)
  - > 0.7 is decent
  - > 0.8 is good
  - > 0.9 is excellent

### Final Results
```
Best Results:
  Epoch: 234
  Validation AUC: 0.8234
  Test AUC: 0.8187
```

## File Outputs

### Embeddings
Saved to: `embeddings/taxo_gnn_new_patent_small_trial0.npy`

Load them:
```python
import numpy as np
emb = np.load('embeddings/taxo_gnn_new_patent_small_trial0.npy')
print(f"Shape: {emb.shape}")  # (num_nodes, out_dim)

# Use for downstream tasks
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(emb[train_idx], labels[train_idx])
```

## Common Issues & Solutions

### Issue: CUDA out of memory
```bash
# Solution 1: Smaller model
--hidden 16 --out_dim 32 --rbatch_size 128

# Solution 2: Use CPU
--device cpu
```

### Issue: Training is slow
```bash
# Solution: Reduce samples and batch size
--sample_num 16 --rbatch_size 128 --max_len 3
```

### Issue: Model not converging (loss not decreasing)
```bash
# Solution 1: Lower learning rate
--lr 0.001

# Solution 2: Adjust loss weights
--lamb 0.5 --theta 0.5 --beta 0.5

# Solution 3: More dropout
--dropout 0.6
```

### Issue: Validation performance not improving
```bash
# Solution 1: Early stopping is working - let it run
--patience 500

# Solution 2: Try different eta
--eta 0.5  # or 0.3, 0.7, 1.0
```

### Issue: FileNotFoundError
```bash
# Make sure you're in the right directory
cd /home/julio/repos/Taxo-GNN
python train_new_patent_small.py

# Or check the data folder name (has space!)
ls -la "Taxo-GNN data/new-patent-small/"
```

## Experiment Tracking

### Compare Different Settings
```bash
# Experiment 1: Pure graph
python train_new_patent_small.py --eta 1.0 --lamb 0.0 --trial 1

# Experiment 2: Hybrid
python train_new_patent_small.py --eta 0.5 --lamb 1.0 --trial 2

# Experiment 3: Taxonomy-heavy
python train_new_patent_small.py --eta 0.3 --lamb 2.0 --trial 3
```

Track in a spreadsheet:
| Trial | eta | lamb | theta | beta | Val AUC | Test AUC |
|-------|-----|------|-------|------|---------|----------|
| 0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.823 | 0.819 |
| 1 | 0.5 | 1.0 | 1.0 | 1.0 | 0.841 | 0.835 |
| ... | ... | ... | ... | ... | ... | ... |

## Model Inspection

### Check Taxonomy Embeddings
```python
import torch
from models import TaxoGNN

# After training, load model
# model = ...

# Inspect taxonomy means
means = model.taxo_mean.weight.data  # (num_taxo, hidden_dim)
stds = torch.exp(model.taxo_std_log.weight.data)

print(f"Mean range: [{means.min():.3f}, {means.max():.3f}]")
print(f"Std range: [{stds.min():.3f}, {stds.max():.3f}]")

# High std = uncertain category
# Low std = confident category
```

### Visualize with t-SNE
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Reduce dimensions
means_2d = TSNE(n_components=2).fit_transform(means.cpu().numpy())

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(means_2d[:, 0], means_2d[:, 1], alpha=0.5)
plt.title("Taxonomy Embeddings (t-SNE)")
plt.savefig("taxonomy_tsne.png")
```

## Next Steps

1. **Run exploration**: `python explore_dataset.py`
2. **Start training**: `python train_new_patent_small.py`
3. **Try different eta values**: See what works best
4. **Inspect results**: Check embeddings and taxonomy stats
5. **Compare**: Run ablation studies (with/without taxonomy)

## Further Reading

- Full documentation: `UNDERSTANDING_TAXO_GNN.md`
- Paper: [Taxonomy-Enhanced GNN (CIKM 2022)](https://dl.acm.org/doi/10.1145/3511808.3557467)
- Code files: `models.py`, `utils.py`, `train*.py`

## Questions?

**Q: What's the difference between the three training scripts?**
A: Different downstream tasks:
- `train.py`: Link prediction (predict edges)
- `train_nc.py`: Node classification (predict labels)
- `train_tp.py`: Taxonomy prediction (predict categories)

**Q: Which one should I use?**
A: Start with link prediction (`train.py` or `train_new_patent_small.py`)

**Q: What does eta control?**
A: Balance between graph structure (features) and taxonomy structure
- eta=1.0: Standard GNN (no taxonomy)
- eta<1.0: Hybrid with taxonomy
- eta=0.0: Pure taxonomy (no graph features)

**Q: How do I know if it's working?**
A: Watch the validation AUC - should increase over epochs and reach > 0.7

**Q: Can I use this on my own dataset?**
A: Yes! Format your data like the example datasets:
- `edges.txt`: Graph edges
- `features.npy`: Node features  
- `taxo-tree.txt`: Taxonomy hierarchy
- `taxo.txt`: Node-category mapping
Then run preprocessing.

