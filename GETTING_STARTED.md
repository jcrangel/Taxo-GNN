# Getting Started Checklist

## ✅ Understanding Phase

- [ ] Read `ARCHITECTURE_SUMMARY.md` - High-level overview
- [ ] Read `UNDERSTANDING_TAXO_GNN.md` - Deep dive into implementation
- [ ] Understand the three key components:
  - [ ] Gaussian taxonomy embeddings (μ, σ)
  - [ ] Dual attention mechanism (η parameter)
  - [ ] Adaptive blending (τ parameter)

## ✅ Setup Phase

- [ ] Verify Python environment has required packages:
  ```bash
  pip install torch dgl numpy scipy scikit-learn networkx
  ```

- [ ] Check dataset exists:
  ```bash
  ls -la "Taxo-GNN data/new-patent-small/"
  ```

- [ ] Explore the dataset:
  ```bash
  python explore_dataset.py
  ```

## ✅ Initial Training

- [ ] Run a quick test (fewer epochs to verify setup):
  ```bash
  python train_new_patent_small.py --epochs 50 --device cuda:0
  ```

- [ ] Check outputs:
  - [ ] Loss decreasing?
  - [ ] AUC increasing?
  - [ ] No errors?

- [ ] If working, run full training:
  ```bash
  python train_new_patent_small.py --epochs 1000 --device cuda:0
  ```

## ✅ Understanding Results

- [ ] Check final AUC scores (printed at end)
- [ ] Verify embeddings saved: `embeddings/taxo_gnn_new_patent_small_trial0.npy`
- [ ] Review taxonomy statistics (mean/std ranges printed at end)

## ✅ Experimentation

- [ ] Baseline: Pure graph (no taxonomy)
  ```bash
  python train_new_patent_small.py --eta 1.0 --lamb 0.0 --trial 1
  ```

- [ ] Balanced: Graph + Taxonomy
  ```bash
  python train_new_patent_small.py --eta 0.5 --lamb 1.0 --trial 2
  ```

- [ ] Taxonomy-heavy:
  ```bash
  python train_new_patent_small.py --eta 0.3 --lamb 2.0 --trial 3
  ```

- [ ] Compare results across trials

## ✅ Analysis

- [ ] Load and inspect embeddings:
  ```python
  import numpy as np
  emb = np.load('embeddings/taxo_gnn_new_patent_small_trial0.npy')
  print(f"Embedding shape: {emb.shape}")
  ```

- [ ] Visualize if desired (t-SNE, PCA)
- [ ] Use embeddings for downstream tasks

## 🎯 Success Criteria

You've successfully understood and trained Taxo-GNN if:

1. ✅ You can explain the difference between information distillation and knowledge fusion
2. ✅ You understand what η, τ, and the Gaussian parameters do
3. ✅ Training completes without errors
4. ✅ Validation AUC > 0.7 (decent), > 0.8 (good)
5. ✅ You can explain why there are three training files
6. ✅ You can run experiments with different hyperparameters

## 📚 Quick Reference Files

| File | Purpose | When to Read |
|------|---------|--------------|
| `QUICK_REFERENCE.md` | Commands and tips | Before training |
| `ARCHITECTURE_SUMMARY.md` | Visual overview | For understanding |
| `UNDERSTANDING_TAXO_GNN.md` | Complete guide | For deep dive |
| `explore_dataset.py` | Dataset analysis | Before training |
| `train_new_patent_small.py` | Ready-to-use training | For training |

## 🚀 Next Steps

After completing the checklist:

1. **Experiment**: Try different hyperparameters (see `QUICK_REFERENCE.md`)
2. **Analyze**: Inspect taxonomy embeddings, attention weights
3. **Extend**: Try other datasets in the repo (Aminer, news, rocket)
4. **Apply**: Use learned embeddings for your downstream tasks

## 💡 Common Questions (Quick Answers)

**Q: Which training script to use?**  
A: `train_new_patent_small.py` for new-patent-small dataset

**Q: What's a good starting η?**  
A: 0.7 (balanced) or 1.0 (baseline)

**Q: How long does training take?**  
A: 5-30 minutes depending on hardware (GPU recommended)

**Q: What if loss isn't decreasing?**  
A: Lower learning rate: `--lr 0.001`

**Q: What if validation AUC is low (<0.6)?**  
A: Try different η values or adjust loss weights

**Q: How do I know if taxonomy is helping?**  
A: Compare η=1.0 (no taxonomy) vs η<1.0 (with taxonomy)

## 📧 Need Help?

1. Check error messages carefully
2. Review `UNDERSTANDING_TAXO_GNN.md` for detailed explanations
3. Try the debugging tips in `QUICK_REFERENCE.md`
4. Inspect model outputs (print intermediate values)

---

Happy training! 🎉
