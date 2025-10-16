# Taxo-GNN Architecture Summary

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       Taxo-GNN Model                            │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Information Distillation                   │    │
│  │         (Graph → Taxonomy, Bottom-Up)                  │    │
│  │                                                         │    │
│  │  • Node features inform taxonomy embeddings            │    │
│  │  • Leaf categories learn from member nodes             │    │
│  │  • Parent categories aggregate children                │    │
│  └────────────────────────────────────────────────────────┘    │
│                            ↓                                    │
│  ┌────────────────────────────────────────────────────────┐    │
│  │               Knowledge Fusion                          │    │
│  │         (Taxonomy → Graph, Top-Down)                   │    │
│  │                                                         │    │
│  │  • Taxonomy-aware attention weights                     │    │
│  │  • Adaptive blending (τ) of node + taxonomy            │    │
│  │  • Layer-wise taxonomy-guided aggregation              │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Taxonomy Representation (Gaussian Distributions)

```python
taxo_mean     : (num_taxo, hidden_dim)  # Mean vector μ
taxo_std_log  : (num_taxo, hidden_dim)  # Log std log(σ)

# Each taxonomy is: N(μ, σ²)
# Captures uncertainty in category representation
```

**Why Gaussian?**
- Represents category as distribution (not single point)
- Variance captures uncertainty/breadth of category
- Enables probabilistic hierarchy modeling

### 2. Dual Attention Mechanism

```
ConvLayer Attention:

α = η · α_f + (1-η) · α_t

where:
  α_f = softmax(LeakyReLU(W[h_src || h_dst]))  # Feature-based
  α_t = softmax(⟨tax_src, tax_dst⟩)            # Taxonomy-based

η controls the balance:
  η = 1.0 → Pure graph (standard GNN)
  η = 0.5 → Balanced hybrid
  η = 0.0 → Pure taxonomy
```

### 3. Adaptive Blending (τ)

```python
# For each taxonomy layer i:
stpl = sigmoid(Bilinear(h_node, μ_taxo))  # Compatibility score
τ = stpl · exp(-σ_taxo²)                  # Less uncertainty → higher τ

# Blend node and taxonomy:
embedding = (1 - τ) · h_node + τ · μ_taxo

# Intuition:
# • High taxonomy uncertainty → trust node features (small τ)
# • Low taxonomy uncertainty → trust taxonomy (large τ)
```

## Data Flow (Step-by-Step)

```
INPUT
  ├─ Graph: G = (V, E)
  ├─ Features: X ∈ ℝ^(N×d)
  └─ Taxonomy: T (hierarchical tree)

STEP 1: Initialize
  h = PReLU(W·X + dropout)                    → (N, h_dim)

STEP 2: Extract Taxonomy Context
  For each taxonomy layer i:
    μ_i, σ_i = lookup taxonomy embeddings
  
  tax_context = concat(μ_1, μ_2, ..., μ_L)
  tax_context = aggregate_to_nodes(tax_context)  → (N, L·h_dim)
  
  Store: g.ndata['h'] = h
         g.ndata['tax'] = tax_context

STEP 3: GNN Convolution (repeated num_layer times)
  For each GNN layer:
    # Compute dual attention
    α_f = feature_attention(h_src, h_dst)
    α_t = taxonomy_attention(tax_src, tax_dst)
    α = η·α_f + (1-η)·α_t
    
    # Message passing
    h = Σ_{j∈N(i)} α_ij · h_j
    h = W(h)
    
  raw = h                                       → (N, h_dim)

STEP 4: Generate Task Embeddings
  # Node-specific part
  emb_node = normalize(MLP(raw))                → (N, item_dim)
  
  # Taxonomy-layer parts
  For each taxonomy layer i:
    h_paths = aggregate_nodes_to_paths(raw)
    
    # Compute blending weight
    τ = sigmoid(Bilinear(h_paths, μ_i)) · exp(-σ_i²)
    
    # Blend
    h_blend = (1-τ)·h_paths + τ·μ_i
    h_nodes = aggregate_paths_to_nodes(h_blend)
    
    emb_i = normalize(MLP_i(h_nodes))           → (N, taxo_dim)
  
  # Concatenate all parts
  emb = [emb_node || emb_1 || emb_2 || ...]    → (N, out_dim)

OUTPUT
  ├─ raw: for taxonomy losses
  └─ emb: for downstream tasks (LP, NC, TP)
```

## Loss Functions

```
Total Loss = L_rec + λ·L_sim + θ·L_leaf + β·L_inner

┌────────────────────────────────────────────────────────────────┐
│ L_rec: Graph Reconstruction (NCE)                              │
│   • Sample edges: (u, v) positive, random negative            │
│   • Minimize: -log σ(-d(u,v)) + Σ_k -log σ(d(u, neg_k))      │
│   • Ensures: Connected nodes → similar embeddings             │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ L_sim: Taxonomy Similarity (Random Walk)                       │
│   • Hybrid walk: alternates taxonomy ↔ node ↔ taxonomy        │
│   • Co-visited taxonomies should be similar                    │
│   • Margin loss: max(0, m + d(sim) - d(dissim))              │
│   • Uses Wasserstein distance between Gaussians               │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ L_leaf: Leaf Taxonomy Alignment                                │
│   • Sample from leaf Gaussian: z ~ N(μ_leaf, σ_leaf²)         │
│   • Positive: member nodes should match z                      │
│   • Negative: random nodes shouldn't match z                   │
│   • Ensures: Leaf categories align with actual members        │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ L_inner: Hierarchical Consistency                              │
│   • Parent should be mixture of children                       │
│   • Sample from parent: z_p ~ N(μ_p, σ_p²)                    │
│   • Sample from children: z_c ~ Σ_i p_i·N(μ_ci, σ_ci²)       │
│   • Minimize: BCE(z_p, z_c)                                   │
│   • Weighted by layer depth (higher layers matter more)        │
└────────────────────────────────────────────────────────────────┘
```

## Key Design Choices

### 1. Why Gaussian Embeddings?

**Problem**: Categories have varying specificity
- "Computer Science" (broad) vs "Deep Learning" (narrow)

**Solution**: Gaussian with variance
- Broad categories → high variance
- Narrow categories → low variance
- Enables hierarchical modeling via mixture

### 2. Why Dual Attention?

**Problem**: Graph structure and taxonomy may conflict
- Graph says: nodes A, B similar (connected)
- Taxonomy says: nodes A, B dissimilar (different categories)

**Solution**: Learn to balance both signals
- η parameter controls trade-off
- Model learns which signal to trust per dataset

### 3. Why Adaptive Blending (τ)?

**Problem**: Not all taxonomies equally reliable
- Some categories well-defined (low uncertainty)
- Others vague or noisy (high uncertainty)

**Solution**: Trust proportional to confidence
- High confidence taxonomy → use it more
- Low confidence taxonomy → trust node features
- Per-node, per-layer adaptation

## Comparison with Standard GNN

```
┌────────────────────┬──────────────────┬────────────────────┐
│                    │   Standard GNN   │    Taxo-GNN        │
├────────────────────┼──────────────────┼────────────────────┤
│ Attention          │ Feature-based    │ Feature + Taxonomy │
│                    │ (node features)  │ (dual attention)   │
├────────────────────┼──────────────────┼────────────────────┤
│ Aggregation        │ Neighbor avg     │ Taxonomy-weighted  │
│                    │                  │ neighbor avg       │
├────────────────────┼──────────────────┼────────────────────┤
│ Context            │ Local graph      │ Local graph +      │
│                    │                  │ global taxonomy    │
├────────────────────┼──────────────────┼────────────────────┤
│ Embedding          │ Single vector    │ Multi-scale:       │
│                    │ per node         │ node + per layer   │
├────────────────────┼──────────────────┼────────────────────┤
│ Inductive bias     │ Homophily        │ Homophily +        │
│                    │                  │ hierarchical       │
└────────────────────┴──────────────────┴────────────────────┘
```

## When Does Taxo-GNN Excel?

### ✅ Good Scenarios:
1. **Informative taxonomy**: Categories align well with task
2. **Heterogeneous graphs**: Nodes of different types
3. **Long-range dependencies**: Taxonomically similar but graph-distant
4. **Limited labels**: Taxonomy provides extra supervision
5. **Hierarchical structure matters**: Parent-child relationships important

### ⚠️ Challenging Scenarios:
1. **Noisy taxonomy**: Categories don't match graph structure
2. **Homogeneous graphs**: All nodes similar type
3. **Strong graph signal**: Graph structure alone sufficient
4. **Flat taxonomy**: No meaningful hierarchy (single layer)

## Practical Tips

### Hyperparameter Selection:

```python
# Graph-dominant setting (strong homophily)
eta = 1.0      # Trust graph
lamb = 0.1     # Light taxonomy regularization

# Balanced setting (typical)
eta = 0.5-0.7  # Mix graph + taxonomy
lamb = 1.0     # Equal taxonomy weight

# Taxonomy-dominant (weak graph, good taxonomy)
eta = 0.3      # Trust taxonomy more
lamb = 2.0     # Strong taxonomy guidance
```

### Debugging:

```python
# Check if taxonomy is being used:
# - Print attention weights: g.edata['alpha']
# - High α_t → taxonomy matters
# - High α_f → features matter

# Check blending:
# - Print τ values
# - High τ → trusting taxonomy
# - Low τ → trusting nodes

# Check taxonomy embeddings:
# - Print std: torch.exp(model.taxo_std_log.weight)
# - High std → uncertain categories
# - Low std → confident categories
```

## Example Use Cases

### 1. Scientific Paper Classification
- **Graph**: Citation network
- **Taxonomy**: Research field hierarchy (CS → ML → Deep Learning)
- **Task**: Predict paper topics
- **Why Taxo-GNN**: Papers cite across fields; taxonomy provides structure

### 2. E-commerce Product Recommendation
- **Graph**: Co-purchase network
- **Taxonomy**: Product category tree (Electronics → Computers → Laptops)
- **Task**: Link prediction (recommend products)
- **Why Taxo-GNN**: Categories guide what products are similar

### 3. Patent Analysis (new-patent-small)
- **Graph**: Patent citations
- **Taxonomy**: Technology classification (IPC codes)
- **Task**: Predict patent relationships
- **Why Taxo-GNN**: Technology categories reveal innovation pathways

## Mathematical Notation Summary

```
Symbols:
  G = (V, E)              Graph
  X ∈ ℝ^(N×d)            Node features
  T                       Taxonomy tree
  μ_t ∈ ℝ^h              Taxonomy mean
  σ_t ∈ ℝ^h              Taxonomy std
  h_i ∈ ℝ^h              Node embedding
  α_ij ∈ [0,1]           Attention weight
  τ ∈ [0,1]              Blending weight
  η ∈ [0,1]              Attention trade-off

Dimensions:
  N                       Number of nodes
  E                       Number of edges
  d                       Input feature dimension
  h                       Hidden dimension
  T                       Number of taxonomies
  L                       Number of taxonomy layers
```

## References

- **Paper**: Taxonomy-Enhanced Graph Neural Networks (CIKM 2022)
- **Key Innovation**: Joint learning of graph and taxonomy structure
- **Main Contribution**: Dual attention + adaptive blending mechanisms

---

For complete details, see `UNDERSTANDING_TAXO_GNN.md`
For quick commands, see `QUICK_REFERENCE.md`
