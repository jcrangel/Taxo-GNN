"""
Script to explore and visualize the new-patent-small dataset structure
"""

import os
import numpy as np
import json
from collections import Counter

dataset_path = "Taxo-GNN data/new-patent-small"

print("="*80)
print("Exploring new-patent-small Dataset")
print("="*80)

# 1. Check what files exist
print("\n1. Dataset Files:")
print("-" * 40)
for root, dirs, files in os.walk(dataset_path):
    level = root.replace(dataset_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in sorted(files)[:10]:  # Show first 10 files
        print(f'{subindent}{file}')
    if len(files) > 10:
        print(f'{subindent}... and {len(files) - 10} more files')

# 2. Analyze edges
print("\n2. Graph Structure:")
print("-" * 40)
edges_file = os.path.join(dataset_path, "edges.txt")
if os.path.exists(edges_file):
    edges = np.loadtxt(edges_file, dtype=int)
    print(f"Number of edges: {len(edges):,}")
    print(f"Edge format (first 5 rows):")
    print(edges[:5])
    
    unique_nodes = np.unique(edges)
    print(f"Number of unique nodes: {len(unique_nodes):,}")
    print(f"Node ID range: [{unique_nodes.min()}, {unique_nodes.max()}]")
    
    # Degree distribution
    degrees = Counter()
    for e in edges:
        degrees[e[0]] += 1
        degrees[e[1]] += 1
    degree_values = list(degrees.values())
    print(f"\nDegree statistics:")
    print(f"  Min degree: {min(degree_values)}")
    print(f"  Max degree: {max(degree_values)}")
    print(f"  Average degree: {np.mean(degree_values):.2f}")

# 3. Analyze features
print("\n3. Node Features:")
print("-" * 40)
features_file = os.path.join(dataset_path, "features.npy")
if os.path.exists(features_file):
    try:
        # Patent datasets often have sparse features
        features = np.load(features_file, allow_pickle=True)
        if isinstance(features.item(), np.ndarray):
            features = features.item()
        print(f"Feature type: {type(features)}")
        if hasattr(features, 'shape'):
            print(f"Feature shape: {features.shape}")
            print(f"Feature dtype: {features.dtype}")
            if hasattr(features, 'nnz'):
                sparsity = (1 - features.nnz / (features.shape[0] * features.shape[1])) * 100
                print(f"Sparsity: {sparsity:.2f}% (sparse matrix)")
    except Exception as e:
        print(f"Error loading features: {e}")

# 4. Analyze taxonomy structure
print("\n4. Taxonomy Structure:")
print("-" * 40)

# Read taxonomy tree
taxo_tree_file = os.path.join(dataset_path, "taxo-tree.txt")
if os.path.exists(taxo_tree_file):
    with open(taxo_tree_file, 'r') as f:
        lines = f.readlines()
    print(f"Number of taxonomy relationships: {len(lines)}")
    print(f"Format (first 5 lines):")
    for line in lines[:5]:
        print(f"  {line.strip()}")

# Read processed taxonomy
taxo_p2c_file = os.path.join(dataset_path, "taxo-tree-p2c.json")
if os.path.exists(taxo_p2c_file):
    with open(taxo_p2c_file, 'r') as f:
        taxo_p2c = json.load(f)
    print(f"\nTaxonomy layers: {len(taxo_p2c)}")
    for layer_idx in sorted(taxo_p2c.keys(), key=int):
        layer_data = taxo_p2c[layer_idx]
        num_parents = len(layer_data)
        num_children = sum(len(v) for v in layer_data.values())
        print(f"  Layer {layer_idx}: {num_parents} parents, {num_children} total children")

# Read leaf taxonomies
leafs_file = os.path.join(dataset_path, "leafs.txt")
if os.path.exists(leafs_file):
    leafs = np.loadtxt(leafs_file, dtype=int)
    print(f"\nNumber of leaf taxonomies: {len(leafs)}")

# Read inner taxonomies
inners_file = os.path.join(dataset_path, "inners.txt")
if os.path.exists(inners_file):
    inners = np.loadtxt(inners_file, dtype=int)
    print(f"Number of inner taxonomies: {len(inners)}")

# 5. Node-taxonomy mapping
print("\n5. Node-Taxonomy Mapping:")
print("-" * 40)

taxo_path_file = os.path.join(dataset_path, "taxo-path.txt")
if os.path.exists(taxo_path_file):
    taxo_paths = np.loadtxt(taxo_path_file, dtype=int)
    print(f"Number of node-path instances: {len(taxo_paths):,}")
    print(f"Path format (columns): {taxo_paths.shape}")
    print(f"First 5 paths:")
    print(taxo_paths[:5])
    print(f"\nExplanation: Each row is [path_idx, node_idx, layer0_taxo, layer1_taxo, ...]")
    
    # Count paths per node
    unique_nodes_in_paths = len(np.unique(taxo_paths[:, 1]))
    avg_paths_per_node = len(taxo_paths) / unique_nodes_in_paths
    print(f"Average paths per node: {avg_paths_per_node:.2f}")

# Leaf-to-nodes mapping
leaf_nodes_file = os.path.join(dataset_path, "taxo-leaf-to-nodes.json")
if os.path.exists(leaf_nodes_file):
    with open(leaf_nodes_file, 'r') as f:
        leaf2nodes = json.load(f)
    node_counts = [len(nodes) for nodes in leaf2nodes.values()]
    print(f"\nLeaf taxonomy node counts:")
    print(f"  Number of leaf categories: {len(leaf2nodes)}")
    print(f"  Min nodes per leaf: {min(node_counts) if node_counts else 0}")
    print(f"  Max nodes per leaf: {max(node_counts) if node_counts else 0}")
    print(f"  Average nodes per leaf: {np.mean(node_counts) if node_counts else 0:.2f}")

# 6. Training splits
print("\n6. Training Splits:")
print("-" * 40)

splits_dir = os.path.join(dataset_path, "splits")
if os.path.exists(splits_dir):
    split_files = [f for f in os.listdir(splits_dir) if f.endswith('.txt')]
    
    # Group by task
    tasks = {}
    for f in split_files:
        task = f.split('_')[-2]  # lp, nc, or tp
        if task not in tasks:
            tasks[task] = []
        tasks[task].append(f)
    
    for task_name, files in sorted(tasks.items()):
        print(f"\n  {task_name.upper()} Task:")
        # Count splits
        train_splits = [f for f in files if 'train' in f]
        val_splits = [f for f in files if 'val' in f]
        test_splits = [f for f in files if 'test' in f]
        print(f"    Number of splits: {len(train_splits)}")
        
        # Show sizes for first split
        if train_splits:
            split_idx = 0
            train_file = os.path.join(splits_dir, f"train_idx_{task_name}_{split_idx}.txt")
            if os.path.exists(train_file):
                train_idx = np.loadtxt(train_file, dtype=int)
                val_file = os.path.join(splits_dir, f"val_idx_{task_name}_{split_idx}.txt")
                test_file = os.path.join(splits_dir, f"test_idx_{task_name}_{split_idx}.txt")
                
                print(f"    Split 0 sizes:")
                print(f"      Train: {len(train_idx):,}")
                if os.path.exists(val_file):
                    val_idx = np.loadtxt(val_file, dtype=int)
                    print(f"      Val: {len(val_idx):,}")
                if os.path.exists(test_file):
                    test_idx = np.loadtxt(test_file, dtype=int)
                    print(f"      Test: {len(test_idx):,}")

# 7. Summary
print("\n" + "="*80)
print("Summary")
print("="*80)
print("\nThe new-patent-small dataset is a:")
print("  ✓ Graph with nodes and edges (patent citation network)")
print("  ✓ Node features (likely bag-of-words or similar)")
print("  ✓ Hierarchical taxonomy (multi-layer category tree)")
print("  ✓ Node-category associations (nodes belong to taxonomy leaves)")
print("  ✓ Pre-split data for multiple tasks (LP, NC, TP)")
print("\nYou can train Taxo-GNN on this dataset for:")
print("  1. Link Prediction (LP): Predict missing edges")
print("  2. Node Classification (NC): Predict node categories")  
print("  3. Taxonomy Prediction (TP): Infer categories from partial info")

print("\n" + "="*80)
