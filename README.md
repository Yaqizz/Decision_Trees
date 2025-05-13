# Code for Introduction to ML CW1 - Decision Tree

**Team Project**

## Table of Contents
- [Overview](#overview)
- [Run](#run)
  - [Setup](#setup)
  - [Running the Method](#running-the-method)
  - [Example Outputs](#example-outputs)
  - [Visualize Dataset](#visualize-dataset)
- [Function Design](#function-design)
  - [Tree Class Methods](#tree-class-methods-13)
  - [Helper Functions](#helper-functions-4)

## Overview
This repository contains our implementation of a **decision tree algorithm** for the **classification task** of the indoor locations based on WIFI signals strengths.

## Run

### Setup
Make sure you have installed `numpy` and `matplotlib`.
```
python -c "import numpy as np; print(np.__version__); import matplotlib; print(matplotlib.__version__)"
```

### Running the Method
1. Modify Line 376 in `main.py` to load your own datasets:

```
# Default path is "data/source/noisy_dataset.txt"
data = np.loadtxt('path/to/your/dataset') 
```

2. Run `main.py` and it will automatically gives output and figures of unpruned decision tree (and its confusion matrix) first, followed by results and figures of pruned decision tree.

### Example Outputs

1. Unpruned Decision Tree trained on noisy dataset:
![](./assets/Unpruned_Decision_Tree.png)

2. Confusion Matrix of previous Decision Tree:
![](./assets/Unpruned_Confusion_Matrix_Noisy_Dataset.png)

3. Pruned Decision Tree trained on noisy dataset:
![](./assets/Pruned_Decision_Tree.png)

4. Confusion Matrix of previous Decision Tree:
![](./assets/Pruned_Confusion_Matrix_Noisy_Data.png)

### Visualize Dataset

we utilized t-distributed stochastic neighbor embedding (t-SNE) to visualize the differences between clean and noisy set.

Run `tsne_visualization_diff_data.py`

---

## Function Design

### Tree Class Methods (13)

#### Training (6)
- `__init__()`: Initialize empty tree
- `fit(X, y)`: Train model
- `_build_tree(X, y)`: Recursive tree construction
- `find_split(X, y)`: Find best split point
- `information_gain(parent, left, right)`: Calculate split gain
- `entropy(y)`: Calculate node entropy

#### Prediction (2)
- `predict(X)`: Predict multiple samples
- `_predict_sample(sample, tree)`: Predict single sample

#### Evaluation (2)
- `nested_cross_validation(X, y, outer_k, inner_k, use_pruning)`: Evaluate model
- `evaluate(confusion_matrix)`: Calculate metrics

#### Pruning (1)
- `prune(X_val, y_val)`: Prune tree using validation data

#### Visualization (2)
- `plot_confusion_matrix(confusion_matrix)`: Plot confusion matrix
- `plot_tree(node, depth, pos, ax, width, gap)`: Plot tree structure

### Helper Functions (4)

#### Pruning Helpers (3)
- `_get_node_error(node, X, y)`: Count misclassifications
- `_is_decision_node(node)`: Check node type
- `_prune_node(node, X, y)`: Perform pruning

#### Utility (1)
- `load_data(filepath)`: Load and preprocess data

Total: 17 functions organized by their roles in the implementation


