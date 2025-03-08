## Node Classification on Cora 

This project explores the task of node classification in the Cora dataset, a citation network where nodes represent scientific papers and edges represent citations. Each node has 1433 binary features, and nodes belong to one of seven categories, which are the target classes for classification.

### Approach Overview

We compare three approaches for solving the node classification problem:

1. **Baseline Model**: 
   - This model reduces the feature dimensions using Principal Component Analysis (PCA) and uses the adjacency matrix of the graph for classification.

2. **DeepWalk**: 
   - DeepWalk generates node embeddings by performing random walks on the graph, capturing the graph structure to improve classification performance. For further details, check the [DeepWalk paper](https://arxiv.org/pdf/1403.6652).

3. **GraphSAGE**:
   - GraphSAGE generates node embeddings by aggregating features from a node's neighbors. It allows scalable node classification by sampling neighbors and generalizing to unseen nodes. For further details, check the [GraphSAGE paper](https://arxiv.org/pdf/1706.02216).

### Notebooks and Implementation

- The results, discussions, and visualizations are provided in the `notebooks` folder, while the DeepWalk and GraphSAGE implementations are located in the `src` folder.

