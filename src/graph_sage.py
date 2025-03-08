import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, agg_type="mean"):
        super().__init__()
        self.agg_type = agg_type  
        self.linear = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x, edge_index):
        """
        x: Node feature matrix (num_nodes, in_channels)
        edge_index: Graph edges (2, num_edges)
        """
        row, col = edge_index  # Source and target nodes
        neighbor_feats = x[col]  # Features of neighboring nodes

        if self.agg_type == "mean":
            agg_feats = torch.zeros_like(x)
            agg_feats.index_add_(0, row, neighbor_feats)
            agg_feats /= torch.bincount(row, minlength=x.size(0)).unsqueeze(1) + 1e-6  # Avoid division by zero
        elif self.agg_type == "max":
            agg_feats = torch.zeros_like(x)
            agg_feats.index_copy_(0, row, torch.max(neighbor_feats, dim=0)[0])
        else:
            raise ValueError("Unsupported aggregation type")

        h = torch.cat([x, agg_feats], dim=1)  # Concatenate self-features & aggregated features
        return F.relu(self.linear(h)) 

# Define GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, agg_type="mean"):
        super().__init__()
        self.conv1 = GraphSAGEConv(in_channels, hidden_channels, agg_type=agg_type)
        self.conv2 = GraphSAGEConv(hidden_channels, out_channels, agg_type=agg_type)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)