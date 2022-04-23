import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops

class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='mean'):
        super().__init__(aggr=aggr, flow='source_to_target', node_dim=0)
        self.fc = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(2)])

    def forward(self, x, edge_index, edge_weight=None):
        feature_i = self.fc[0](x)
        feature_j = self.propagate(edge_index=edge_index, x=self.fc[1](x), norm=edge_weight)
        return feature_i + feature_j

    def message(self, x_j, norm):
        return x_j if norm==None else norm.view(-1,1)*x_j
        