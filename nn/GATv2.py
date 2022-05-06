import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops, softmax
from collections import defaultdict

class GATv2Conv(MessagePassing):
    def __init__(self, in_channels, out_channels, add_loop=True, heads=1):
        super().__init__(aggr='add', flow='source_to_target', node_dim=0)
        self.W = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(heads)])
        self.a = nn.ModuleList([nn.Linear(2*out_channels, 1) for _ in range(heads)])
        self.edge_index = None
        self.add_loop = add_loop
        self.heads = heads

    def forward(self, x, edge_index):
        if self.edge_index==None:
            self.edge_index = add_self_loops(edge_index)[0] if self.add_loop==True else edge_index

        row, col = self.edge_index

        output = []
        for i in range(self.heads):
            x_ = self.W[i](x)
            e = F.leaky_relu(torch.cat([x_[row], x_[col]], dim=-1))
            e = self.a[i](e).view(-1)
            alpha = softmax(e, index=row)
            output.append(self.propagate(self.edge_index, x=x_, norm=alpha))

        return torch.cat(output, dim=-1)

    def message(self, x_j, norm):
        return norm.view(-1,1)*x_j