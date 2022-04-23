'''
The gated graph convolution operator from the “Gated Graph Sequence Neural Networks” paper
https://arxiv.org/abs/1511.05493

h(0) = [x||0]  : in_channels -> out_channels
m(l+1)_i = sum(e_ij·h(l)_j·W)
h(l+1)_i = GRU(m(l+1)_i, h(l)_i)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops

class GatedConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_layer=2, aggr='mean'):
        '''
        这里对原操作做点改进:
        h(0) = x·W : in_channels -> out_channels
        m(l+1)_i = sum(e_ij·h(l)_j)
        h(l+1)_i = GRU(m(l+1)_i, h(l)_i)
        '''
        super().__init__(aggr=aggr, flow='source_to_target', node_dim=0)
        self.num_layer = num_layer
        self.gru = nn.GRUCell(out_channels, out_channels)
        self.W = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.W(x)
        for i in range(self.num_layer):
            m = self.propagate(edge_index, x=h, norm=edge_weight)
            h = self.gru(m, h)
        
        return h

    def message(self, x_j, norm):
        return x_j if norm==None else norm.view(-1,1)*x_j
