import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as gnn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops

class GCNConv(MessagePassing):
    '''
    X' = D^(0.5)AD^(-0.5)XW

    node-wise:
        x' = \sum_j{[e_ij/sqrt(d_i*d_j)]*(x_j·W)}
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add', flow='source_to_target', node_dim=0)
        self.fc = nn.Linear(in_channels, out_channels)
        self.edge_index = None
        self.A = None

    def forward(self, x, edge_index):
        if(self.A==None):
            self.edge_index = add_self_loops(edge_index, num_nodes=x.shape[0])[0]
            deg_inv = degree(self.edge_index[0]).pow(-0.5)
            # eij -> eij/sqrt(di*dj)
            row, col = self.edge_index
            self.A = deg_inv[row]*deg_inv[col]

        x = self.fc(x)
        return self.propagate(self.edge_index, x=x, norm=self.A)


    def message(self, x_j, norm):
        return norm.view(-1,1)*x_j