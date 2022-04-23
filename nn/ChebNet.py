import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops, get_laplacian, remove_self_loops

class ChebConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K=1):
        assert K>=0
        super().__init__(aggr='add', flow='source_to_target', node_dim=0)
        self.fc = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(K+1)])
        self.k = K
        self.coef = nn.ModuleList([])

    def forward(self, x, edge_index):
            # L~ = 2L/lambda_max - I -> L-I = -D^(-0.5)AD^(-0.5)
        row, col = edge_index
        deg = degree(row, num_nodes=x.shape[0]).pow(-0.5)
        edge_weight = -deg[row]*deg[col]
        # print(edge_weight.shape)
        # print(x.shape)
        # 先添加0阶和1阶
        T_0 = x
        T_1 = self.propagate(edge_index=edge_index, x=x, norm=edge_weight) if self.k>0 else 0
        ans = (self.fc[0](T_0) + self.fc[1](T_1)) if self.k>0 else self.fc[0](T_0)
        for i in range(2, self.k+1):
            tmp = 2*self.propagate(edge_index=edge_index, x=T_1, norm=edge_weight) - T_0
            ans += self.fc[i](tmp)
            T_0, T_1 = T_1, tmp
            
        return ans

    def message(self, x_j, norm):
        return norm.view(-1,1)*x_j
