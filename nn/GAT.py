import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_cluster
import torch_geometric
import torch_geometric.nn as gnn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops, softmax
from collections import defaultdict

class GATConv(MessagePassing):
    '''
    x' = \sum a_ij*x_j
    '''
    def __init__(self, in_channels, out_channels, add_loop=True, n_head = 1):
        super().__init__(aggr='add', flow='source_to_target', node_dim=0)
        self.W = nn.Linear(in_channels, out_channels)
        self.a = nn.Linear(2*out_channels, 1, bias=False) 
        if(n_head!=1):
            self.fc = nn.Linear(n_head*out_channels, out_channels, bias=False)
        self.edge_index = None
        self.add_loop = add_loop
        self.n_head = n_head

    def forward(self, x, edge_index):
        if self.edge_index==None:
            self.edge_index = add_self_loops(edge_index)[0] if self.add_loop==True else edge_index

        row ,col = self.edge_index
        # 计算attention系数a_ij
        x = self.W(x)

        e = F.leaky_relu(self.a(torch.cat([x[row], x[col]], dim=-1))).view(-1)
        '''
        e_all = torch.zeros(x.shape[0], dtype=float).to(x.device)
        for idx in range(len(row)):
            e_all[row[idx]] = e_all[row[idx]] + e[idx]

        # 直接用e表示alpha
        alpha = e.clone()
        for idx in range(len(e)):
            alpha[idx] = e[idx]/e_all[row[idx]]
        '''
        # 注意utils中softmax的使用
        alpha = softmax(e, index=row)

        return self.propagate(self.edge_index, x=x, norm=e)

    def message(self, x_j, norm):
        return norm.view(-1,1)*x_j
        
