import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops, softmax
from collections import defaultdict

class GATConv(MessagePassing):
    '''
    x' = \sum a_ij*x_j
    '''
    def __init__(self, in_channels, out_channels, add_loop=True, heads = 1):
        super().__init__(aggr='add', flow='source_to_target', node_dim=0)
        self.W = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(heads)])
        self.a = nn.ModuleList([nn.Linear(2*out_channels, 1, bias=False) for _ in range(heads)])
        if(heads!=1):
            self.fc = nn.Linear(heads*out_channels, out_channels, bias=False)
        self.edge_index = None
        self.add_loop = add_loop
        self.heads = heads

    def forward(self, x, edge_index):
        if self.edge_index==None:
            self.edge_index = add_self_loops(edge_index)[0] if self.add_loop==True else edge_index

        row ,col = self.edge_index
        # 计算attention系数a_ij
        multi_heads = []
        for i in range(self.heads):
            x_ = self.W[i](x)

            e = F.leaky_relu(self.a[i](torch.cat([x_[row], x_[col]], dim=-1))).view(-1)
            '''
            e_all = torch.zeros(x.shape[0], dtype=float).to(x.device)
            for idx in range(len(row)):
                e_all[row[idx]] = e_all[row[idx]] + e[idx]

            # 直接用e表示alpha
            alpha = e.clone()
            for idx in range(len(e)):
                alpha[idx] = e[idx]/e_all[row[idx]]
            '''
            # 注意utils中softmax的使用, 可以实现节点级的softmax, index参数是target节点数组
            alpha = softmax(e, index=row)
            multi_heads.append(self.propagate(self.edge_index, x=x_, norm=alpha))

        return torch.cat(multi_heads, dim=-1)

    def message(self, x_j, norm):
        return norm.view(-1,1)*x_j
        
