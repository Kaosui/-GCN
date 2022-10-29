'''
Predict then Propagate: Graph Neural Networks meet Personalized PageRank
ICLR, 2019a.
这里增加原文没有的一个设定: 各层的权重矩阵可以不同
H0 = f(X)
H' -> (1-a)*PH + a*H0
P是归一化, 添加自环的拉普拉斯矩阵 P = D^(-0.5)·A·D^(-0.5)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops

class APPNPConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K=10, alpha=0.1, same_weight=True):
        super().__init__(aggr='add', flow='source_to_target', node_dim=0)
        self.W = nn.Linear(in_channels, out_channels)
        if same_weight==False:
            self.Ws = nn.ModuleList([self.W] + [nn.Linear(out_channels, out_channels) for _ in range(K-1)])
        else:
            self.Ws = nn.Linear(out_channels, out_channels)
        self.same_weight = same_weight
        self.k = K
        self.alpha = alpha
        self.P = None
        self.edge_index = None

    def forward(self, x, edge_index):
        if self.P==None:
            self.edge_index = add_self_loops(edge_index, num_nodes=x.shape[0])[0]
            row, col = self.edge_index
            deg_inv = degree(row).pow(-0.5)
            self.P = deg_inv[row]*deg_inv[col]
        H0 = self.W(x)
        H = H0
        for i in range(self.k):
            if i>=1:
                if self.same_weight==False:
                    H = self.Ws[i](out)
                else:
                    H = self.Ws(out)
            out = self.propagate(edge_index=self.edge_index, x=H, norm=self.P)
            out = (1-self.alpha)*out + self.alpha*H0
        return out

    def message(self, x_j, norm):
        return norm.view(-1,1)*x_j