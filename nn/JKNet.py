'''
Representation Learning on Graphs with Jumping Knowledge Networks
ICML 2018
此处不设原文的LSTM-ATTENTION, 采用单独的attention去设计
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops, softmax
from GCN import GCNConv


class JKConv(nn.Module):
    '''
    默认采用GCN的结构
    k: 网络深度
    type: 三种池化方式： 'max', 'concat', 'att'
    '''
    def __init__(self, in_channels, out_channels, K=8, type='max'):
        super(JKConv, self).__init__()
        assert type in ['max', 'concat', 'att'], "\'type\' must be in [\'max\', \'concat\', \'att\']"
        if type=='att':
            self.W = nn.Linear(out_channels, 1)
        self.GCNlayer = nn.ModuleList(
            [GCNConv(in_channels, out_channels)] + [GCNConv(out_channels, out_channels) for _ in range(K-1)]
        )
        self.k = K
        self.type = type
        
    def forward(self, x, edge_index):
        output = []
        for i in range(self.k-1):
            x = self.GCNlayer[i](x, edge_index)
            x = F.elu(x)
            output.append(x)
        output.append(self.GCNlayer[self.k-1](x, edge_index))

        if self.type == 'max':
            output = torch.stack(output, dim=-1) # [N, d, k]
            return nn.MaxPool1d(kernel_size=self.k)(output).squeeze(2) #[N, d]

        elif self.type == 'concat':
            return torch.cat(output, dim=-1)

        elif self.type == 'att':
            output = torch.stack(output, dim=0) # [k, N, d]
            alpha = self.W(output) #[k, N, 1]
            alpha = torch.softmax(alpha, dim=0) #[k, N, 1]
            return torch.sum(alpha*output, dim=0) #[N, d]
