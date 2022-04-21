# torch_geometric.nn的MessagePassing类简介

消息传递的过程可以写成：  
$
x_i^{k+1} = \gamma_{\Theta}\bigl(x_i, aggr_{j\in N(i)} \phi_{\Theta}(x_i^k, x_j^k, e_{ij})\bigr)
$
其中  
$\phi_{\Theta}$是聚合函数，聚合节点i和邻居节点j的特征  
$\gamma_{\Theta}$是节点更新函数  
$aggr$表示对聚合后节点特征的操作, 可以是add, mean, max等

## 子类继承MessagePassing
一般而言只需要复写:  
message(x_j->Tensor, edge_weight=None->OptTensor, **kw):  对应$\phi_{\Theta}$  

###待补充