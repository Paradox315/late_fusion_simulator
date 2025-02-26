import functools

import torch
import torch.nn as nn
from pygmtools import sinkhorn
import torch.nn.functional as F


# 定义适配 torch_geometric 的图卷积层
class NGMConvLayer(nn.Module):
    def __init__(
        self,
        in_node_features,
        in_edge_features,
        out_node_features,
        out_edge_features,
        sk_channel=0,
    ):
        super(NGMConvLayer, self).__init__()
        self.in_nfeat = in_node_features
        self.in_efeat = in_edge_features
        self.out_efeat = out_edge_features
        self.sk_channel = sk_channel

        # 输出节点特征与边特征的维度检查
        assert out_node_features == out_edge_features + self.sk_channel

        # 跳跃连接的处理
        if self.sk_channel > 0:
            self.out_nfeat = out_node_features - self.sk_channel
            self.classifier = nn.Linear(self.out_nfeat, self.sk_channel)
        else:
            self.out_nfeat = out_node_features
            self.classifier = None

        # 定义更新节点特征的 MLP
        self.n_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat),
            nn.ReLU(),
        )

        # 更新自身节点特征的 MLP
        self.n_self_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat),
            nn.ReLU(),
        )

    def forward(self, A, W, x, n1=None, n2=None, sk_func=None):
        """
        :param A: 邻接矩阵 (num_nodes, num_nodes)
        :param W: 边特征矩阵 (num_nodes, num_nodes, in_edge_features)
        :param x: 节点特征矩阵 (num_nodes, in_node_features)
        :param n1: 图 1 的节点数
        :param n2: 图 2 的节点数
        :param sk_func: Sinkhorn 函数
        """
        # 传递消息
        x1 = self.n_func(x)  # 更新邻居特征
        # A = F.normalize(A, p=1, dim=1)
        A = A / (A.sum(dim=1, keepdim=True) + 1e-8)
        x2 = (
            torch.matmul(
                (A.unsqueeze(-1) * W).permute(2, 0, 1), x1.unsqueeze(1).permute(2, 0, 1)
            )
            .squeeze(-1)
            .t()
        )
        x2 += self.n_self_func(x)  # 加上自身特征

        # 跳跃连接
        if self.classifier is not None:
            x3 = self.classifier(x2).view(n1, n2)
            x4 = (
                sk_func(x3, n1, n2, dummy_row=True)
                .contiguous()
                .view(-1, self.sk_channel)
            )
            x_new = torch.cat((x2, x4), dim=-1)
        else:
            x_new = x2

        return W, x_new


# 定义适配 torch_geometric 的 NGM_Net
class GCN_Net(torch.nn.Module):
    """
    Pytorch implementation of NGM network
    """

    def __init__(self, gnn_channels, sk_emb):
        super(GCN_Net, self).__init__()
        self.gnn_layer = len(gnn_channels)

        # 初始化图卷积层
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = NGMConvLayer(
                    1, 1, gnn_channels[i] + sk_emb, gnn_channels[i], sk_channel=sk_emb
                )
            else:
                gnn_layer = NGMConvLayer(
                    gnn_channels[i - 1] + sk_emb,
                    gnn_channels[i - 1],
                    gnn_channels[i] + sk_emb,
                    gnn_channels[i],
                    sk_channel=sk_emb,
                )
            self.add_module(f"gnn_layer_{i}", gnn_layer)

        # 分类器层
        self.classifier = nn.Linear(gnn_channels[-1] + sk_emb, 1)

    def forward(self, K, n1, n2):
        """
        :param K: shape (n1*n2, n1*n2)  # 邻接矩阵
        :param n1: # 图 1 的节点数
        :param n2: # 图 2 的节点数
        :return:
        """
        _sinkhorn_func = functools.partial(
            sinkhorn,
            dummy_row=False,
            max_iter=20,
            tau=0.05,
            batched_operation=False,
        )

        A = (K != 0).float()
        emb_K = K.unsqueeze(-1)
        emb = torch.diag(K).reshape(n1 * n2, 1)
        # emb = torch.ones((n1 * n2, 1))
        # NGM qap solver with GNN layers
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, f"gnn_layer_{i}")
            emb_K, emb = gnn_layer(
                A,
                emb_K,
                emb,
                n1=n1,
                n2=n2,
                sk_func=_sinkhorn_func,
            )

        # 分类器输出
        v = self.classifier(emb)
        s = v.view(n2, -1).t()
        # 应用 Sinkhorn 函数
        return _sinkhorn_func(s, n1, n2, dummy_row=True)
