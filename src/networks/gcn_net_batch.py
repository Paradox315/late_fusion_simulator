import torch
import torch.nn as nn


# 定义适配 torch_geometric 的图卷积层
class ConvLayer(nn.Module):
    def __init__(
        self,
        in_node_features,
        out_node_features,
        max_size=32,
    ):
        super(ConvLayer, self).__init__()
        self.in_nfeat = in_node_features
        self.out_nfeat = out_node_features
        self.max_size = max_size

        # 跳跃连接的处理
        self.classifier = nn.Linear(self.out_nfeat, 1)

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

    def forward(self, A, W, x, vaild_mask):
        """
        :param A: 邻接矩阵 (max_size**2, max_size**2)
        :param W: 边特征矩阵 (max_size**2, max_size**2)
        :param x: 节点特征矩阵 (max_size**2, in_node_features)
        :param vaild_mask: 有效节点掩码 (max_size,max_size) where 1 means valid
        """
        n1 = torch.sum(torch.any(vaild_mask, dim=1))
        n2 = torch.sum(torch.any(vaild_mask, dim=0))
        # 传递消息
        x1 = self.n_func(x)  # shape (max_size^2, out_node_features)
        A = A / (A.sum(dim=1, keepdim=True) + 1e-8)
        x2 = torch.matmul(A * W, x1)  # 计算邻居特征
        x2 += self.n_self_func(x)  # shape (max_size^2, out_node_features)
        # 跳跃连接
        # x3 = torch.zeros(
        #     (self.max_size**2, self.max_size**2),
        #     device=x.device,
        # )
        # x3[:n1, :n2] = self.classifier(x2[: n1 * n2]).view(
        #     n1, -1
        # )  # shape (max_size, max_size)
        # x3[:n1, :n2] = torch.softmax(x3[:n1, :n2], dim=1)
        # x4 = torch.zeros((self.max_size**2, 1), device=x.device)
        # x4[: n1 * n2] = x3[:n1, :n2].reshape(-1, 1)
        # x_new = x4 + x2  # shape (max_size^2, out_node_features)
        x_new = x2

        return x_new


class GCN_Net(torch.nn.Module):
    def __init__(self, gnn_channels, max_size=32):
        super(GCN_Net, self).__init__()
        self.gnn_layer = len(gnn_channels)
        self.max_size = max_size

        # 初始化图卷积层
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = ConvLayer(1, gnn_channels[i])
            else:
                gnn_layer = ConvLayer(
                    gnn_channels[i - 1],
                    gnn_channels[i],
                )
            self.add_module(f"gnn_layer_{i}", gnn_layer)

        # 分类器层
        self.classifier = nn.Linear(gnn_channels[-1], 1)

    def forward(self, K, vaild_mask):
        """
        :param K: shape (max_size^2,max_size^2)  # 邻接矩阵
        :param valid_mask: shape (max_size,max_size) where 1 means valid
        :return:
        """
        n1 = torch.sum(torch.any(vaild_mask, dim=1))
        n2 = torch.sum(torch.any(vaild_mask, dim=0))
        A = (K != 0).float()
        x = torch.diagonal(K).unsqueeze(-1)
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, f"gnn_layer_{i}")
            x = gnn_layer(A, K, x, vaild_mask)

        # 分类器输出
        scores = torch.zeros((self.max_size, self.max_size), device=x.device)
        scores[:n1, :n2] = self.classifier(x[: n1 * n2]).view(n2, n1).t()
        scores[:n1, :n2] = torch.softmax(scores[:n1, :n2], dim=-1)
        return scores
