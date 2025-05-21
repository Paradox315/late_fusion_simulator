import torch
import torch.nn as nn
import pygmtools as pygm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

import functools

from utils.tools import build_conn_edge, build_graph


class GATConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        max_size=32,
        sk_channel=0,
        # sk_func=None,
    ):
        super(GATConvLayer, self).__init__()
        self.max_size = max_size
        self.sk_channel = sk_channel
        self.out_channels = out_channels

        # 分类器和相关函数
        self.classifier = (
            nn.Linear(self.out_channels, sk_channel) if sk_channel > 0 else None
        )
        # self.sk_func = sk_func

        # 图卷积层和自环层
        self.conv = GATv2Conv(in_channels, self.out_channels)
        self.self_loop = nn.Sequential(
            nn.Linear(in_channels, self.out_channels), nn.ReLU()
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * self.out_channels, self.out_channels), nn.Sigmoid()
        )

        # 添加层归一化
        self.norm = nn.LayerNorm(self.out_channels)

    def forward(self, x, x_mask, edge_index, edge_mask, vaild_mask) -> torch.Tensor:
        """
        Forward pass of the model.
        :param x: shape(max_size^2, in_channels) - node features
        :param x_mask: shape(max_size) - True for valid entries
        :param edge_index: shape(2, max_size^2) - edge index
        :param edge_mask: shape(max_size^2) - True for valid entries
        :param vaild_mask: shape(max_size, max_size) - True for valid entries
        """
        # 每行有1则该行有效，求和得到有效行数
        n1 = torch.sum(torch.any(vaild_mask, dim=-1))
        # 每列有1则该列有效，求和得到有效列数
        n2 = torch.sum(torch.any(vaild_mask, dim=1))
        x_valid = x[x_mask]
        edge_index_valid = edge_index[:, edge_mask].int()
        x_self = self.self_loop(x_valid)
        x_neigh = self.conv(x_valid, edge_index_valid)
        # x_out = x_self + x_neigh
        w = self.gate(torch.cat([x_self, x_neigh], dim=-1))
        x_out = x_self * w + x_neigh * (1 - w)

        if self.classifier is not None:
            sk_features = self.classifier(x_out)
            sk_matrix = sk_features.view(n1, n2)
            sk_output = torch.softmax(sk_matrix, dim=-1)
            x_out = x_out + sk_output.reshape(-1, self.sk_channel)
        x_out_padded = torch.zeros(
            (self.max_size**2, self.out_channels), device=x_out.device
        )
        x_out_padded[: n1 * n2] = x_out
        return x_out_padded


class GAT_Net(nn.Module):
    def __init__(self, channels, sk_channels):
        super(GAT_Net, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(len(channels)):
            in_dim = 1 if i == 0 else channels[i - 1]
            self.layers.append(
                GATConvLayer(
                    in_channels=in_dim,
                    out_channels=channels[i],
                    sk_channel=sk_channels,
                )
            )

        self.classifier = nn.Linear(channels[-1], 1)
        self.node_mlp = NodeMLP()

    def forward(
        self,
        ego_preds: torch.Tensor,
        ego_mask: torch.Tensor,
        cav_preds: torch.Tensor,
        cav_mask: torch.Tensor,
    ):
        """
        Forward pass of the model.
        :param ego_preds: shape(batch_size, max_size, 8) - features: (x,y,h,w,alpha,cls,score)
        :param ego_mask: shape(batch_size, max_size,) - True for valid entries
        :param cav_preds: shape(batch_size, max_size, 8) - features: (x,y,h,w,alpha,cls,score)
        :param cav_mask: shape(batch_size, max_size,) - True for valid entries
        :return: output: shape(batch_size, max_size, max_size) - affinity matrix
        """
        # Calculate affinities
        node_aff_mat = self.node_mlp(ego_preds, ego_mask, cav_preds, cav_mask) # shape(batch_size, max_size, max_size)
        # Get edge indices
        x = node_aff_mat.view(-1,1)
        
        edge_index, edge_attr, ne =pygm.utils.dense_to_sparse(node_aff_mat)
        for layer in self.layers:
        # Get final scores


class NodeMLP(nn.Module):
    """
    Computes affinity scores between two sets of predictions based on features,
    class matching, and confidence scores.
    """

    def __init__(self, feat_dim: int = 5, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.feat_dim = feat_dim  # (x,y,h,w,alpha)

        # Feature extraction network
        self.branch = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.distance = nn.CosineSimilarity(dim=-1)

    def forward(
        self,
        preds1: torch.Tensor,
        mask1: torch.Tensor,
        preds2: torch.Tensor,
        mask2: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param preds1: shape (max_size, feat_dim + 2)
        :param mask1: shape (max_size,)
        :param preds2: shape (n2, feat_dim + 2)
        :param mask2: shape (n2,)
        :return:
        """
        batch_size = preds1.shape[0]

        # Extract components
        x1 = preds1[:, :, : self.feat_dim]  # [B, N1, feat_dim]
        x2 = preds2[:, :, : self.feat_dim]  # [B, N2, feat_dim]
        cls1 = preds1[:, :, self.feat_dim]  # [B, N1]
        cls2 = preds2[:, :, self.feat_dim]  # [B, N2]
        conf1 = preds1[:, :, self.feat_dim + 1 :]  # [B, N1, C]
        conf2 = preds2[:, :, self.feat_dim + 1 :]  # [B, N2, C]

        # Process all batches at once
        # Feature similarity
        feat1 = self.branch(x1.reshape(-1, self.feat_dim))
        feat1 = feat1.reshape(batch_size, -1, feat1.shape[-1])  # [B, N1, out_dim]
        feat2 = self.branch(x2.reshape(-1, self.feat_dim))
        feat2 = feat2.reshape(batch_size, -1, feat2.shape[-1])  # [B, N2, out_dim]

        # Expand dimensions for broadcasting
        feat1 = feat1.unsqueeze(2)  # [B, N1, 1, out_dim]
        feat2 = feat2.unsqueeze(1)  # [B, 1, N2, out_dim]
        feature_sim = self.distance(feat1, feat2)  # [B, N1, N2]

        # Class matching with batch dimension
        cls1_expanded = cls1.unsqueeze(2)  # [B, N1, 1]
        cls2_expanded = cls2.unsqueeze(1)  # [B, 1, N2]
        class_sim = (cls1_expanded == cls2_expanded).float()  # [B, N1, N2]

        # Confidence similarity with batch dimension
        conf1_expanded = conf1.unsqueeze(2)  # [B, N1, 1, C]
        conf2_expanded = conf2.unsqueeze(1)  # [B, 1, N2, C]
        conf_sim = torch.sum(
            torch.sqrt(conf1_expanded * conf2_expanded), dim=-1
        )  # [B, N1, N2]

        # Combine similarities
        affinity = feature_sim * class_sim * conf_sim  # [B, N1, N2]

        # Apply masks
        mask_matrix = mask1.unsqueeze(2) & mask2.unsqueeze(1)  # [B, N1, N2]
        affinity = affinity * mask_matrix.float()

        return affinity
