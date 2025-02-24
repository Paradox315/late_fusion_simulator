import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pygmtools import sinkhorn
import functools
from src.utils.lap import build_affinity_matrix_v2
from src.utils.tools import build_graph, build_conn_edge


class NGMConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, sk_channel=0, sk_func=None):
        super(NGMConvLayer, self).__init__()
        self.sk_channel = sk_channel
        self.out_channels = out_channels
        self.classifier = (
            nn.Linear(self.out_channels, sk_channel) if sk_channel > 0 else None
        )
        self.sk_func = sk_func

        self.conv = GCNConv(in_channels, self.out_channels)
        self.self_loop = nn.Linear(in_channels, self.out_channels)

    def forward(self, x, edge_index, n1, n2):
        x_self = self.self_loop(x)
        x_neigh = self.conv(x, edge_index)
        x_out = x_self + x_neigh

        if self.classifier is not None:
            sk_features = self.classifier(x_out)
            sk_matrix = sk_features.view(n1, n2)
            sk_output = self.sk_func(sk_matrix, n1, n2, dummy_row=True)
            # x_out = torch.cat([x_out, sk_output.reshape(-1, self.sk_channel)], dim=-1)
            x_out = x_out + sk_output.reshape(-1, self.sk_channel)

        return x_out


class GCN_Net(nn.Module):
    def __init__(self, channels, sk_channels=1):
        super(GCN_Net, self).__init__()
        assert len(channels) > 0
        self.node_mlp = NodeMLP()
        self.edge_mlp = EdgeMLP()
        self.layers = nn.ModuleList()
        self.sinkhorn_fn = functools.partial(
            sinkhorn, dummy_row=False, max_iter=20, tau=0.05, batched_operation=False
        )
        for i in range(len(channels)):
            in_dim = 1 if i == 0 else channels[i - 1]
            # print(f"Layer {i}: {in_dim} -> {channels[i]}")
            self.layers.append(
                NGMConvLayer(
                    in_channels=in_dim,
                    out_channels=channels[i],
                    sk_channel=sk_channels,
                    sk_func=self.sinkhorn_fn,
                )
            )

        self.classifier = nn.Linear(channels[-1], 1)

    def forward(self, ego_preds, cav_preds):
        ego_preds, cav_preds = ego_preds.squeeze(0), cav_preds.squeeze(0)
        ego_graph, cav_graph = build_graph(ego_preds), build_graph(cav_preds)
        n1, n2 = torch.tensor([ego_graph.shape[0]]), torch.tensor([cav_graph.shape[0]])
        conn1, edge1 = build_conn_edge(ego_graph)
        conn2, edge2 = build_conn_edge(cav_graph)
        node_aff_mat = self.node_mlp(ego_preds, cav_preds)
        edge_aff_mat = self.edge_mlp(edge1, edge2)
        K = build_affinity_matrix_v2(node_aff_mat, edge_aff_mat, conn1, conn2)
        edge_index = (K != 0).nonzero(as_tuple=False).t()
        x = torch.diag(K).view(-1, 1)

        for layer in self.layers:
            x = layer(x, edge_index, n1, n2)

        scores = self.classifier(x)
        s = scores.view(n2, -1).t()

        return self.sinkhorn_fn(s, n1, n2, dummy_row=True)


# Node affinity function
class NodeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 共享权重分支
        self.branch = nn.Sequential(
            nn.Linear(5, 64),  # 输入(x,y,h,w,\alpha)
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        # 相似度计算
        self.distance = nn.CosineSimilarity(dim=-1)

    def forward(self, preds1, preds2):
        x1, x2 = preds1[:, 1:6], preds2[:, 1:6]
        cls1, cls2 = preds1[:, 6], preds2[:, 6]
        conf1, conf2 = preds1[:, 7:], preds2[:, 7:]
        feat1 = self.branch(x1).unsqueeze(1)
        feat2 = self.branch(x2).unsqueeze(0)
        dist_mat = self.distance(feat1, feat2)  # (n1, n2)
        # check if the two nodes are of the same class
        cls_dist = cls1.view(-1, 1) == cls2.view(1, -1)  # (n1, n2)
        # calculate the confidence affinity
        conf_dist = torch.sum(
            torch.sqrt(conf1.unsqueeze(1) * conf2.unsqueeze(0)), dim=-1
        )  # (n1, n2)
        return cls_dist * conf_dist * dist_mat


# edge affinity function
class EdgeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 共享权重分支
        self.branch = nn.Sequential(
            nn.Linear(3, 64),  # 输入(\Delta dist, \Delta \theta, \Delta \alpha)
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        # 相似度计算
        self.distance = nn.CosineSimilarity(dim=-1)

    def forward(self, edges1, edges2):
        x1, x2 = edges1[:, :3], edges2[:, :3]
        feat1 = self.branch(x1).unsqueeze(1)
        feat2 = self.branch(x2).unsqueeze(0)
        dist_mat = self.distance(feat1, feat2)  # (n1, n2)
        # check if the two nodes are of the same class
        cls_edge1, cls_edge2 = edges1[:, 3:].int(), edges2[:, 3:].int()

        def compare_tensors(tensor1, tensor2):
            tensor1_exp = tensor1.unsqueeze(1).expand(-1, tensor2.size(0), -1)
            tensor2_exp = tensor2.unsqueeze(0).expand(tensor1.size(0), -1, -1)
            return torch.eq(tensor1_exp, tensor2_exp).all(dim=-1)

        cls_aff = compare_tensors(cls_edge1, cls_edge2)
        # calculate the confidence affinity

        return cls_aff * dist_mat
