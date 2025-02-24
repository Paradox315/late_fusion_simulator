import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv
from src.utils.lap import build_affinity_matrix_v2
from src.utils.tools import build_conn_edge, build_graph


def sinkhorn_log(scores, eps=0.1, n_iter=3):
    """
    Sinkhorn algorithm in log-space
    :param scores: shape(n, m) - input matrix
    :param eps: (float) - regularization parameter
    :param n_iter: (int) - number of iterations
    :return: normalized matrix
    """
    n, m = scores.shape

    # Initialize log domain matrix L = log Q = -scores / eps
    L = -scores / eps

    # Set log scaling factors (r are ones, c are scaled to match the desired sum along columns)
    log_r = torch.zeros(n, device=L.device)  # since r=1 => log(1)=0
    log_c = torch.log(torch.ones(m, device=L.device) * (n / m))

    for _ in range(n_iter):
        # Normalize columns: compute log sum over rows
        logsum_cols = torch.logsumexp(L, dim=0)  # shape: (m,)
        log_u = log_c - logsum_cols  # adjustment for columns
        # Broadcast addition over columns
        L = L + log_u.unsqueeze(0)

        # Normalize rows: compute log sum over columns
        logsum_rows = torch.logsumexp(L, dim=1)  # shape: (n,)
        log_v = log_r - logsum_rows  # adjustment for rows
        # Broadcast addition over rows
        L = L + log_v.unsqueeze(1)

    # Convert back from log-domain
    Q = torch.exp(L)
    return Q


def pad_predictions(preds, max_size=32):
    """Pad predictions to fixed size
    :param preds: shape(n, 8) - features: (x,y,h,w,alpha,cls,score)
    :param max_size: int - maximum number of objects
    :return: padded_preds: shape(max_size, 8) - padded features
    :return: mask: shape(max_size,) - True for valid entries
    """
    n = preds.shape[0]
    assert n <= max_size, f"Number of objects ({n}) exceeds maximum size ({max_size})"

    device = preds.device
    padded_preds = torch.zeros((max_size, 8), device=device)
    mask = torch.zeros(max_size, dtype=torch.bool, device=device)

    # Copy valid predictions
    padded_preds[:n] = preds
    mask[:n] = True

    return padded_preds, mask


class NGMConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        max_size=32,
        sk_channel=0,
        # sk_func=None,
    ):
        super(NGMConvLayer, self).__init__()
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
        self.self_loop = nn.Linear(in_channels, self.out_channels)

        # 添加层归一化
        self.norm = nn.LayerNorm(self.out_channels)

    def forward(self, x, x_mask, edge_index, edge_mask, vaild_mask) -> torch.Tensor:
        """
        Forward pass of the model.
        :param x: shape(max_size**2, in_channels) - node features
        :param x_mask: shape(max_size,) - True for valid entries
        :param edge_index: shape(2, max_size**2) - edge index
        :param edge_mask: shape(max_size**2,) - True for valid entries
        :param vaild_mask: shape(max_size, max_size) - True for valid entries
        """
        # 每行有1则该行有效，求和得到有效行数
        n1 = torch.sum(torch.any(vaild_mask, dim=1))
        # 每列有1则该列有效，求和得到有效列数
        n2 = torch.sum(torch.any(vaild_mask, dim=0))
        x_valid = x[x_mask]
        edge_index_valid = edge_index[:, edge_mask].int()
        x_self = self.self_loop(x_valid)
        x_neigh = self.conv(x_valid, edge_index_valid)
        x_out = x_self + x_neigh

        if self.classifier is not None:
            sk_features = self.classifier(x_out)
            sk_matrix = sk_features.view(n1, n2)
            # sk_output = self.sk_func(
            #     sk_matrix, torch.tensor([n1]), torch.tensor([n2]), dummy_row=True
            # )
            sk_output = torch.softmax(sk_matrix, dim=-1)
            x_out = x_out + sk_output.reshape(-1, self.sk_channel)
        x_out_padded = torch.zeros(
            (self.max_size**2, self.out_channels), device=x_out.device
        )
        x_out_padded[: n1 * n2] = x_out
        return x_out_padded


class GCN_Net(nn.Module):
    def __init__(self, channels, sk_channels=1, max_size=32):
        super(GCN_Net, self).__init__()
        self.max_size = max_size
        self.node_mlp = NodeMLP()
        self.edge_mlp = EdgeMLP()
        self.layers = nn.ModuleList()
        # self.sinkhorn_fn = functools.partial(
        #     sinkhorn, dummy_row=False, max_iter=20, tau=0.05, batched_operation=False
        # )

        for i in range(len(channels)):
            in_dim = 1 if i == 0 else channels[i - 1]
            self.layers.append(
                NGMConvLayer(
                    in_channels=in_dim,
                    out_channels=channels[i],
                    sk_channel=sk_channels,
                    # sk_func=self.sinkhorn_fn,
                )
            )

        self.classifier = nn.Linear(channels[-1], 1)

    def forward(
        self,
        ego_preds: torch.Tensor,
        ego_mask: torch.Tensor,
        cav_preds: torch.Tensor,
        cav_mask: torch.Tensor,
    ):
        """
        Forward pass of the model.
        :param ego_preds: shape(max_size, 8) - features: (x,y,h,w,alpha,cls,score)
        :param ego_mask: shape(max_size,) - True for valid entries
        :param cav_preds: shape(max_size, 8) - features: (x,y,h,w,alpha,cls,score)
        :param cav_mask: shape(max_size,) - True for valid entries
        :return: output: shape(max_size, max_size) - affinity matrix
        """
        ego_preds_valid = ego_preds[ego_mask]
        cav_preds_valid = cav_preds[cav_mask]
        # Build graphs with padding
        ego_graph = build_graph(ego_preds_valid)
        cav_graph = build_graph(cav_preds_valid)

        # Get original dimensions
        n1, n2 = torch.sum(ego_mask), torch.sum(cav_mask)

        # Build connections and edges
        conn1, edge1 = build_conn_edge(ego_graph)
        conn2, edge2 = build_conn_edge(cav_graph)

        # Calculate affinities
        node_aff_mat = self.node_mlp(ego_preds_valid, cav_preds_valid)
        edge_aff_mat = self.edge_mlp(edge1, edge2)

        # Build affinity matrix
        K_padded = torch.zeros(
            (self.max_size**2, self.max_size**2), device=ego_preds.device
        )
        K_valid = build_affinity_matrix_v2(node_aff_mat, edge_aff_mat, conn1, conn2)
        K_padded[: n1 * n2, : n1 * n2] = K_valid

        # Build node vectors and node masks
        x_padded = torch.zeros((self.max_size**2, 1), device=ego_preds.device)
        x_padded[: n1 * n2] = node_aff_mat.t().reshape(-1, 1)
        # x_padded = torch.diagonal(K_padded).unsqueeze(-1)
        x_mask = torch.zeros(
            (self.max_size**2), dtype=torch.bool, device=ego_preds.device
        )
        x_mask[: n1 * n2] = True

        # Build edge vectors and edge masks
        edge_index = (K_padded != 0).nonzero(as_tuple=False).t()
        edge_padded = torch.zeros(
            (2, (2 * self.max_size) ** 2), device=ego_preds.device
        )
        edge_padded[:, : edge_index.shape[1]] = edge_index
        edge_mask = torch.zeros(
            ((2 * self.max_size) ** 2), dtype=torch.bool, device=ego_preds.device
        )
        edge_mask[: edge_index.shape[1]] = True
        # 创建有效区域掩码
        valid_mask_2d = torch.zeros(
            (self.max_size, self.max_size), dtype=torch.bool, device=ego_preds.device
        )
        valid_mask_2d[:n1, :n2] = True
        for layer in self.layers:
            x_padded = layer(x_padded, x_mask, edge_padded, edge_mask, valid_mask_2d)

        # Get final scores
        scores = self.classifier(x_padded[x_mask])
        s = scores.view(n2, n1).t()

        # Apply masks to get original size output
        # output = self.sinkhorn_fn(
        #     s,
        #     torch.tensor([n1]),
        #     torch.tensor([n2]),
        #     dummy_row=True,
        # )
        # output = sinkhorn_log(s)
        output = torch.softmax(s, dim=-1)

        # Pad output to fixed size
        padded_output = torch.zeros(
            (self.max_size, self.max_size), device=output.device
        )
        padded_output[:n1, :n2] = output

        return padded_output


# Node affinity function
class NodeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 共享权重分支
        self.branch = nn.Sequential(
            nn.Linear(5, 64),  # 输入(x,y,h,w,\alpha)
            nn.GELU(),
            nn.Linear(64, 32),
        )
        # 相似度计算
        self.distance = nn.CosineSimilarity(dim=-1)

    def forward(self, preds1, preds2):
        x1, x2 = preds1[:, :5], preds2[:, :5]
        cls1, cls2 = preds1[:, 5], preds2[:, 5]
        conf1, conf2 = preds1[:, 6:], preds2[:, 6:]
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
            nn.GELU(),
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
