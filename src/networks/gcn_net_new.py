import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pygmtools import sinkhorn
import functools


class NGMConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, sk_channel=0):
        super(NGMConvLayer, self).__init__()
        self.sk_channel = sk_channel
        self.out_channels = out_channels
        self.classifier = (
            nn.Linear(self.out_channels, sk_channel) if sk_channel > 0 else None
        )

        self.conv = GCNConv(in_channels, self.out_channels)
        self.self_loop = nn.Linear(in_channels, self.out_channels)

    def forward(self, x, edge_index, n1, n2, sk_func=None):
        x_self = self.self_loop(x)
        x_neigh = self.conv(x, edge_index)
        x_out = x_self + x_neigh

        if self.classifier is not None:
            sk_features = self.classifier(x_out)
            sk_matrix = sk_features.view(n1, n2)
            sk_output = sk_func(sk_matrix, n1, n2, dummy_row=True)
            # x_out = torch.cat([x_out, sk_output.reshape(-1, self.sk_channel)], dim=-1)
            x_out = x_out + sk_output.reshape(-1, self.sk_channel)

        return x_out


class GCN_Net(nn.Module):
    def __init__(self, channels, sk_channels):
        super(GCN_Net, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(len(channels)):
            in_dim = 1 if i == 0 else channels[i - 1]
            self.layers.append(
                NGMConvLayer(
                    in_channels=in_dim,
                    out_channels=channels[i],
                    sk_channel=sk_channels,
                )
            )

        self.classifier = nn.Linear(channels[-1], 1)

    def forward(self, K, n1, n2):
        sinkhorn_fn = functools.partial(
            sinkhorn, dummy_row=False, max_iter=20, tau=0.05, batched_operation=False
        )

        edge_index = (K != 0).nonzero(as_tuple=False).t()
        x = torch.diag(K).view(-1, 1)

        for layer in self.layers:
            x = layer(x, edge_index, n1, n2, sinkhorn_fn)

        scores = self.classifier(x)
        s = scores.view(n2, -1).t()

        return sinkhorn_fn(s, n1, n2, dummy_row=True)
