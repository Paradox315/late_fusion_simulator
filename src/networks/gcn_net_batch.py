from typing import Dict, List, Optional, Tuple, Union

import pygmtools as pygm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import DeQuantStub, QuantStub


class ConvLayer(nn.Module):
    """优化的图卷积层，支持批处理和通过特征拼接整合信息"""

    def __init__(
        self,
        in_node_features: int,
        out_node_features: int,
        max_size: int = 32,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        update_type: str = "linear",
        kernel_size: int = 3,
    ):
        super(ConvLayer, self).__init__()
        self.in_nfeat = in_node_features
        self.out_nfeat = out_node_features
        self.max_size = max_size
        self.matmul = FloatFunctional()
        self.concat = FloatFunctional()

        # 创建基于更新类型的特征处理模块
        self.feature_processor = self._create_feature_processor(
            update_type, kernel_size
        )

        # 维度匹配层
        self.dim_match = (
            nn.Linear(in_node_features, out_node_features)
            if in_node_features != out_node_features
            else nn.Identity()
        )

        # 如果使用线性投影，初始化权重
        if isinstance(self.dim_match, nn.Linear):
            nn.init.orthogonal_(self.dim_match.weight)

        # 特征融合层
        self.feature_fusion = nn.Linear(out_node_features * 2, out_node_features)
        nn.init.xavier_uniform_(self.feature_fusion.weight)
        nn.init.zeros_(self.feature_fusion.bias)

        # 激活、Dropout 和 LayerNorm
        self.activation = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def _create_feature_processor(self, update_type, kernel_size):
        """创建基于更新类型的特征处理模块"""
        if update_type == "conv":
            return nn.Sequential(
                # 将特征重塑为卷积格式
                Reshape(-1, self.in_nfeat, self.max_size, self.max_size),
                # 应用卷积
                nn.Conv2d(
                    self.in_nfeat,
                    self.out_nfeat,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                # 重塑回原始格式
                Reshape(-1, self.max_size * self.max_size, self.out_nfeat),
            )
        else:  # "linear"
            return nn.Linear(self.in_nfeat, self.out_nfeat)

    def _get_activation(self, activation_name):
        """获取激活函数"""
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "leaky_relu":
            return nn.LeakyReLU(0.2)
        elif activation_name == "gelu":
            return nn.GELU()
        elif activation_name == "none":
            return nn.Identity()
        else:
            raise ValueError(f"不支持的激活函数: {activation_name}")

    def forward(self, W, x):
        """
        :param W: 邻接矩阵，shape (batch_size, max_size**2, max_size**2)
        :param x: 节点特征，shape (batch_size, max_size**2, in_node_features)
        :return: 节点特征, shape (batch_size, max_size**2, out_node_features)
        """
        # 保存原始输入用于残差连接
        identity = x

        # 处理特征 - 无条件分支
        x = self.feature_processor(x)

        # 应用边特征传播
        x_propagated = self.matmul.matmul(W, x)

        # 调整原始特征维度
        identity_proj = self.dim_match(identity)

        # 拼接并融合特征
        combined = self.concat.cat([x_propagated, identity_proj], dim=-1)
        x = self.feature_fusion(combined)

        # 应用后处理
        x = self.activation(x)
        x = self.dropout(x)

        return x


# 实用工具类：用于在卷积后重塑张量
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.reshape(self.shape)


class GCN_Net(torch.nn.Module):
    """可配置的图卷积网络，支持多种更新方式"""

    def __init__(
        self,
        gnn_channels: Union[List[int], Tuple[int, ...]],
        input_dim: int = 32,
        max_size: int = 32,
        dropout_rates: Optional[List[float]] = None,
        activations: Optional[List[str]] = None,
        quantize: bool = True,
        update_type: str = "linear",  # 新增参数："linear"或"conv"
        kernel_sizes: Optional[List[int]] = None,  # 卷积核大小列表
    ):
        super(GCN_Net, self).__init__()
        self.gnn_layer_count = len(gnn_channels)
        self.max_size = max_size
        self.quantize = quantize
        self.gnn_channels = gnn_channels
        self.conv_layers = nn.ModuleList()
        self.update_type = update_type

        # 量化桩
        if quantize:
            self.quantW = QuantStub()
            self.quantx = QuantStub()
            self.output_quant = DeQuantStub()

        # 默认参数设置
        if dropout_rates is None:
            dropout_rates = [0.0] * self.gnn_layer_count
        if activations is None:
            activations = ["relu"] * self.gnn_layer_count
        if kernel_sizes is None:
            kernel_sizes = [3] * self.gnn_layer_count  # 默认使用3x3卷积核

        # 确保参数列表长度一致
        assert len(dropout_rates) == self.gnn_layer_count, "dropout_rates长度必须与网络层数相同"
        assert len(activations) == self.gnn_layer_count, "activations长度必须与网络层数相同"
        if update_type == "conv":
            assert len(kernel_sizes) == self.gnn_layer_count, "kernel_sizes长度必须与网络层数相同"

        # 初始化图卷积层
        for i in range(self.gnn_layer_count):
            if i == 0:
                in_features = input_dim
            else:
                in_features = gnn_channels[i - 1]

            gnn_layer = ConvLayer(
                in_node_features=in_features,
                out_node_features=gnn_channels[i],
                max_size=max_size,
                activation=activations[i],
                dropout_rate=dropout_rates[i],
                update_type=update_type,
                kernel_size=kernel_sizes[i] if update_type == "conv" else 1,
            )
            self.conv_layers.append(gnn_layer)

    def forward(self, W, x):
        """
        :param W: 邻接矩阵，shape (batch_size, 1024, 1024)
        :param x: 节点特征，shape (batch_size, 1024, input_dim)
        :return: 节点特征, shape (batch_size, 1024, gnn_channels[-1])
        """

        # 输入量化
        if self.quantize:
            W = self.quantW(W)
            x = self.quantx(x)

        # 消息传递
        for i in range(self.gnn_layer_count):
            gnn_layer = self.conv_layers[i]
            x = gnn_layer(W, x)

        # 输出反量化
        if self.quantize:
            x = self.output_quant(x)

        return x


class ScoreLayer(nn.Module):
    """可配置的评分层，支持多种归一化和后处理选项"""

    def __init__(
        self,
        in_feat: int = 32,
        max_size: int = 64,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        normalization: str = "softmax",
        temperature: float = 1.0,
    ):
        super(ScoreLayer, self).__init__()
        self.max_size = max_size
        self.normalization = normalization
        self.temperature = temperature

        # 构建评分网络
        layers = []

        # 隐藏层
        if hidden_dims:
            current_dim = in_feat
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, hidden_dim))

                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU(0.2))
                elif activation == "gelu":
                    layers.append(nn.GELU())

                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

                current_dim = hidden_dim

            layers.append(nn.Linear(current_dim, 1))
        else:
            # 无隐藏层
            layers.append(nn.Linear(in_feat, 1))

        self.score_net = nn.Sequential(*layers)

    def forward(self, x, n1, n2):
        """
        :param x: 节点特征，shape (max_size^2, in_feat)
        :param n1: 第一组节点数量
        :param n2: 第二组节点数量
        :return: 匹配分数矩阵，shape (n1, n2)
        """
        # 提取有效的节点特征并计算评分
        valid_features = x[: n1 * n2]
        scores = self.score_net(valid_features).view(n2, n1).t()

        # 分数归一化
        if self.normalization == "softmax":
            scores = torch.softmax(scores / self.temperature, dim=-1)
        elif self.normalization == "sinkhorn":
            scores = pygm.sinkhorn(scores, tau=self.temperature, max_iter=30)
        elif self.normalization == "sigmoid":
            scores = torch.sigmoid(scores / self.temperature)
        elif self.normalization == "none":
            pass
        else:
            raise ValueError(f"不支持的归一化方法: {self.normalization}")

        return scores
