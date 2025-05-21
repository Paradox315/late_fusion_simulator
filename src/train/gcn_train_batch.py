import argparse
import json
import os

import pygmtools as pygm
import torch
from tqdm import tqdm

from match_dataset_v2 import create_data_loaders
from src.networks.gcn_net_batch import GCN_Net, ScoreLayer


def parse_args():
    parser = argparse.ArgumentParser(description="GCN匹配网络训练")
    # 数据集参数
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../../data/detect_dataset",
        help="数据集路径",
    )
    parser.add_argument("--preload_data", action="store_true", help="预加载数据")
    parser.add_argument("--cache_size", type=int, default=100, help="缓存大小")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载器工作线程数")
    parser.add_argument("--batch_size", type=int, default=4, help="批大小")
    parser.add_argument("--max_size", type=int, default=8, help="最大节点数量")

    # GCN网络参数
    parser.add_argument(
        "--gnn_channels", type=str, default="32,64,32", help="GCN各层通道数,逗号分隔"
    )
    parser.add_argument("--input_dim", type=int, default=32, help="输入特征维度")
    parser.add_argument(
        "--update_type", type=str, default="linear", help="更新类型: linear,conv"
    )
    parser.add_argument(
        "--dropout_rates",
        type=str,
        default="0.0,0.1,0.1",
        help="各层Dropout比率,逗号分隔",
    )
    parser.add_argument(
        "--activations",
        type=str,
        default="relu,relu,relu",
        help="各层激活函数,逗号分隔",
    )
    parser.add_argument("--concat_features", action="store_true", help="拼接特征而不是相加")

    # ScoreLayer参数
    parser.add_argument(
        "--score_hidden_dims",
        type=str,
        default="32,32",
        help="评分层隐藏层维度,逗号分隔",
    )
    parser.add_argument("--score_dropout", type=float, default=0.1, help="评分层dropout比率")
    parser.add_argument("--score_activation", type=str, default="relu", help="评分层激活函数")
    parser.add_argument(
        "--normalization",
        type=str,
        default="sinkhorn",
        help="归一化方法: softmax,sinkhorn,sigmoid,none",
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="温度参数")

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--lr_step", type=int, default=10, help="学习率衰减步长")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="学习率衰减因子")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="权重衰减")
    parser.add_argument("--save_interval", type=int, default=5, help="保存间隔(轮)")

    # 添加早停相关参数
    parser.add_argument("--early_stop", action="store_true", help="启用早停")
    parser.add_argument("--patience", type=int, default=2, help="早停耐心值")
    parser.add_argument("--min_delta", type=float, default=0.001, help="最小提升阈值")

    # 添加量化相关参数
    parser.add_argument("--quantize", action="store_true", help="启用模型量化")

    # 其他参数
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="../../checkpoints",
        help="检查点保存路径",
    )
    parser.add_argument("--config", type=str, default=None, help="配置文件路径(将覆盖命令行参数)")
    parser.add_argument("--device", type=str, default=None, help="指定设备")

    args = parser.parse_args()

    # 从配置文件加载参数(如果提供)
    if args.config is not None and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
            args.__dict__.update(config)

    # 处理列表参数
    args.gnn_channels = [int(x) for x in args.gnn_channels.split(",")]
    args.dropout_rates = [float(x) for x in args.dropout_rates.split(",")]
    args.activations = args.activations.split(",")
    if args.score_hidden_dims:
        args.score_hidden_dims = [int(x) for x in args.score_hidden_dims.split(",")]
    else:
        args.score_hidden_dims = None

    # 确保参数列表长度匹配
    if len(args.dropout_rates) != len(args.gnn_channels):
        args.dropout_rates = [args.dropout_rates[0]] * len(args.gnn_channels)
    if len(args.activations) != len(args.gnn_channels):
        args.activations = [args.activations[0]] * len(args.gnn_channels)

    return args


class EarlyStoppingCallback:
    """早停回调函数，在验证指标不再改善时停止训练"""

    def __init__(self, patience=3, min_delta=0.001, mode="max"):
        """
        初始化早停回调

        参数:
            patience (int): 容忍验证指标停滞的轮数
            min_delta (float): 最小增益阈值
            mode (str): 'min' 对于越小越好的指标，'max' 对于越大越好的指标
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, epoch, score):
        """
        检查是否应该停止训练

        参数:
            epoch (int): 当前训练轮数
            score (float): 当前验证指标

        返回:
            bool: 如果应该停止训练则返回True
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == "min":
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1
        else:  # mode == "max"
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            print(f"早停触发! 连续 {self.patience} 轮未改善。最佳性能在第 {self.best_epoch} 轮")
            return True

        return False


def init_models(args, device):
    """初始化模型"""
    # GCN网络
    gcn_model = GCN_Net(
        gnn_channels=args.gnn_channels,
        input_dim=args.input_dim,
        max_size=args.max_size,
        dropout_rates=args.dropout_rates,
        activations=args.activations,
        update_type=args.update_type,
        quantize=args.quantize,
    ).to(device)

    # 评分层
    score_model = ScoreLayer(
        in_feat=args.gnn_channels[-1],
        max_size=args.max_size,
        hidden_dims=args.score_hidden_dims,
        activation=args.score_activation,
        dropout_rate=args.score_dropout,
        normalization=args.normalization,
        temperature=args.temperature,
    ).to(device)

    return gcn_model, score_model


def load_checkpoint(gcn_model, score_model, checkpoint_path, device):
    """加载检查点"""
    epoch = 0

    if os.path.exists(checkpoint_path):
        # 查找GCN模型检查点
        gcn_checkpoints = [
            f
            for f in os.listdir(checkpoint_path)
            if f.startswith("gcn_layer_model_v1_linear")
            and f.endswith(".pth")
            and f.split("_")[-1].split(".")[0].isdigit()
        ]
        score_checkpoints = [
            f
            for f in os.listdir(checkpoint_path)
            if f.startswith("gcn_score_model_v1_linear")
            and f.endswith(".pth")
            and f.split("_")[-1].split(".")[0].isdigit()
        ]

        if gcn_checkpoints and score_checkpoints:
            # 解析最新epoch
            gcn_epochs = [int(f.split("_")[-1].split(".")[0]) for f in gcn_checkpoints]
            score_epochs = [
                int(f.split("_")[-1].split(".")[0]) for f in score_checkpoints
            ]

            common_epochs = set(gcn_epochs).intersection(score_epochs)
            if common_epochs:
                epoch = max(common_epochs)

                # 加载模型
                print(f"加载第 {epoch} 轮检查点...")
                gcn_state_dict = torch.load(
                    f"{checkpoint_path}/gcn_layer_model_v1_linear_{epoch}.pth",
                    map_location=device,
                )
                score_state_dict = torch.load(
                    f"{checkpoint_path}/gcn_score_model_v1_linear_{epoch}.pth",
                    map_location=device,
                )

                gcn_model.load_state_dict(gcn_state_dict)
                score_model.load_state_dict(score_state_dict)

    return epoch


def save_checkpoint(gcn_model, score_model, checkpoint_path, epoch):
    """保存检查点"""
    print(f"保存第 {epoch} 轮检查点...")
    os.makedirs(checkpoint_path, exist_ok=True)

    # 保存模型
    torch.save(
        gcn_model.state_dict(),
        f"{checkpoint_path}/gcn_layer_model_v1_linear_{epoch}.pth",
    )
    torch.save(
        score_model.state_dict(),
        f"{checkpoint_path}/gcn_score_model_v1_linear_{epoch}.pth",
    )

    # 保存最新检查点链接
    torch.save(
        gcn_model.state_dict(),
        f"{checkpoint_path}/gcn_layer_model_v1_linear_latest.pth",
    )
    torch.save(
        score_model.state_dict(),
        f"{checkpoint_path}/gcn_score_model_v1_linear_latest.pth",
    )


def preprocess_batch_data(K, batched_n1n2, max_size, node_feat_dim):
    """预处理批处理数据，填充K矩阵并生成节点特征
    参数:
        K: 原始矩阵 shape: (batch_size, max_size^2, max_size^2)
        batched_n1n2: 实际节点数 shape: (batch_size, 2)
        max_size: 最大节点数（每一维）
        node_feat_dim: 节点特征维度

    返回:
        W: 加权邻接矩阵 shape: (batch_size, max_size^2, max_size^2)
        x: 节点特征 shape: (batch_size, max_size^2, node_feat_dim)
    """
    batch_size = K.size(0)
    max_nodes_squared = max_size**2

    # 初始化返回结果
    W = K
    x = torch.zeros((batch_size, max_nodes_squared, node_feat_dim), device=K.device)

    # 批处理计算
    # 计算邻接矩阵
    A = (K != 0).float()
    A_sum = A.sum(dim=2, keepdim=True) + 1e-8  # 归一化因子
    A_norm = A / A_sum  # 归一化
    W = A_norm * K  # 加权

    # 节点特征初始化
    for b in range(batch_size):
        # 1. 对角线元素作为第一个特征
        x[b, :, 0] = torch.diagonal(K[b])
        # 2. 每行最大值作为第二个特征
        x[b, :, 1] = torch.max(K[b], dim=1)[0]
        # 3. 每行均值作为第三个特征
        x[b, :, 2] = torch.mean(K[b], dim=1)
        # 4. 每行非零元素数量作为第四个特征
        x[b, :, 3] = (K[b] != 0).float().sum(dim=1) / max_nodes_squared

    # 5. 使用位置编码填充剩余特征 (对所有batch相同)
    position_encoding = torch.zeros(
        (max_nodes_squared, node_feat_dim - 4), device=K.device
    )
    for j in range(node_feat_dim - 4):
        if j % 2 == 0:
            position_encoding[:, j] = torch.sin(
                torch.arange(max_nodes_squared, device=K.device)
                / (10000 ** ((j + 4) / node_feat_dim))
            )
        else:
            position_encoding[:, j] = torch.cos(
                torch.arange(max_nodes_squared, device=K.device)
                / (10000 ** ((j + 3) / node_feat_dim))
            )

    # 复制位置编码到所有批次
    x[:, :, 4:] = position_encoding.unsqueeze(0).expand(batch_size, -1, -1)

    return W, x


def gcn_batch_match(batched_K, batched_n1n2, gcn_model, score_model, max_size):
    """批处理匹配计算"""
    batch_size = batched_K.size(0)
    outputs = torch.zeros((batch_size, max_size, max_size), device=batched_K.device)
    K = batched_K
    W, x = preprocess_batch_data(K, batched_n1n2, max_size, gcn_model.gnn_channels[0])
    # 通过GCN处理
    x = gcn_model(W, x)
    for b in range(batch_size):
        n1, n2 = batched_n1n2[b]
        # 生成评分
        output = score_model(x[b], n1, n2)
        outputs[b, :n1, :n2] = output[:n1, :n2]

    return outputs


def train(
    gcn_model,
    score_model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    args,
    start_epoch=0,
    early_stopper=None,
):
    """训练函数"""
    gcn_model.train()
    score_model.train()

    losses = []
    accs = []
    best_acc = 0

    pygm.set_backend("pytorch")

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        print(f"Epoch {epoch + 1}/{start_epoch + args.num_epochs}")
        total_loss, total_acc = 0, 0

        pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}")
        for batch_idx, (
            ego_preds,
            cav_preds,
            K_batch,
            gt_batch,
            batch_n1n2,
        ) in enumerate(pbar):
            K_batch = K_batch.to(device)
            gt_batch = gt_batch.to(device)
            batch_n1n2 = batch_n1n2.to(device)

            # 前向传播
            output = gcn_batch_match(
                K_batch, batch_n1n2, gcn_model, score_model, args.max_size
            )

            # 计算准确率
            # acc = 0
            # for b in range(batch_size):
            #     match_output = pygm.hungarian(output[b].unsqueeze(0))
            #     acc += (match_output * gt_batch[b]).sum() / (gt_batch[b].sum() + 1e-8)
            # acc /= batch_size
            match_output = pygm.hungarian(output)
            acc = (match_output * gt_batch).sum() / (gt_batch.sum() + 1e-8)

            # 计算损失并反向传播
            loss = criterion(output, gt_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新统计
            total_loss += loss.item()
            total_acc += acc.item()

            # 更新进度条
            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "acc": acc.item(),
                    "avg_loss": total_loss / (batch_idx + 1),
                    "avg_acc": total_acc / (batch_idx + 1),
                }
            )

        # 学习率调整
        scheduler.step()

        # 计算平均指标
        mean_loss = total_loss / len(train_loader)
        mean_acc = total_acc / len(train_loader)

        print(f"Epoch {epoch + 1}, Loss: {mean_loss:.4f}, Acc: {mean_acc:.4f}")
        losses.append(mean_loss)
        accs.append(mean_acc)

        # 验证
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.num_epochs:
            val_loss, val_acc = test(
                gcn_model, score_model, val_loader, criterion, device, args
            )
            print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            # 保存检查点
            save_checkpoint(gcn_model, score_model, args.checkpoint_path, epoch + 1)

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"新的最佳精度: {best_acc:.4f}, 保存最佳模型...")
                save_checkpoint(gcn_model, score_model, args.checkpoint_path, "best")

            if early_stopper and early_stopper(epoch, val_acc):
                print(f"早停触发! 在第 {epoch + 1} 轮停止训练。")
                break

    return losses, accs


def test(gcn_model, score_model, test_loader, criterion, device, args):
    """测试函数"""
    gcn_model.eval()
    score_model.eval()

    total_loss, total_acc = 0, 0
    batch_count = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="测试")
        for batch_idx, (
            ego_preds,
            cav_preds,
            K_batch,
            gt_batch,
            batch_n1n2,
        ) in enumerate(pbar):
            K_batch = K_batch.to(device)
            gt_batch = gt_batch.to(device)
            batch_n1n2 = batch_n1n2.to(device)

            # 提取各样本的点数
            batch_size = K_batch.size(0)
            # 前向传播
            output = gcn_batch_match(
                K_batch, batch_n1n2, gcn_model, score_model, args.max_size
            )

            # 计算准确率
            acc = 0
            for b in range(batch_size):
                n1, n2 = batch_n1n2[b]
                if n1 > 0 and n2 > 0:  # 确保有效节点存在
                    match_output = pygm.hungarian(output[b].unsqueeze(0))
                    acc += (match_output * gt_batch[b]).sum() / (
                        gt_batch[b].sum() + 1e-8
                    )
                    batch_count += 1

            # 计算损失
            loss = criterion(output, gt_batch)

            # 更新统计
            total_loss += loss.item() * batch_size
            total_acc += acc.item()

            # 更新进度条
            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "avg_loss": total_loss / (batch_idx + 1) / batch_size,
                }
            )

    return total_loss / len(test_loader), total_acc / batch_count


if __name__ == "__main__":
    # 解析参数
    args = parse_args()

    # 确定设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 设置PyGM后端
    pygm.set_backend("pytorch")

    # 初始化数据集
    train_loader, val_loader, test_loader = create_data_loaders(args)
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")

    # 初始化模型
    gcn_model, score_model = init_models(args, device)

    # 尝试加载检查点
    start_epoch = load_checkpoint(gcn_model, score_model, args.checkpoint_path, device)

    # 打印模型结构
    print("GCN模型结构:")
    print(gcn_model)
    print("\nScore模型结构:")
    print(score_model)
    print(
        "\n模型参数总数:",
        sum(p.numel() for p in gcn_model.parameters())
        + sum(p.numel() for p in score_model.parameters()),
    )

    # 初始化优化器和损失函数
    optimizer = torch.optim.Adam(
        [{"params": gcn_model.parameters()}, {"params": score_model.parameters()}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma
    )

    criterion = pygm.utils.permutation_loss
    # 初始化早停
    early_stopper = None
    if args.early_stop:
        early_stopper = EarlyStoppingCallback(
            patience=args.patience,
            min_delta=args.min_delta,
            mode="max",  # 使用准确率作为早停指标，越高越好
        )
    # 训练模型
    print(f"开始训练 {args.num_epochs} 轮...")
    losses, accs = train(
        gcn_model,
        score_model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        args,
        start_epoch,
        early_stopper,
    )

    # 保存最终模型
    save_checkpoint(
        gcn_model, score_model, args.checkpoint_path, start_epoch + args.num_epochs
    )

    # 最终测试
    print("最终测试评估...")
    test_loss, test_acc = test(
        gcn_model, score_model, test_loader, criterion, device, args
    )
    print(f"测试结果 - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    # 保存训练历史
    history = {
        "train_loss": losses,
        "train_acc": accs,
        "final_test_loss": test_loss,
        "final_test_acc": test_acc,
    }

    with open(f"{args.checkpoint_path}/training_history.json", "w") as f:
        json.dump(history, f)
