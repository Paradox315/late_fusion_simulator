import argparse
import json
import os
import time

import pygmtools as pygm
import torch
from tqdm import tqdm
import onnxruntime as ort
from match_dataset_v2 import create_data_loaders
from src.networks.gcn_net_batch import GCN_Net, ScoreLayer


def parse_args():
    parser = argparse.ArgumentParser(description="GCN匹配网络时延测试")
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
    parser.add_argument(
        "--batch_size", type=int, default=8, help="批大小"
    )  # 该参数表示模型一次推理几帧的数据，这个可以随意调整
    parser.add_argument(
        "--max_size", type=int, default=32, help="最大节点数量"
    )  # 这个参数决定了模型的输入大小，GCN网络的输入特征矩阵的大小为 (batch_size, max_size^2, input_dim)，FPGA 上 max_size=8，rk3588上 max_size=32

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

    # 测试参数
    parser.add_argument("--num_tests", type=int, default=100, help="时延测试次数")
    parser.add_argument("--warmup", type=int, default=10, help="预热次数")

    # 量化参数
    parser.add_argument("--quantize", action="store_true", help="启用模型量化")

    # ONNX参数
    parser.add_argument("--use_onnx", action="store_true", help="使用ONNX模型进行推理")
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="../../onnx_ckpts",
        help="ONNX模型路径",
    )
    parser.add_argument("--onnx_optimize", action="store_true", help="优化ONNX模型推理")

    # 其他参数
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="../../checkpoints",
        help="检查点保存路径",
    )
    parser.add_argument("--model_version", type=str, default="best", help="要加载的模型版本")
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


def export_model_to_onnx(gcn_model, onnx_path, max_size, input_dim, device):
    """将PyTorch模型导出为ONNX格式"""
    gcn_model.eval()

    # 创建示例输入
    dummy_W = torch.randn(1, max_size**2, max_size**2, device=device)
    dummy_x = torch.randn(1, max_size**2, input_dim, device=device)

    print(f"导出ONNX模型到: {onnx_path}")
    torch.onnx.export(
        gcn_model,
        (dummy_W, dummy_x),
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["W", "x"],
        output_names=["output"],
        dynamic_axes={
            "W": {0: "batch_size"},
            "x": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print("ONNX模型导出成功!")
    return True


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


def init_onnx_model(onnx_path, optimize=False):
    """
    初始化ONNX运行时
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"找不到ONNX模型: {onnx_path}")

    print(f"加载ONNX模型: {onnx_path}")
    sess_options = ort.SessionOptions()
    if optimize:
        # 启用优化
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = os.cpu_count()

    # 创建推理会话
    onnx_session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # 获取输入输出信息
    input_names = [input.name for input in onnx_session.get_inputs()]
    output_names = [output.name for output in onnx_session.get_outputs()]

    print(f"ONNX模型输入: {input_names}")
    print(f"ONNX模型输出: {output_names}")

    return onnx_session, input_names, output_names


def load_checkpoint(gcn_model, score_model, checkpoint_path, model_version, device):
    """加载检查点"""
    if os.path.exists(checkpoint_path):
        print(f"加载模型版本: {model_version}")
        try:
            gcn_state_dict = torch.load(
                f"{checkpoint_path}/gcn_layer_model_v1_linear_{model_version}.pth",
                map_location=device,
            )
            score_state_dict = torch.load(
                f"{checkpoint_path}/gcn_score_model_v1_linear_{model_version}.pth",
                map_location=device,
            )

            gcn_model.load_state_dict(gcn_state_dict)
            score_model.load_state_dict(score_state_dict)
            print("模型加载成功!")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False
    else:
        print(f"检查点路径不存在: {checkpoint_path}")
        return False


def preprocess_batch_data(K, batched_n1n2, max_size, node_feat_dim):
    """预处理批处理数据，填充K矩阵并生成节点特征"""
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


def gcn_forward(W, x, gcn_model):
    """仅执行GCN前向传播"""
    return gcn_model(W, x)


def gcn_forward_onnx(W, x, onnx_session, input_names, output_names):
    """使用ONNX运行时执行GCN前向传播"""
    # 准备输入数据字典
    onnx_inputs = {input_names[0]: W.cpu().numpy(), input_names[1]: x.cpu().numpy()}

    # 运行推理
    onnx_outputs = onnx_session.run(output_names, onnx_inputs)

    # 转换输出为PyTorch张量
    return torch.from_numpy(onnx_outputs[0]).to(W.device)


def time_test(
    gcn_model,
    score_model,
    test_loader,
    device,
    args,
    onnx_session=None,
    input_names=None,
    output_names=None,
):
    """测量模型处理时间"""
    if gcn_model is not None:
        gcn_model.eval()
    score_model.eval()

    # 是否使用ONNX模型
    use_onnx = onnx_session is not None

    # 预热阶段
    print(f"预热 {args.warmup} 次...")
    for batch_idx, (_, _, K_batch, _, batch_n1n2) in enumerate(test_loader):
        if batch_idx >= args.warmup:
            break

        K_batch = K_batch.to(device)
        batch_n1n2 = batch_n1n2.to(device)

        node_feat_dim = args.gnn_channels[0] if not use_onnx else args.input_dim
        W, x = preprocess_batch_data(K_batch, batch_n1n2, args.max_size, node_feat_dim)

        if use_onnx:
            _ = gcn_forward_onnx(W, x, onnx_session, input_names, output_names)
        else:
            _ = gcn_forward(W, x, gcn_model)

    # 实际测量
    print(f"进行 {args.num_tests} 次时延测试...")
    preprocess_times = []
    gcn_times = []
    score_times = []
    total_times = []

    with torch.no_grad():
        for batch_idx, (_, _, K_batch, _, batch_n1n2) in enumerate(test_loader):
            if batch_idx >= args.num_tests:
                break

            K_batch = K_batch.to(device)
            batch_n1n2 = batch_n1n2.to(device)

            batch_size = K_batch.size(0)

            # 测量预处理时间
            t0 = time.time()
            node_feat_dim = args.gnn_channels[0] if not use_onnx else args.input_dim
            W, x = preprocess_batch_data(
                K_batch, batch_n1n2, args.max_size, node_feat_dim
            )
            t1 = time.time()
            preprocess_times.append((t1 - t0) * 1000)  # 毫秒

            # 测量GCN前向传播时间
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.time()
            if use_onnx:
                x_out = gcn_forward_onnx(W, x, onnx_session, input_names, output_names)
            else:
                x_out = gcn_forward(W, x, gcn_model)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t1 = time.time()
            gcn_times.append((t1 - t0) * 1000)  # 毫秒

            # 测量评分时间
            t0 = time.time()
            for b in range(batch_size):
                n1, n2 = batch_n1n2[b]
                _ = score_model(x_out[b], n1, n2)
            t1 = time.time()
            score_times.append((t1 - t0) * 1000)

            # 计算总时间
            total_times.append(preprocess_times[-1] + gcn_times[-1] + score_times[-1])

    # 计算统计数据
    avg_preprocess = sum(preprocess_times) / (len(preprocess_times) * batch_size)
    avg_gcn = sum(gcn_times) / (len(gcn_times) * batch_size)
    avg_score = sum(score_times) / (len(score_times) * batch_size)
    avg_total = sum(total_times) / (len(total_times) * batch_size)

    # 计算每秒可处理批次数
    batches_per_second = 1000 / avg_total

    print("\n时延测试结果:")
    print(f"预处理平均时间: {avg_preprocess:.2f} 毫秒")
    print(f"GCN前向传播平均时间: {avg_gcn:.2f} 毫秒")
    print(f"评分层平均时间: {avg_score:.2f} 毫秒")
    print(f"总平均时间: {avg_total:.2f} 毫秒")
    print(f"每秒处理批次数: {batches_per_second:.2f}")

    results = {
        "preprocess_time_ms": avg_preprocess,
        "gcn_time_ms": avg_gcn,
        "score_time_ms": avg_score,
        "total_time_ms": avg_total,
        "batches_per_second": batches_per_second,
        "device": str(device),
        "batch_size": args.batch_size,
        "max_size": args.max_size,
        "use_onnx": use_onnx,
    }

    return results


if __name__ == "__main__":
    # 解析参数
    args = parse_args()

    # 确定设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"批大小: {args.batch_size}")

    # 设置PyGM后端
    pygm.set_backend("pytorch")

    # 初始化数据集
    _, _, test_loader = create_data_loaders(args)
    print(f"测试集大小: {len(test_loader.dataset)}")

    # ONNX模式 vs PyTorch模式
    if args.use_onnx:
        # 检查ONNX模型是否存在
        onnx_model_name = f"{args.onnx_path}/gcn_layer_model_v1_max{args.max_size}.onnx"
        if not os.path.exists(onnx_model_name):
            print(f"ONNX模型不存在: {onnx_model_name}")
            print("尝试从PyTorch模型导出ONNX模型...")

            # 需要先加载PyTorch模型
            gcn_model, score_model = init_models(args, device)
            success = load_checkpoint(
                gcn_model, score_model, args.checkpoint_path, args.model_version, device
            )
            if not success:
                print("无法加载PyTorch模型，退出测试。")
                exit(1)

            # 导出ONNX模型
            export_model_to_onnx(
                gcn_model, onnx_model_name, args.max_size, args.gnn_channels[0], device
            )
        # 加载ONNX模型
        try:
            onnx_session, input_names, output_names = init_onnx_model(
                onnx_model_name, optimize=args.onnx_optimize
            )

            # 仍然需要Score模型
            gcn_model, score_model = init_models(args, device)
            success = load_checkpoint(
                gcn_model, score_model, args.checkpoint_path, args.model_version, device
            )
            gcn_model = None  # 不需要GCN模型
            if not success:
                print("无法加载Score模型，退出测试。")
                exit(1)

            print("ONNX模式准备就绪!")
        except Exception as e:
            print(f"ONNX模型加载失败: {e}")
            print("切换回PyTorch模式...")
            args.use_onnx = False
            onnx_session = input_names = output_names = None
    else:
        # PyTorch模式
        onnx_session = input_names = output_names = None

    # 如果不使用ONNX或ONNX加载失败，则使用PyTorch模型
    if not args.use_onnx:
        gcn_model, score_model = init_models(args, device)

        # 加载模型权重
        success = load_checkpoint(
            gcn_model, score_model, args.checkpoint_path, args.model_version, device
        )
        if not success:
            print("无法加载模型，退出测试。")
            exit(1)

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

        # 执行时延测试
    results = time_test(
        gcn_model,
        score_model,
        test_loader,
        device,
        args,
        onnx_session=onnx_session,
        input_names=input_names,
        output_names=output_names,
    )

    # 保存结果
    os.makedirs(args.checkpoint_path, exist_ok=True)
    model_type = "onnx" if args.use_onnx else "pytorch"
    results_file_name = f"latency_test_results_{model_type}_batch{args.batch_size}_max{args.max_size}.json"
    with open(
        f"{args.checkpoint_path}/{results_file_name}",
        "w",
    ) as f:
        json.dump(results, f, indent=4)

    print(f"测试结果已保存到 {args.checkpoint_path}/{results_file_name}")
