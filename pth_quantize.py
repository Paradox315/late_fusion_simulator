import torch
from torch.utils.data import DataLoader
import pygmtools as pygm
from networks.gcn_net_batch import ScoreLayer
from src.match_dataset_v2 import MatchDataset
from src.networks.gcn_net_batch import GCN_Net
from tqdm import tqdm

from train.gcn_train_batch import gcn_batch_match

pygm.set_backend("pytorch")


def print_model_info(model):
    """Prints model architecture and parameter information."""
    print("Model structure:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    for name, param in model.named_parameters():
        print(f"Layer: {name}, Type: {param.dtype}, Shape: {param.shape}")


def quant_model(net):
    import torch.ao.quantization as quantizer

    activation_quant = quantizer.FakeQuantize.with_args(
        observer=quantizer.MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )

    weight_quant = quantizer.FakeQuantize.with_args(
        observer=quantizer.MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )

    # assign qconfig to model
    net.qconfig = quantizer.QConfig(activation=activation_quant, weight=weight_quant)

    # prepare qat model using qconfig settings
    quantizer.prepare_qat(net, inplace=True)


if __name__ == "__main__":
    # Load dataset and model
    model = GCN_Net(
        gnn_channels=[32, 64, 32],
        input_dim=32,
        max_size=8,
        dropout_rates=[0, 0.1, 0.1],
        activations=["relu", "relu", "relu"],
        update_type="linear",
    )
    # 评分层
    score_model = ScoreLayer(
        in_feat=32,
        max_size=8,
        hidden_dims=[32, 32],
        activation="relu",
        dropout_rate=0.1,
        normalization="softmax",
        temperature=0.1,
    )

    state_dict = torch.load(
        "checkpoints/gcn_layer_model_v1_linear_best.pth", weights_only=True
    )
    model.load_state_dict(state_dict)
    state_dict = torch.load(
        "checkpoints/gcn_score_model_v1_linear_best.pth", weights_only=True
    )
    score_model.load_state_dict(state_dict)
    # Create dummy input
    max_size = 8

    K = torch.randn((1, max_size**2, max_size**2), dtype=torch.float)
    x = torch.randn((1, max_size**2, 32), dtype=torch.float)
    dummy_input = [K, x]

    # Print original model info
    print("Original model:")
    print_model_info(model)

    # Quantize model

    quant_model(model)

    calidate_dataset = MatchDataset(
        "./data/detect_dataset",
        max_size=max_size,
        split="train",
    )

    calidate_loader = DataLoader(
        calidate_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    criterion = pygm.utils.permutation_loss
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": score_model.parameters()}],
        lr=1e-3,
        weight_decay=1e-4,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pbar = tqdm(calidate_loader, desc="测试")
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
        output = gcn_batch_match(K_batch, batch_n1n2, model, score_model, 8)

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

        # 更新进度条
        pbar.set_postfix(
            {
                "loss": loss.item(),
                "acc": acc.item(),
            }
        )

    torch.backends.quantized.engine = "qnnpack"
    to_trace_model = torch.ao.quantization.convert(model, inplace=False)
    ts = torch.jit.trace(to_trace_model, dummy_input, strict=False, check_trace=False)
    torch.jit.save(ts, "./checkpoints/gcn_layer_model_linear_quantized.pt")
