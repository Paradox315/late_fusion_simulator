import torch
import torch.onnx
import pygmtools as pygm

from match_dataset_v2 import MatchDataset
from tqdm import tqdm
from src.networks.gcn_net_batch import GCN_Net, ScoreLayer
from torch.utils.data import DataLoader

from train.gcn_train_batch import gcn_batch_match, preprocess_batch_data

pygm.set_backend("pytorch")
dataset_path = "./data/detect_dataset"
max_size = 8
# 1. 定义或加载PyTorch模型
model = GCN_Net(
    gnn_channels=[32, 64, 32],
    input_dim=32,
    max_size=max_size,
    dropout_rates=[0, 0.1, 0.1],
    activations=["relu", "relu", "relu"],
    update_type="linear",
)
state_dict = torch.load(
    "checkpoints/gcn_layer_model_v1_linear_best.pth", weights_only=True
)
model.load_state_dict(state_dict)
model.eval()  # 设置为评估模式
# 评分层
score_model = ScoreLayer(
    in_feat=32,
    max_size=max_size,
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
model.eval()
state_dict = torch.load(
    "checkpoints/gcn_score_model_v1_linear_best.pth", weights_only=True
)
score_model.load_state_dict(state_dict)
score_model.eval()

cali_dataset = MatchDataset(
    "./data/detect_dataset",
    max_size=max_size,
    split="train",
)

cali_loader = DataLoader(
    cali_dataset,
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
pbar = tqdm(cali_loader, desc="测试")
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
# 2. 创建虚拟输入（需与模型输入尺寸一致）
ego_preds, cav_preds, K_batch, gt_batch, batch_n1n2 = next(iter(cali_loader))
dummy_input = preprocess_batch_data(
    K_batch, batch_n1n2, max_size, model.gnn_channels[0]
)
# 3. 导出为ONNX
onnx_path = "checkpoints/gcn_layer_model.onnx"
torch.onnx.select_model_mode_for_export(model, torch.onnx.TrainingMode.EVAL)
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,  # 是否导出训练好的参数
    input_names=["W", "x"],  # 输入节点名称
    output_names=["x"],  # 输出节点名称
)
print("ONNX model saved to", onnx_path)
