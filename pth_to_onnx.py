import torch
import torch.onnx

from src.match_dataset import MatchDataset
from src.networks.gcn_net_batch import GCN_Net
from src.train.gcn_train_v2 import get_network
from torch.utils.data import DataLoader

dataset_path = "./data/match_dataset"
# 1. 定义或加载PyTorch模型（示例为ResNet18）
model = GCN_Net((32, 32, 32))
state_dict = torch.load("checkpoints/gcn_v3_model_3.pth", weights_only=False)
model.load_state_dict(state_dict)

model.eval()  # 设置为评估模式[2,9](@ref)
test_dataset = MatchDataset(f"{dataset_path}/test_parts.json", dataset_path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
max_size = 32
ego_preds, cav_preds, K, _ = next(iter(test_loader))
K = K[0]
n1, n2 = ego_preds.shape[1], cav_preds.shape[1]
K_padded = torch.zeros((max_size**2, max_size**2))
K_padded[: n1 * n2, : n1 * n2] = K
mask = torch.zeros((max_size, max_size))
mask[:n1, :n2] = True
# 2. 创建虚拟输入（需与模型输入尺寸一致）
dummy_input = (K_padded, mask)
# traced_model = torch.jit.trace(model, dummy_input)
# traced_model.save("checkpoints/gcn_v2_model_1.pt")
# print("Traced model saved to checkpoints/gcn_v2_model_1.pt")
# print(traced_model.code)  # 输出计算图结构
# 3. 导出为ONNX
onnx_path = "checkpoints/gcn_net_v3_1.onnx"
torch.onnx.select_model_mode_for_export(model, torch.onnx.TrainingMode.EVAL)
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,  # 是否导出训练好的参数
    input_names=["K", "mask"],  # 输入节点名称
    output_names=["output"],  # 输出节点名称
)
print("ONNX model saved to", onnx_path)
