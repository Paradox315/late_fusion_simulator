import torch
import torch.onnx

from src.train.gcn_train_v2 import get_network

# 1. 定义或加载PyTorch模型（示例为ResNet18）
model = torch.load("checkpoints/gcn_v2_model_1.pth", weights_only=False)
model.eval()  # 设置为评估模式[2,9](@ref)

# 2. 创建虚拟输入（需与模型输入尺寸一致）
ego_preds = torch.randn(32, 8)
ego_mask = torch.zeros(32, dtype=torch.bool)
cav_preds = torch.randn(32, 8)
cav_mask = torch.zeros(32, dtype=torch.bool)
dummy_input = (ego_preds, ego_mask, cav_preds, cav_mask)
# traced_model = torch.jit.trace(model, dummy_input)
# traced_model.save("checkpoints/gcn_v2_model_1.pt")
# print("Traced model saved to checkpoints/gcn_v2_model_1.pt")
# print(traced_model.code)  # 输出计算图结构
# 3. 导出为ONNX
onnx_path = "checkpoints/gcn_net_v2.onnx"
# torch.onnx.select_model_mode_for_export(model, torch.onnx.TrainingMode.EVAL)
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,  # 导出模型参数[2,10](@ref)
    input_names=["ego_preds", "ego_mask", "cav_preds", "cav_mask"],  # 输入节点名称
    output_names=["output"],  # 输出节点名称
)
print("ONNX model saved to", onnx_path)
