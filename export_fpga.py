from networks.gcn_net_batch import GCN_Net
import torch

checkpoint_path = "checkpoints"

gcn_model = GCN_Net(
    gnn_channels=[32, 32, 32],
    input_dim=32,
    max_size=32,
    dropout_rates=[0, 0.1, 0.1],
    activations=["relu", "relu", "relu"],
    layer_norm=False,
    update_type="conv",
)

gcn_model_small = GCN_Net(
    gnn_channels=[32, 32, 32],
    input_dim=32,
    max_size=8,
    dropout_rates=[0, 0.1, 0.1],
    activations=["relu", "relu", "relu"],
    layer_norm=False,
    update_type="conv",
)
state_dict = torch.load(
    f"{checkpoint_path}/gcn_layer_model_v1_best.pth",
)


def print_model_param(model):
    print("Model Parameters:")
    print(f"Model Name: {model.__class__.__name__}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.data.shape}")


print("Small Model Load State Dict")
gcn_model_small.load_state_dict(state_dict)
print("Large Model Load State Dict")
gcn_model.load_state_dict(state_dict)
print_model_param(gcn_model)
print_model_param(gcn_model_small)
