from torch import Tensor
import torch


def _check_and_init_gm(K, n1, n2, n1max, n2max, x0):
    # get batch number
    batch_num = K.shape[0]
    n1n2 = K.shape[1]

    # get values of n1, n2, n1max, n2max and check
    if n1 is None:
        n1 = torch.full_like((batch_num,), n1max, dtype=torch.int, device=K.device)
    elif type(n1) is Tensor and len(n1.shape) == 0:
        n1 = n1.unsqueeze(0)
    if n2 is None:
        n2 = torch.full_like((batch_num,), n2max, dtype=torch.int, device=K.device)
    elif type(n2) is Tensor and len(n2.shape) == 0:
        n2 = n2.unsqueeze(0)
    if n1max is None:
        n1max = torch.max(n1)
    if n2max is None:
        n2max = torch.max(n2)

    if not n1max * n2max == n1n2:
        raise ValueError("the input size of K does not match with n1max * n2max!")

    # initialize x0 (also v0)
    if x0 is None:
        x0 = torch.zeros(batch_num, n1max, n2max, dtype=K.dtype, device=K.device)
        for b in range(batch_num):
            x0[b, 0 : n1[b], 0 : n2[b]] = torch.tensor(1.0) / (n1[b] * n2[b])
    v0 = x0.transpose(1, 2).reshape(batch_num, n1n2, 1)

    return batch_num, n1, n2, n1max, n2max, n1n2, v0


n = 3
K = torch.randn(1, n * n, n * n)
n1, n2 = torch.tensor([n]), torch.tensor([n])
sk_max_iter = torch.tensor(20)
sk_tau = torch.tensor(0.05)
batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(
    K, n1, n2, None, None, None
)
v0 = v0 / torch.mean(v0)
model = torch.load("checkpoints/ngm_match.pth", map_location=torch.device("cpu"))
output = model(K, n1, n2, n1max, n2max, v0, sk_max_iter, sk_tau)
# export model to onnx
# torch.onnx.export(
#     model,
#     args=(K, n1, n2, n1max, n2max, v0, sk_max_iter, sk_tau),
#     dynamic_axes={
#         "K": [0, 1, 2],
#         "v0": [0, 1, 2],
#         "X": [0, 1, 2],
#     },
#     f="checkpoints/ngm_match.onnx",
#     input_names=["K", "n1", "n2", "n1max", "n2max", "v0", "sk_max_iter", "sk_tau"],
#     output_names=["X"],
#     export_params=True,
#     opset_version=11,
#     verbose=True,
# )
# export model to pt
torch_script_model = torch.jit.trace(
    model, (K, n1, n2, n1max, n2max, v0, sk_max_iter, sk_tau)
)
torch.jit.save(torch_script_model, "checkpoints/ngm_match.pt")
