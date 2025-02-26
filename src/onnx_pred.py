import numpy as np
import onnx
import onnxruntime as ort
import pygmtools as pygm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.match_dataset import MatchDataset
from src.networks.gcn_net_batch import GCN_Net
from src.train.gcn_train_batch import test as gcn_test

dataset_path = "../data/match_dataset"
pygm.set_backend("pytorch")
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model_name = "gcn_v2"
torch.manual_seed(42)
max_size = 32


def init():
    train_dataset = MatchDataset(f"{dataset_path}/train_parts.json", dataset_path)
    test_dataset = MatchDataset(f"{dataset_path}/test_parts.json", dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def test(ort_session, loader):
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for ego_preds, cav_preds, K, gt in tqdm(loader):
            K = K[0]
            n1, n2 = ego_preds.shape[1], cav_preds.shape[1]
            K_padded = torch.zeros((max_size**2, max_size**2), device=device)
            K_padded[: n1 * n2, : n1 * n2] = K
            mask = torch.zeros((max_size, max_size), device=device)
            mask[:n1, :n2] = True
            inputs = {
                "K": to_numpy(K_padded),
                "mask": to_numpy(mask),
            }
            output = ort_session.run(None, inputs)
            output = torch.tensor(output[0])[:n1, :n2].unsqueeze(0)
            acc = (pygm.hungarian(output) * gt).sum() / gt.sum()
            total_acc += acc.item()
    return total_acc / len(loader)


if __name__ == "__main__":
    train_loader, test_loader = init()
    model = onnx.load("../checkpoints/gcn_net_v3_1.onnx")
    onnx.checker.check_model(model)
    ort_session = ort.InferenceSession("../checkpoints/gcn_net_v3_1.onnx")
    K = torch.randn((1024, 1024))
    mask = torch.zeros((32, 32))
    net = GCN_Net((32, 32, 32))
    state_dict = torch.load("../checkpoints/gcn_v3_model_3.pth", weights_only=False)
    net.load_state_dict(state_dict)
    net.eval()
    torch_out = net(K, mask)
    ort_inputs = {
        "K": to_numpy(K),
        "mask": to_numpy(mask),
    }
    ort_outs = ort_session.run(None, ort_inputs)
    print(torch_out)
    print(ort_outs[0])
    # Compare the output with PyTorch
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    acc1 = test(ort_session, test_loader)
    print(f"Test Accuracy: {acc1}")
    _, acc2 = gcn_test(net, test_loader, pygm.utils.permutation_loss)
    print(f"Test Accuracy: {acc2}")
