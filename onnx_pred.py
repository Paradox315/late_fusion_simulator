import onnx
import onnxruntime as ort
import pygmtools as pygm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.match_dataset import MatchDataset

dataset_path = "data/match_dataset"
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
    model = onnx.load("checkpoints/gcn_net_v3_1.onnx")
    onnx.checker.check_model(model)
    ort_session = ort.InferenceSession("checkpoints/gcn_net_v3_1.onnx")
    acc1 = test(ort_session, test_loader)
    print(f"Test Accuracy: {acc1}")
