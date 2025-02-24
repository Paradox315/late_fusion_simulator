import os

import onnx
import pygmtools as pygm
import torch
import onnxruntime as ort
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.match_dataset import MatchDataset
from src.networks.gcn_net_v2 import pad_predictions

dataset_path = "../data/match_dataset"
pygm.set_backend("pytorch")
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model_name = "gcn_v2"
torch.manual_seed(42)


def init():
    train_dataset = MatchDataset(f"{dataset_path}/train_parts.json", dataset_path)
    test_dataset = MatchDataset(f"{dataset_path}/test_parts.json", dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


def test(ort_session, loader):
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for ego_preds, cav_preds, K, gt in tqdm(loader):
            ego_preds, ego_mask = pad_predictions(ego_preds[0, :, 1:])
            cav_preds, cav_mask = pad_predictions(cav_preds[0, :, 1:])
            n1, n2 = ego_mask.sum(), cav_mask.sum()
            inputs = {
                "ego_preds": ego_preds.numpy(),
                "ego_mask": ego_mask.numpy(),
                "cav_preds": cav_preds.numpy(),
                "cav_mask": cav_mask.numpy(),
            }
            output = ort_session.run(None, inputs)
            output = torch.tensor(output[0])[:n1, :n2].unsqueeze(0)
            acc = (pygm.hungarian(output) * gt).sum() / gt.sum()
            total_acc += acc.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    train_loader, test_loader = init()
    # model = onnx.load("../checkpoints/gcn_net_v2.onnx")
    # onnx.checker.check_model(model)
    ort_session = ort.InferenceSession("../checkpoints/gcn_net_v2.onnx")
    acc = test(ort_session, test_loader)
    print(f"Test accuracy: {acc}")
