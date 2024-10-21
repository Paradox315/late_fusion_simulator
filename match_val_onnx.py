from concurrent.futures import ThreadPoolExecutor
import json
import mmap
from os import cpu_count
import os
import pickle

import numpy as np
import onnxruntime
from torch import Tensor
import onnx
import torch
import pygmtools as pygm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from src.match_dataset import MatchDataset

dataset_path = (
    "/Users/huyaowen/Projects/python/late_fusion_simulator/data/match_dataset"
)
checkpoint_path = "/Users/huyaowen/Projects/python/late_fusion_simulator/checkpoints"
pygm.set_backend("pytorch")
device = "cuda" if torch.cuda.is_available() else "cpu"


def init():
    print("Loading dataset...")
    val_dataset = MatchDataset(f"{dataset_path}/validate_parts.json", dataset_path)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
    )
    print("Dataset loaded.")
    return val_loader


def _check_and_init_gm(K, n1, n2, n1max, n2max, x0):
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


def to_numpy(tensor):
    x = tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    return x.astype(np.float32) if x.dtype == np.float64 else x


# 定义测试函数
def onnx_ngm(K, n1, n2, sk_max_iter=20, sk_tau=0.05):
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(
        K, n1, n2, None, None, None
    )
    v0 = v0 / torch.mean(v0)
    K, n1, n2, n1max, n2max, v0 = (
        to_numpy(K),
        to_numpy(n1),
        to_numpy(n2),
        to_numpy(n1max),
        to_numpy(n2max),
        to_numpy(v0),
    )
    session = onnxruntime.InferenceSession(f"{checkpoint_path}/ngm_match.onnx")
    input_feed = {
        "K": K,
        "n1": n1,
        "n2": n2,
        # "n1max": n1max,
        "n2max": n2max,
        "v0": v0,
        "sk_tau": np.array(sk_tau, dtype=np.float32),
    }
    output_name = session.get_outputs()[0].name  # 获取输出层的名称

    result = session.run([output_name], input_feed=input_feed)
    return torch.tensor(result[0])


def test(model, loader, criterion):
    total_loss, total_acc = 0, 0
    for ego_preds, cav_preds, K, gt in tqdm(loader):
        n1 = torch.tensor([ego_preds.shape[1]])
        n2 = torch.tensor([cav_preds.shape[1]])
        output = onnx_ngm(K, n1, n2)
        acc = (pygm.hungarian(output) * gt).sum() / gt.sum()
        loss = criterion(output, gt)
        total_acc += acc.item()
        total_loss += loss.item()

    return total_loss / len(loader), total_acc / len(loader)


def test_classic(algo, loader, criterion):
    total_loss, total_acc = 0, 0
    for ego_preds, cav_preds, K, gt in tqdm(loader):
        n1 = torch.tensor([ego_preds.shape[1]])
        n2 = torch.tensor([cav_preds.shape[1]])
        output = algo(K, n1, n2)
        acc = (pygm.hungarian(output) * gt).sum() / gt.sum()
        loss = criterion(output, gt)
        total_acc += acc.item()
        total_loss += loss.item()
    return total_loss / len(loader), total_acc / len(loader)


def get_network():
    if os.path.exists(f"{checkpoint_path}/ngm_match.onnx"):
        print("Loading ONNX Model...")
        net = onnx.load(f"{checkpoint_path}/ngm_match.onnx")
        onnx.checker.check_model(net)
    else:
        raise FileNotFoundError("Model not found.")
    return net


if __name__ == "__main__":
    val_loader = init()
    net = get_network()
    criterion = pygm.utils.permutation_loss
    print("Testing...")
    test_loss, test_acc = test(net, val_loader, criterion)
    print(f"Test Loss: {test_loss}, Test Acc: {test_acc}")
    print("Testing classic algorithms...")
    test_loss, test_acc = test_classic(pygm.rrwm, val_loader, criterion)
    print(f"Test Loss: {test_loss}, Test Acc: {test_acc}")
    test_loss, test_acc = test_classic(pygm.ipfp, val_loader, criterion)
    print(f"Test Loss: {test_loss}, Test Acc: {test_acc}")
    test_loss, test_acc = test_classic(pygm.sm, val_loader, criterion)
    print(f"Test Loss: {test_loss}, Test Acc: {test_acc}")
