import os
from typing import Dict

import numpy as np
import pygmtools as pygm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
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


# 定义测试函数
def test(model, loader, criterion):
    model.eval()
    loss_hist, acc_hist = [], []
    with torch.no_grad():
        for ego_preds, cav_preds, K, gt in tqdm(loader):
            n1 = torch.tensor([ego_preds.shape[1]])
            n2 = torch.tensor([cav_preds.shape[1]])
            output = pygm.ngm(K, n1, n2, network=model)
            acc = (pygm.hungarian(output) * gt).sum() / gt.sum()
            loss = criterion(output, gt)
            loss_hist.append(loss.item())
            acc_hist.append(acc.item())
    return np.array(loss_hist), np.array(acc_hist)


def test_classic(algo, loader, criterion):
    loss_hist, acc_hist = [], []
    for ego_preds, cav_preds, K, gt in tqdm(loader):
        n1 = torch.tensor([ego_preds.shape[1]])
        n2 = torch.tensor([cav_preds.shape[1]])
        output = algo(K, n1, n2)
        acc = (pygm.hungarian(output) * gt).sum() / gt.sum()
        loss = criterion(output, gt)
        loss_hist.append(loss.item())
        acc_hist.append(acc.item())
    return np.array(loss_hist), np.array(acc_hist)


def get_network():
    if os.path.exists(f"{checkpoint_path}/ngm_match.pth"):
        print("Loading checkpoint...")
        net = torch.load(f"{checkpoint_path}/ngm_match.pth", map_location=device)
    else:
        net = pygm.utils.get_network(pygm.ngm, pretrain="voc")
    net.to(device)
    return net


def plot_results(result_dict: Dict):
    plt.figure(figsize=(15, 8),dpi=300)  # 调整图像大小

    # 绘制 accuracy 趋势
    plt.subplot(1, 2, 1)
    for method, metrics in result_dict.items():
        # 使用滑动窗口平均，窗口大小为10
        accuracy = pd.Series(metrics["accuracy"]).rolling(window=50).mean()
        plt.plot(accuracy, label=method)
    plt.title("Accuracy Trend")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # 绘制 loss 趋势
    plt.subplot(1, 2, 2)
    for method, metrics in result_dict.items():
        # 使用滑动窗口平均，窗口大小为10
        loss = pd.Series(metrics["loss"]).rolling(window=50).mean()
        plt.plot(loss, label=method)
    plt.title("Loss Trend")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        "images/trend.png"
    )
    plt.show()


def plot_error_bar(result_dict):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5),dpi=300)

    methods = ["ngm", "rrwm", "sm", "ipfp"]
    metrics = ["accuracy", "loss"]
    for i, metric in enumerate(metrics):
        for method in methods:
            mean = result_dict[method][metric + "_mean"]
            std = result_dict[method][metric + "_std"]
            ax[i].errorbar(method, mean, std, fmt="o")
        ax[i].legend(methods)
        ax[i].set_title(f"{metric.capitalize()} Error Bar")
    fig.tight_layout()
    plt.savefig("images/error_bar.png")
    plt.show()


if __name__ == "__main__":
    val_loader = init()
    net = get_network()
    criterion = pygm.utils.permutation_loss
    print("Testing...")
    methods_dict = {"ngm": net, "rrwm": pygm.rrwm, "ipfp": pygm.ipfp, "sm": pygm.sm}
    result_dict={}
    for method, func in methods_dict.items():
        print(f"Testing {method} match algorithm...")
        if method == "ngm":
            loss_hist, acc_hist = test(func, val_loader, criterion)
        else:
            loss_hist, acc_hist = test_classic(func, val_loader, criterion)
        result_dict[method] = {
            "loss": loss_hist,
            "accuracy": acc_hist,
            "loss_mean": loss_hist.mean(),
            "loss_std": loss_hist.std(),
            "accuracy_mean": acc_hist.mean(),
            "accuracy_std": acc_hist.std(),
        }

        print(f"{method} mean acc:{result_dict[method]["accuracy_mean"]}")
        print(f"{method} mean loss:{result_dict[method]["loss_mean"]}")

    plot_results(result_dict)
    plot_error_bar(result_dict)
