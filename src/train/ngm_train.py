import os

import pygmtools as pygm
from pygmtools.pytorch_backend import NGM_Net
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.match_dataset import MatchDataset
from src.networks.gcn_net import GCN_Net
from src.utils.lap import ngm_match

dataset_path = "../../data/match_dataset"
checkpoint_path = "../../checkpoints"
pygm.set_backend("pytorch")
device = torch.device("mps" if torch.cuda.is_available() else "cpu")


def init():
    train_dataset = MatchDataset(f"{dataset_path}/train_parts.json", dataset_path)
    test_dataset = MatchDataset(f"{dataset_path}/test_parts.json", dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


# 定义训练函数
def train(model, loader, criterion, optimizer, num_epochs=3):
    model.train()
    losses = []
    accs = []
    optimizer.zero_grad()  # Reset gradients tensors
    for epoch in range(num_epochs):
        total_loss, total_acc = 0, 0
        for i, (ego_preds, cav_preds, K, gt) in enumerate(tqdm(loader)):
            n1 = torch.tensor([ego_preds.shape[1]])
            n2 = torch.tensor([cav_preds.shape[1]])
            output = ngm_match(K, n1, n2, model)
            acc = (pygm.hungarian(output) * gt).sum() / gt.sum()
            loss = criterion(output, gt)
            loss.backward()  # Backward pass
            optimizer.step()  # Now we can do an optimizer step
            optimizer.zero_grad()  # Reset gradients tensors
            total_loss += loss.item()
            total_acc += acc.item()
        mean_loss = total_loss / len(loader)
        mean_acc = total_acc / len(loader)
        print(f"Epoch {epoch + 1}, Loss: {mean_loss}, Acc: {mean_acc}")
        losses.append(mean_loss)
        accs.append(mean_acc)

    return losses, accs


# 定义测试函数
def test(model, loader, criterion):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for ego_preds, cav_preds, K, gt in tqdm(loader):
            n1 = torch.tensor([ego_preds.shape[1]])
            n2 = torch.tensor([cav_preds.shape[1]])
            output = ngm_match(K, n1, n2, model)
            acc = (pygm.hungarian(output) * gt).sum() / gt.sum()
            loss = criterion(output, gt)
            total_acc += acc.item()
            total_loss += loss.item()
    return total_loss / len(loader), total_acc / len(loader)


def get_network():
    if os.path.exists(f"{checkpoint_path}/ngm_model.pth"):
        print("Loading checkpoint...")
        net = torch.load(f"{checkpoint_path}/ngm_model.pth", map_location=device)
    else:
        print("Initializing network...")
        net = NGM_Net(
            gnn_channels=(16, 16, 16),
            sk_emb=1,
        )
    net.to(device)
    return net


def save_checkpoint(model):
    print("Saving checkpoint...")
    torch.save(model, f"{checkpoint_path}/ngm_model.pth")


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


if __name__ == "__main__":
    train_loader, test_loader = init()
    # net = get_network()
    net = NGM_Net(
        gnn_channels=(16, 16, 16),
        sk_emb=1,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = pygm.utils.permutation_loss
    losses, accs = train(net, train_loader, criterion, optimizer, num_epochs=1)
    # save_checkpoint(net)
    print(f"Train Loss: {losses}, Train Acc: {accs}")

    print("Testing...")
    test_loss, test_acc = test(net, test_loader, criterion)
    print(f"Test Loss: {test_loss}, Test Acc: {test_acc}")

    print("Testing classic algorithms...")
    test_loss, test_acc = test_classic(pygm.rrwm, test_loader, criterion)
    print(f"Test Loss: {test_loss}, Test Acc: {test_acc}")
