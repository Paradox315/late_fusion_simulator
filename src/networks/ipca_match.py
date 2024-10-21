import pickle
import torch
import pygmtools as pygm
import matplotlib.pyplot as plt
from src.match_dataset import MatchDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

dataset_path = "../../data/match_dataset"
train = pickle.load(open(dataset_path + "/train.pkl", "rb"))
test = pickle.load(open(dataset_path + "/test.pkl", "rb"))
train_dataset = MatchDataset(train)
test_dataset = MatchDataset(test)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
pygm.set_backend("pytorch")


# 定义训练函数
def train(model, loader, criterion, optimizer, accumulation_steps=4):
    model.train()
    total_loss = 0
    total_acc = 0
    losses = []
    optimizer.zero_grad()  # Reset gradients tensors
    for i, (ego_preds, cav_preds, K, gt) in enumerate(tqdm(loader)):
        ego_preds_len, cav_preds_len = ego_preds.shape[1], cav_preds.shape[1]
        A1 = torch.ones(1, ego_preds_len, ego_preds_len).float()
        torch.diagonal(A1, dim1=1, dim2=2)[:] = 0
        A2 = torch.ones(1, cav_preds_len, cav_preds_len).float()
        torch.diagonal(A2, dim1=1, dim2=2)[:] = 0
        output = pygm.ipca_gm(ego_preds, cav_preds, A1, A2, network=model)
        acc = (pygm.hungarian(output) * gt).sum() / gt.sum()
        loss = criterion(output, gt)
        loss = loss / accumulation_steps  # Normalize our loss (if averaged)
        loss.backward()  # Backward pass
        if (i + 1) % accumulation_steps == 0:  # Wait for several backward steps
            optimizer.step()  # Now we can do an optimizer step
            optimizer.zero_grad()  # Reset gradients tensors
            if (i + 1) % (100 * accumulation_steps) == 0:
                print(f"Epoch {i+1}, Loss: {loss.item()}, Acc: {acc.item()}")
                losses.append(loss.item())
        total_loss += loss.item()
        total_acc += acc.item()
    return losses, total_loss / len(loader), total_acc / len(loader)


# 定义测试函数
def test(model, loader, criterion):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for ego_preds, cav_preds, K, gt in tqdm(loader):
            ego_preds_len, cav_preds_len = ego_preds.shape[1], cav_preds.shape[1]
            A1 = torch.ones(1, ego_preds_len, ego_preds_len)
            A2 = torch.ones(1, cav_preds_len, cav_preds_len)
            output = pygm.ipca_gm(ego_preds, cav_preds, A1, A2, network=model)
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


if __name__ == "__main__":
    net = pygm.utils.get_network(
        pygm.ipca_gm,
        in_channel=9,
        hidden_channel=512,
        out_channel=128,
        num_layers=3,
        pretrain=False,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = pygm.utils.permutation_loss
    losses, train_loss, train_acc = train(net, train_loader, criterion, optimizer)
    print(f"Train Loss: {train_loss}, Train Acc: {train_acc}")
    plt.plot(losses)
    plt.show()
    print("Testing...")
    test_loss, test_acc = test(net, test_loader, criterion)
    print(f"Test Loss: {test_loss}, Test Acc: {test_acc}")
    print("Testing classic algorithms...")
    test_loss, test_acc = test_classic(pygm.rrwm, test_loader, criterion)
    print(f"Test Loss: {test_loss}, Test Acc: {test_acc}")
