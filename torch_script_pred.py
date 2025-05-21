import pygmtools as pygm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from networks.gcn_net_batch import GCN_Net, ScoreLayer
from src.match_dataset import MatchDataset
from train.gcn_train_batch import gcn_batch_match

torch.backends.quantized.engine = "qnnpack"
dataset_path = "data/match_dataset"
pygm.set_backend("pytorch")
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
max_size = 32

score_model = ScoreLayer()
state_dict = torch.load("checkpoints/gcn_score_model_5.pth", weights_only=True)
score_model.load_state_dict(state_dict)


def init():
    train_dataset = MatchDataset(f"{dataset_path}/train_parts.json", dataset_path)
    test_dataset = MatchDataset(f"{dataset_path}/test_parts.json", dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


# def jit_pt_test():
#     train_loader, test_loader = init()
#     model = torch.jit.load("checkpoints/gcn_layer_model_quantized.pt")
#     model.eval()
#     total_acc = 0
#     with torch.no_grad():
#         for ego_preds, cav_preds, K, gt in tqdm(test_loader):
#             K = K[0]
#             n1, n2 = ego_preds.shape[1], cav_preds.shape[1]
#             output = gcn_batch_match(K, n1, n2, model, score_model)
#             acc = (pygm.hungarian(output) * gt).sum() / gt.sum()
#             total_acc += acc
#     print(f"total acc: {total_acc/len(test_loader)}")


def pth_test():
    train_loader, test_loader = init()
    model = GCN_Net((32, 32, 32))
    state_dict = torch.load("checkpoints/gcn_layer_model_5.pth", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    total_acc = 0
    with torch.no_grad():
        for ego_preds, cav_preds, K, gt in tqdm(test_loader):
            K = K[0]
            n1, n2 = ego_preds.shape[1], cav_preds.shape[1]
            output = gcn_batch_match(K, n1, n2, model, score_model)
            acc = (pygm.hungarian(output) * gt).sum() / gt.sum()
            total_acc += acc
    print(f"total acc: {total_acc/len(test_loader)}")


def jit_pt_quantized_test():
    train_loader, test_loader = init()
    model = torch.jit.load("checkpoints/gcn_layer_model_quantized.pt")
    model.eval()

    total_acc = 0
    with torch.no_grad():
        for ego_preds, cav_preds, K, gt in tqdm(test_loader):
            K = K[0]
            n1, n2 = ego_preds.shape[1], cav_preds.shape[1]
            output = gcn_batch_match(K, n1, n2, model, score_model)
            acc = (pygm.hungarian(output) * gt).sum() / gt.sum()
            total_acc += acc
    print(f"total acc: {total_acc/len(test_loader)}")


if __name__ == "__main__":
    print("pth test")
    pth_test()
    print("jit_pt_quantized_test")
    jit_pt_quantized_test()
