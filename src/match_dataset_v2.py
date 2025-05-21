import json
import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MatchDataset(Dataset):
    def __init__(
        self, dataset_path, max_size=32, preload=False, cache_size=100, split="train"
    ):
        # 加载文件列表
        self.file_list = json.load(
            open(os.path.join(dataset_path, f"{split}_parts.json"))
        )
        self.dataset_path = dataset_path
        self.file_paths = [os.path.join(dataset_path, file) for file in self.file_list]
        self.max_size = max_size

        # 数据缓存设置
        self.preload = preload
        self.cache_size = min(cache_size, len(self.file_paths))
        self.cache = {}

        # 预加载部分数据
        if self.preload:
            print(f"预加载 {self.cache_size} 个样本到内存...")
            indices = np.random.choice(
                len(self.file_paths), self.cache_size, replace=False
            )
            for idx in indices:
                self.cache[idx] = self._load_data(idx)

    def __len__(self):
        return len(self.file_paths)

    def _load_data(self, idx):
        """从文件中加载数据"""
        try:
            with open(self.file_paths[idx], "rb") as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"加载文件 {self.file_paths[idx]} 时出错: {e}")
            # 返回空占位符数据
            return [
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
            ]

    def __getitem__(self, idx):
        """获取单个样本并预处理"""
        # 尝试从缓存获取
        if idx in self.cache:
            data = self.cache[idx]
        else:
            data = self._load_data(idx)
            # 可选: 动态缓存管理
            if len(self.cache) < self.cache_size:
                self.cache[idx] = data

        # 解包数据
        ego_preds, cav_preds, K, match_gt, gt = data

        # 转换为张量
        ego_tensor = torch.tensor(ego_preds, dtype=torch.float32)
        cav_tensor = torch.tensor(cav_preds, dtype=torch.float32)

        # 处理匹配矩阵K和ground truth
        K_tensor = torch.tensor(K, dtype=torch.float32)
        match_gt_tensor = torch.tensor(match_gt, dtype=torch.float32)
        # 获取实际节点数
        n1, n2 = ego_tensor.shape[0], cav_tensor.shape[0]

        # 填充特征数据到最大尺寸
        ego_padded = torch.zeros((self.max_size, ego_tensor.shape[1]))
        cav_padded = torch.zeros((self.max_size, cav_tensor.shape[1]))

        # 填充有效数据
        ego_padded[:n1] = ego_tensor
        cav_padded[:n2] = cav_tensor

        # 填充K和gt矩阵到固定大小
        K_padded = torch.zeros((self.max_size**2, self.max_size**2))
        match_gt_padded = torch.zeros((self.max_size, self.max_size))

        # 填充实际数据
        n1 = min(n1, self.max_size)
        n2 = min(n2, self.max_size)
        K_padded[: n1 * n2, : n1 * n2] = K_tensor
        match_gt_padded[:n1, :n2] = match_gt_tensor

        # 返回填充数据和有效点数
        return ego_padded, cav_padded, K_padded, match_gt_padded, torch.tensor([n1, n2])


def create_data_loaders(args):
    """创建训练和测试数据加载器"""
    train_dataset = MatchDataset(
        args.dataset_path,
        max_size=args.max_size,
        preload=args.preload_data,
        cache_size=args.cache_size,
        split="train",
    )

    val_dataset = MatchDataset(
        args.dataset_path,
        max_size=args.max_size,
        split="validate",
    )

    test_dataset = MatchDataset(args.dataset_path, max_size=args.max_size, split="test")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
