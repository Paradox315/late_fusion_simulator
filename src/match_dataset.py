import json
import mmap
import pickle
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, pickle_files, dataset_path):
        # 假设pickle_files是一个包含多个pickle文件路径的列表
        pickle_files = json.load(open(pickle_files))
        self.data_files = [f"{dataset_path}/{file}" for file in pickle_files]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 可选：如果数据预处理非常耗时，可以在这里预先处理数据

    def __len__(self):
        # 返回数据集的大小
        return len(self.data_files)

    def __getitem__(self, index):
        # 根据索引加载数据
        with open(self.data_files[index], "rb") as file:
            mm = mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ)
            try:
                # 从内存映射对象中反序列化数据
                data = pickle.loads(mm)
            finally:
                # 关闭内存映射对象
                mm.close()
        # 对数据进行必要的预处理，例如类型转换、标准化等
        # 假设data是一个字典，包含4个矩阵
        return tuple(map(self.preprocess_matrix, data))

    def preprocess_matrix(self, matrix):
        # 这里实现矩阵的预处理逻辑
        # 例如，转换为torch.Tensor，标准化等
        return torch.tensor(matrix, dtype=torch.float32, device=self.device)


class MatchDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=cpu_count())  # 根据硬件调整线程数

    def __getitem__(self, index):
        # 使用线程池来并行加载数据
        future = self.executor.submit(super().__getitem__, index)
        return future.result()
