import unittest
import numpy as np
import torch
from torch import cdist
import pygmtools as pygm

# pygm.set_backend("pytorch")
# torch.manual_seed(1)


class MyTestCase(unittest.TestCase):
    def test_hungarian(self):
        ego_preds = torch.rand(1, 10, 2) * 10
        cavs_preds = torch.rand(3, 8, 2) * 10

        dist_mat = cdist(ego_preds, cavs_preds)
        x = pygm.hungarian(-dist_mat.numpy())
        x = torch.from_numpy(x)
        print((x * dist_mat).sum())

    def test_hungarian1(self):
        ego_preds = torch.rand(1, 5, 2) * 10
        cavs_shape = [3, 4, 5]
        max_cav_shape = max(cavs_shape)
        cavs_preds = torch.full(
            (len(cavs_shape), max_cav_shape, 2), fill_value=99999999.0
        )
        for i in range(len(cavs_shape)):
            cavs_preds[i, : cavs_shape[i], :] = torch.rand(cavs_shape[i], 2) * 10
        cavs_preds1 = [
            cavs_preds[i, : cavs_shape[i], :] for i in range(len(cavs_shape))
        ]
        dist_mat = cdist(ego_preds, cavs_preds)
        x = pygm.hungarian(-dist_mat.numpy())
        for i in range(len(cavs_shape)):
            x[i, :, cavs_shape[i] :] = 0
        x = torch.from_numpy(x)
        sum1 = (x * dist_mat).sum()
        print("matrix parallel sum", sum1)
        dist_mat_list = [
            cdist(ego_preds, cavs_preds1[i]) for i in range(len(cavs_shape))
        ]
        sum2 = 0
        for i in range(len(cavs_shape)):
            x1 = pygm.hungarian(-dist_mat_list[i].numpy())
            x1 = torch.from_numpy(x1)
            sum2 += (x1 * dist_mat_list[i]).sum()
        print("sequential sum", sum2)
        self.assertEqual(sum1, sum2)

    def test_hungarian2(self):
        cost_mat1 = np.random.rand(100, 90)
        x1 = pygm.hungarian(-cost_mat1)
        print(np.sum(x1 * cost_mat1))
        cost_mat2 = np.full((100, 100), fill_value=10000.0)
        cost_mat2[:, :90] = cost_mat1
        x2 = pygm.hungarian(-cost_mat2)
        x2[:, 90:] = 0
        print(np.sum(x2 * cost_mat2))
        self.assertAlmostEqual(np.sum(x1 * cost_mat1), np.sum(x2 * cost_mat2))


if __name__ == "__main__":
    unittest.main()
