import math
import random

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class GTBoxGenerator:
    def __init__(self, x, y, n, target_shape, co_visible=0, co_proportion=0.5):
        self.x = x
        self.y = y
        self.n = n
        self.target_shape = target_shape
        self.co_visible = co_visible
        self.co_proportion = co_proportion
        self.objs = None

    def generate(self):
        """
        生成检测框

        返回值:
        - boxes: 生成的检测框数组，每个检测框表示为一个四元组(x_start, y_start, x_end, y_end)
        """
        x, y = self.x, self.y
        co_space = ((0, x * self.co_proportion), (0, y))
        remaining_space = ((x * self.co_proportion, x), (0, y))
        co_objs = self._generate_objects(co_space, self.co_visible)
        remaining_boxes = self._generate_objects(
            remaining_space, self.n - self.co_visible
        )
        self.objs = np.vstack((co_objs, remaining_boxes))

    def _generate_objects(self, space, n):
        x_min, x_max = space[0]
        y_min, y_max = space[1]
        centers = np.random.rand(n, 2)
        shapes = [random.choice(self.target_shape) for _ in range(n)]

        centers = np.hstack((centers, np.array(shapes).reshape(-1, 2)))

        diag_len = max(map(lambda x: math.dist(x[0], x[1]), self.target_shape))
        x_min, x_max = x_min + diag_len / 2, x_max - diag_len / 2
        y_min, y_max = y_min + diag_len / 2, y_max - diag_len / 2
        # 保证生成的检测框不会超出边界，x_min<=x<=x_max, y_min<=y<=y_max
        centers[:, 0] = centers[:, 0] * (x_max - x_min) + x_min
        centers[:, 1] = centers[:, 1] * (y_max - y_min) + y_min
        return centers


class DetectionGenerator:
    def __init__(self, trues: GTBoxGenerator, noise_pos, noise_shape, cavs=1):
        self.trues = trues
        self.noise_pos = noise_pos
        self.noise_shape = noise_shape
        self.cavs = cavs

    def generate_preds(self, boxes):
        object_shapes = np.array(self.trues.target_shape)
        # 添加噪声
        noise_pos = np.random.randn(*boxes[:, :2].shape) * self.noise_pos
        noise_shape = np.random.randn(*boxes[:, 2:].shape) * self.noise_shape
        pred_boxes = boxes + np.hstack((noise_pos, noise_shape))
        probs = np.zeros((pred_boxes.shape[0], len(object_shapes)))
        # 添加类别概率分布
        for i, object_shape in enumerate(object_shapes):
            close_idxs = np.isclose(
                pred_boxes[:, 2:], object_shape, atol=self.noise_shape * 10
            ).all(axis=1)
            probs[close_idxs, i] = np.random.uniform(0.7, 1, size=close_idxs.sum())
        for prob in probs:
            no_assign_prob = 1 - np.sum(prob)
            if no_assign_prob:
                w = np.random.rand(len(prob[prob == 0]))
                w /= w.sum()
                prob[prob == 0] = w * no_assign_prob
        cls = probs.argmax(axis=1)
        # 添加协同检测(x, y, w, h, cls, prob)
        return np.hstack((pred_boxes, cls[:, np.newaxis], probs))

    def generate(self):
        co_space, ego_space, cav_space = self._generate_space()
        trues = self.trues.objs

        # 生成协同检测
        def isin_space(boxes, space):
            condition1 = np.logical_and(
                boxes[:, 0] >= space[0, 0], boxes[:, 0] < space[0, 1]
            )
            condition2 = np.logical_and(
                boxes[:, 1] >= space[1, 0], boxes[:, 1] < space[1, 1]
            )
            return condition1 & condition2

        co_boxes = trues[isin_space(trues, co_space)]
        ego_boxes = co_boxes.copy()
        cavs_boxes = [co_boxes.copy() for _ in range(self.cavs)]
        ego_remaining = trues[isin_space(trues, ego_space)]
        cavs_remaining = [
            trues[isin_space(trues, cav_space[i])] for i in range(self.cavs)
        ]
        ego_boxes = np.vstack((ego_boxes, ego_remaining))
        for i in range(self.cavs):
            cavs_boxes[i] = np.vstack((cavs_boxes[i], cavs_remaining[i]))

        self.ego_preds = self.generate_preds(ego_boxes)
        self.cavs_preds = [self.generate_preds(cav_boxes) for cav_boxes in cavs_boxes]

    def _generate_space(self):
        x, y = self.trues.x, self.trues.y
        co_space = np.array(((0, x * self.trues.co_proportion), (0, y)))
        ego_space = np.array(((x * self.trues.co_proportion, x), (0, y * 0.4)))
        cav_space = np.zeros((self.cavs, 2, 2))
        cav_space[:, 0] = np.array((x * self.trues.co_proportion, x))
        intervals = np.linspace(0.4 * y, y, num=self.cavs + 1)
        intervals = sliding_window_view(intervals, 2).reshape(-1, 2)
        cav_space[:, 1] = intervals
        return co_space, ego_space, cav_space
