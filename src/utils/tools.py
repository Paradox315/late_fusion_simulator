import time
from functools import wraps

import numpy as np


def centers_to_boxes(centers: np.ndarray):
    """
    将中心表示法转换为框表示法

    参数:
    - centers: 中心表示法，表示为一个四元组(xc, yc, w, h)

    返回值:
    - boxes: 框表示法，表示为一个四元组(x_min, y_min, x_max, y_max)
    """
    boxes = np.zeros_like(centers)
    boxes[:, 0] = centers[:, 0] - centers[:, 2] / 2
    boxes[:, 1] = centers[:, 1] - centers[:, 3] / 2
    boxes[:, 2] = centers[:, 0] + centers[:, 2] / 2
    boxes[:, 3] = centers[:, 1] + centers[:, 3] / 2
    return boxes


def boxes_to_centers(boxes: np.array):
    """
    将框表示法转换为中心表示法

    参数:
    - boxes: 框表示法，表示为一个四元组(x_min, y_min, x_max, y_max)

    返回值:
    - centers: 中心表示法，表示为一个四元组(xc, yc, w, h)
    """
    centers = np.zeros_like(boxes)
    centers[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    centers[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    centers[:, 2] = boxes[:, 2] - boxes[:, 0]
    centers[:, 3] = boxes[:, 3] - boxes[:, 1]
    return centers


def diff_boxes(boxes: np.ndarray, sub_boxes: np.ndarray):
    return np.array(
        [row for row in boxes if not any(np.isclose(row, sub_boxes).all(axis=1))]
    )


def func_timer(unit="ms"):
    def decorator(func):
        func.is_first_call = True

        @wraps(func)
        def wrapper(*args, **kwargs):
            if func.is_first_call:
                start_time = time.perf_counter_ns()  # 获取开始时间
                func.is_first_call = False
                result = func(*args, **kwargs)  # 执行函数
                end_time = time.perf_counter_ns()  # 获取结束时间
                func.is_first_call = True
                duration = end_time - start_time  # 计算函数运行时间
                time_fmt_dict = {
                    "ns": f"{duration:.2f}ns",
                    "us": f"{duration/1e3:.2f}us",
                    "ms": f"{duration/1e6:.2f}ms",
                    "s": f"{duration/1e9:.2f}s",
                }
                duration = time_fmt_dict[unit]
                print(f"{func.__name__} executed in {duration}")  # 打印运行时间信息
                return result
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def compute_iou(boxes1, boxes2):
    """
    box1:(x_center, y_center, width, height)
    box2:(x_center, y_center, width, height)
    """
    # 计算两个检测框的交集
    x1c, y1c, w1, h1 = boxes1
    x2c, y2c, w2, h2 = boxes2
    x1 = x1c - w1 / 2
    y1 = y1c - h1 / 2
    x2 = x2c - w2 / 2
    y2 = y2c - h2 / 2
    x1 = max(x1, x2)
    y1 = max(y1, y2)
    x2 = min(x1c + w1 / 2, x2c + w2 / 2)
    y2 = min(y1c + h1 / 2, y2c + h2 / 2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = w1 * h1 + w2 * h2 - intersection
    iou = intersection / union

    return iou


def compute_helling(probs1, probs2):
    """
    probs1: shape (N,)-N个类别的属性概率
    probs2: shape (N,)-N个类别的属性概率
    """
    # 计算Hellinger距离
    distance = np.sum(np.multiply(np.sqrt(probs1), np.sqrt(probs2)))
    return distance


def compute_joint_dist(preds1, preds2):
    """
    preds1: (id, x_center, y_center, width, height, cls, probs)
    preds2: (id, x_center, y_center, width, height, cls, probs)
    """
    # 计算两个检测框的IoU
    iou = compute_iou(preds1[1:5], preds2[1:5])
    # 计算两个检测框的属性信息的Hellinger距离
    helling = compute_helling(preds1[6:], preds2[6:])
    # 计算联合距离
    joint_dist = 0.7 * iou + 0.3 * helling
    return joint_dist
