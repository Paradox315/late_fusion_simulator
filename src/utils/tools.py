import math
import time
from functools import wraps

import numpy as np
from scipy.spatial.distance import cdist
from shapely import Polygon
from shapely.affinity import rotate


def sector(center, start_angle, end_angle, radius, steps=200):
    def polar_point(origin_point, angle, distance):
        return [
            origin_point.x + math.sin(angle) * distance,
            origin_point.y + math.cos(angle) * distance,
        ]

    if start_angle > end_angle:
        start_angle = start_angle - 2 * math.pi
    else:
        pass
    step_angle_width = (end_angle - start_angle) / steps
    sector_width = end_angle - start_angle
    segment_vertices = []
    start_angle = math.pi / 2 - end_angle
    end_angle = math.pi / 2 - start_angle

    segment_vertices.append(polar_point(center, 0, 0))
    segment_vertices.append(polar_point(center, start_angle, radius))

    for z in range(1, steps):
        segment_vertices.append(
            (polar_point(center, start_angle + z * step_angle_width, radius))
        )
    segment_vertices.append(polar_point(center, start_angle + sector_width, radius))
    segment_vertices.append(polar_point(center, 0, 0))
    return Polygon(segment_vertices)


def centers_to_boxes(centers: np.ndarray):
    """
    将中心表示法转换为框表示法

    参数:
    - centers: 中心表示法，表示为一个四元组(xc, yc, w, h, theta)

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


def center_to_coord(centers: np.ndarray):
    """
    :param centers: (x_center, y_center, width, height, theta)
    :return: the coordinates of the four corners of the rectangle
    """
    assert centers.shape == (5,)
    x, y, w, h, theta = centers
    from shapely.geometry import box
    from shapely.affinity import rotate

    rect = box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)
    rotated_rect = rotate(rect, theta, origin=(x, y), use_radians=True)
    return np.array(rotated_rect.exterior.coords.xy).T[:-1]


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
    box1:(x_center, y_center, width, height, theta)
    box2:(x_center, y_center, width, height, theta)
    """
    # 计算两个检测框的交集
    x1c, y1c, w1, h1, theta1 = boxes1
    x2c, y2c, w2, h2, theta2 = boxes2
    from shapely.geometry import Polygon  # 多边形

    poly1 = Polygon(
        [
            (x1c - w1 / 2, y1c - h1 / 2),
            (x1c - w1 / 2, y1c + h1 / 2),
            (x1c + w1 / 2, y1c + h1 / 2),
            (x1c + w1 / 2, y1c - h1 / 2),
        ]
    )
    poly1 = rotate(poly1, theta1, origin=(x1c, y1c), use_radians=True)
    poly2 = Polygon(
        [
            (x2c - w2 / 2, y2c - h2 / 2),
            (x2c - w2 / 2, y2c + h2 / 2),
            (x2c + w2 / 2, y2c + h2 / 2),
            (x2c + w2 / 2, y2c - h2 / 2),
        ]
    )
    poly2 = rotate(poly2, theta2, origin=(x2c, y2c), use_radians=True)
    if not poly1.intersects(poly2):
        return 0
    else:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        iou = inter_area / union_area
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
    :param preds1: preds1:(x_center, y_center, width, height, theta, class_probs, attr_probs)
    :param preds2: preds2:(x_center, y_center, width, height, theta, class_probs, attr_probs)
    :return:
    """
    # 计算两个检测框的IoU
    iou = compute_iou(preds1[1:6], preds2[1:6])
    # 计算两个检测框的属性信息的Hellinger距离
    helling = compute_helling(preds1[7:], preds2[7:])
    # 计算联合距离
    joint_dist = 0.7 * iou + 0.3 * helling
    return joint_dist


def node_affinity_fn(preds1, preds2):
    """
    :param preds1: preds1:(x_center, y_center, width, height, theta, class_probs, attr_probs)
    :param preds2: preds2:(x_center, y_center, width, height, theta, class_probs, attr_probs)
    :return:
    """
    if len(preds1.shape) == 3:
        preds1 = preds1[0]
        preds2 = preds2[0]
    affinity_mat = cdist(preds1, preds2, metric=compute_joint_dist)
    return np.expand_dims(affinity_mat, axis=0)
