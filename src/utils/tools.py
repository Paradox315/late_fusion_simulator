import math
import time
from functools import wraps
import pygmtools as pygm
import numpy as np
import torch
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
    :param preds1: (id, x_center, y_center, width, height, theta, cls, attr_probs)
    :param preds2: (id, x_center, y_center, width, height, theta, cls, attr_probs)
    :return:
    """
    # 计算两个检测框的IoU
    iou = compute_iou(preds1[1:6], preds2[1:6])
    # 计算两个检测框的属性信息的Hellinger距离
    helling = compute_helling(preds1[7:], preds2[7:])
    # 计算联合距离
    joint_dist = 0.7 * iou + 0.3 * helling
    return joint_dist


def build_graph(preds: torch.Tensor):
    """
    :param preds: (id, x_center, y_center, width, height, theta, cls, attr_probs)
    :return: graph: shape(n,n): the graph of agents
    """
    points = preds[:, 1:3]
    cls = preds[:, 6]
    angles = preds[:, 5]
    dist_mat = torch.cdist(points, points, p=2)
    theta_mat = torch.abs(angles.unsqueeze(1) - angles.unsqueeze(0))
    cls_from = cls.unsqueeze(1).repeat(1, cls.shape[0])
    cls_to = cls.unsqueeze(0).repeat(cls.shape[0], 1)
    return torch.stack([dist_mat, theta_mat, cls_from, cls_to], dim=-1)


def build_conn_edge(graph: torch.Tensor):
    """
    :param graph: shape(n,n,features_len): the graph of agents
    :return: conn: shape(m,2): the connection matrix of agents
             edge: shape(m,2,features_len): the edge matrix of agents
    """
    from_, to_ = torch.where(graph[:, :, 0] != 0)
    conn = torch.stack([from_, to_], dim=0).t()
    edge = graph[conn[:, 0], conn[:, 1]]
    return conn, edge


def node_affinity_fn(
    preds1: torch.Tensor, preds2: torch.Tensor, lamda1=0.5, lamda2=0.1
):
    """
    :param preds1: preds1:(id, x_center, y_center, width, height, theta, cls, attr_probs)
    :param preds2: preds2:(id, x_center, y_center, width, height, theta, cls, attr_probs)
    :return:
    """
    if len(preds1.shape) == 3:
        preds1 = preds1[0]
        preds2 = preds2[0]
    cls1, cls2 = preds1[:, 6].bool(), preds2[:, 6].bool()
    shape1, shape2 = preds1[:, 3:5], preds2[:, 3:5]
    pos1, pos2 = preds1[:, 1:3], preds2[:, 1:3]
    conf1, conf2 = preds1[:, 7:], preds2[:, 7:]
    # check if the two nodes are of the same class
    affinity1 = cls1.view(-1, 1) == cls2.view(1, -1)
    # calculate the shape affinity
    affinity2 = torch.exp(-lamda1 * torch.cdist(shape1, shape2, p=2))
    # calculate the position affinity
    affinity3 = torch.exp(-lamda2 * torch.cdist(pos1, pos2, p=2))
    # calculate the confidence affinity
    affinity4 = torch.sum(torch.sqrt(conf1.unsqueeze(1) * conf2.unsqueeze(0)), dim=-1)
    # calculate the joint affinity
    mu1, mu2 = 0.5, 0.5
    affinity = affinity1 * affinity4 * (mu1 * affinity2 + mu2 * affinity3)
    return affinity[None]


def edge_affinity_fn(edges1, edges2, lamda1=0.5, lamda2=0.1):
    if len(edges1.shape) == 3:
        edges1 = edges1[0]
        edges2 = edges2[0]
    cls_edge1, cls_edge2 = edges1[:, 2:].int(), edges2[:, 2:].int()
    dist1, dist2 = edges1[:, 0], edges2[:, 0]
    angle1, angle2 = edges1[:, 1], edges2[:, 1]

    # check if the two edges are of the same class
    def compare_tensors(tensor1, tensor2):
        tensor1_exp = tensor1.unsqueeze(1).expand(-1, tensor2.size(0), -1)
        tensor2_exp = tensor2.unsqueeze(0).expand(tensor1.size(0), -1, -1)
        return torch.eq(tensor1_exp, tensor2_exp).all(dim=-1)

    affinity1 = compare_tensors(cls_edge1, cls_edge2)
    # calculate the distance affinity
    affinity2 = torch.exp(-lamda1 * (dist1.view(-1, 1) - dist2.view(1, -1)) ** 2)
    # calculate the angle affinity
    affinity3 = torch.exp(
        -lamda2
        * torch.abs(torch.sin(angle1.view(-1, 1)) - torch.sin(angle2.view(1, -1)))
    )
    mu1, mu2 = 0.5, 0.5
    affinity = affinity1 * (mu1 * affinity2 + mu2 * affinity3)
    return affinity[None]
