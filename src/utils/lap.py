from logging import raiseExceptions

import numpy as np
import pygmtools as pygm
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import stats

from src.utils.tools import (
    build_conn_edge,
    build_graph,
    compute_joint_dist,
    edge_affinity_fn,
    func_timer,
    node_affinity_fn,
)

pygm.set_backend("pytorch")
history = []


def expand_to_square(matrix: torch.Tensor, pad_value=0) -> torch.Tensor:
    """
    :param matrix: the matrix to be expanded
    :param pad_value: the value to be filled in the new matrix
    :return:
    """
    rows, cols = matrix.shape
    if rows == cols:
        return matrix

    # 创建新的方形矩阵
    size = max(rows, cols)
    new_matrix = torch.full((size, size), pad_value)

    # 拷贝原始矩阵的值
    new_matrix[:rows, :cols] = matrix
    return new_matrix


@func_timer(history=history)
def hungarian_match(ego_preds: np.ndarray, cav_preds: np.ndarray, threshold=0.5):
    """
    :param ego_preds: ego predictions n*(id, x, y, w, h, theta, cls, probs)
    :param cav_preds: cav predictions m*(id, x, y, w, h, theta, cls, probs)
    :param threshold:
    :return: ego_ids, cav_ids
    """

    dist_mat = cdist(ego_preds, cav_preds, metric=compute_joint_dist)
    ego_ids, cav_ids = linear_sum_assignment(dist_mat, maximize=True)
    matching_indices = np.where(dist_mat[ego_ids, cav_ids] > threshold)
    ego_ids = ego_ids[matching_indices]
    cav_ids = cav_ids[matching_indices]
    return ego_ids, cav_ids


@func_timer(history=history)
def auction_match(ego_preds: np.ndarray, cav_preds: np.ndarray, threshold=0.5):
    """
    :param ego_preds: ego predictions n*(id, x, y, w, h, theta, cls, probs)
    :param cav_preds: cav predictions m*(id, x, y, w, h, theta, cls, probs)
    :param threshold:
    :return: ego_ids, cav_ids
    """
    dist_mat = torch.from_numpy(cdist(ego_preds, cav_preds, metric=compute_joint_dist))
    ego_ids, cav_ids = auction_lap(dist_mat, maximize=True)
    matching_indices = np.where(dist_mat[ego_ids, cav_ids] > threshold)
    ego_ids = ego_ids[matching_indices]
    cav_ids = cav_ids[matching_indices]
    return ego_ids, cav_ids


def auction_lap(cost_matrix: torch.Tensor, eps=None, maximize=True):
    """
    :param cost_matrix: shape(n,n): the cost matrix
    :param eps: eps is the bidding increment
    :param maximize: if True, maximize the profit
    """
    eps = 1 / cost_matrix.shape[0] if eps is None else eps

    num_workers, num_tasks = cost_matrix.shape

    # transpose if num_workers > num_tasks
    if num_workers > num_tasks:
        a, b = auction_lap(cost_matrix.T, maximize=maximize, eps=eps)
        return b, a

    # solve the minimization problem
    if not maximize:
        cost_matrix = -cost_matrix

    # expand to square matrix
    if num_workers != num_tasks:
        cost_matrix = expand_to_square(cost_matrix, pad_value=cost_matrix.max() + 1)
    # --
    # Init

    cost = torch.zeros((1, num_tasks), dtype=torch.float64)
    bidder_assignment = torch.full((num_workers,), fill_value=-1, dtype=torch.long)
    bids = torch.zeros(cost_matrix.shape, dtype=torch.float64)

    while (bidder_assignment == -1).any():
        # --
        # Bidding
        unassigned = torch.where(bidder_assignment == -1)[0]
        value = cost_matrix[unassigned] - cost
        top_value, top_idx = value.topk(2, dim=1)

        first_idx = top_idx[:, 0]  # find the task index with the highest value
        first_value, second_value = (
            top_value[:, 0],
            top_value[:, 1],
        )  # find the highest value and the second highest value

        bid_increments = first_value - second_value + eps  # calculate the bid increment

        bids_ = bids[unassigned]  # find the bids of the unassigned workers
        bids_.scatter_(
            dim=1,
            index=first_idx.contiguous().view(-1, 1),
            src=bid_increments.view(-1, 1),
        )  # update the bids of the unassigned workers

        # --
        # Assignment
        have_bidder = (
            (bids_ > 0).int().sum(dim=0).nonzero()
        )  # find the tasks that have bidders

        high_bids, high_bidders = bids_[:, have_bidder].max(
            dim=0
        )  # find the highest bids and the corresponding bidders
        high_bidders = unassigned[high_bidders.squeeze()]  # find the bidder indices
        cost[:, have_bidder] += high_bids  # update the task costs

        bidder_assignment[
            (bidder_assignment.view(-1, 1) == have_bidder.view(1, -1)).any(dim=1)
        ] = -1  # if the task has a worker, then the worker is unassigned
        bidder_assignment[high_bidders] = have_bidder.squeeze()  # assign the task

    # if num_workers!= num_tasks, then we need to remove the extra workers
    if num_workers != num_tasks:
        bidder_assignment = bidder_assignment[:num_workers]

    return np.arange(len(bidder_assignment)), bidder_assignment.numpy()


# TODO deprecated
def auction(cost_matrix, maximize=False, eps=1e-3):
    """
    :param cost_matrix: cost matrix
    :param maximize: if True, maximize the profit
    :param eps: bidding increment
    :return:
    """
    num_workers, num_tasks = cost_matrix.shape
    if num_workers > num_tasks:
        a, b = auction(cost_matrix.T, maximize=maximize, eps=eps)
        return b, a
    if not maximize:
        cost_matrix = -cost_matrix
        eps = -eps

    cost_matrix = expand_to_square(cost_matrix, pad_value=cost_matrix.max() + 1)

    # 初始化工人和任务的价格向量
    n = max(num_workers, num_tasks)
    worker_price = np.zeros(n)
    task_price = np.zeros(n)

    # 初始化工人和任务的匹配向量
    worker_assignment = -np.ones(n)
    task_assignment = -np.ones(n)

    # 执行算法
    while -1 in task_assignment:
        for worker in range(n):
            if worker_assignment[worker] == -1:
                # 提出出价
                values = cost_matrix[worker, :] - task_price
                task = np.argmax(values)

                # 如果任务已有工人，则需要更新匹配
                if task_assignment[task] != -1:
                    worker_assignment[int(task_assignment[task])] = -1

                # 更新匹配和价格
                task_assignment[task] = worker
                worker_assignment[worker] = task
                worker_price[worker] += eps  # 使用更小的增加步长
                task_price[task] = cost_matrix[worker, task] - worker_price[worker]
    worker_assignment = worker_assignment[worker_assignment != -1]
    if num_workers != num_tasks:
        worker_assignment = worker_assignment[:num_workers]
    return np.arange(num_workers), worker_assignment.astype(np.int32)


@func_timer(history=history)
def graph_based_match(
    ego_preds, cav_preds, associate_func="auction", match_func="rrwm"
):
    """
    :param associate_func: the associate function, hungarian or auction
    :param match_func: the matching function: rrwn, ipfp, sm, ngm
    :param ego_preds: ego predictions, n*(id, x, y, w, h, theta, cls, probs)
    :param cav_preds: cav predictions, m*(id, x, y, w, h, theta, cls, probs)
    :return: ego_ids, cav_ids
    """
    match_func_dict = {
        "rrwm": pygm.rrwm,
        "ipfp": pygm.ipfp,
        "sm": pygm.sm,
        "ngm": pygm.ngm,
    }
    assert associate_func in ["hungarian", "auction"]
    assert match_func in match_func_dict.keys()
    K, n1, n2 = build_affinity_matrix(cav_preds, ego_preds)
    if match_func == "ngm":
        X = pygm.ngm(K.float(), n1, n2)
    else:
        X = match_func_dict[match_func](K, n1, n2)
    if associate_func == "hungarian":
        match = pygm.hungarian(X)  # 使用匈牙利算法进行匹配
        ego_ids, cav_ids = np.where(match == 1)
    else:
        ego_ids, cav_ids = auction_lap(X)
    dist = [
        np.linalg.norm(ego_preds[i][1:3] - cav_preds[j][1:3])
        for i, j in zip(ego_ids, cav_ids)
    ]
    affinities = stats.zscore(dist)
    # Create a boolean mask for values with abs(zscore) <= 2
    mask = np.abs(affinities) <= 2

    return ego_ids[mask], cav_ids[mask]


def gcn_graph_match(ego_preds, cav_preds, gcn_model, score_model):
    """
    :param ego_preds: ego predictions, n*(id, x, y, w, h, theta, cls, probs)
    :param cav_preds: cav predictions, m*(id, x, y, w, h, theta, cls, probs)
    :return: ego_ids, cav_ids
    """
    dist_mat = cdist(ego_preds, cav_preds, metric=compute_joint_dist)
    K, n1, n2 = build_affinity_matrix(cav_preds, ego_preds)
    output = gcn_batch_match(K, n1, n2, gcn_model, score_model)
    ego_ids, cav_ids = np.where(pygm.hungarian(output) == 1)
    return ego_ids, cav_ids
    # return hungarian_match(ego_preds, cav_preds)

    # dist = [
    #     np.linalg.norm(ego_preds[i][1:3] - cav_preds[j][1:3])
    #     for i, j in zip(ego_ids, cav_ids)
    # ]
    # affinities = stats.zscore(dist)
    # # Create a boolean mask for values with abs(zscore) <= 2
    # mask = np.abs(affinities) <= 2
    # return ego_ids[mask], cav_ids[mask]


def build_affinity_matrix(cav_preds: np.ndarray, ego_preds: np.ndarray):
    ego_preds, cav_preds = torch.from_numpy(ego_preds), torch.from_numpy(cav_preds)
    ego_graph, cav_graph = build_graph(ego_preds), build_graph(cav_preds)
    n1, n2 = torch.tensor([ego_graph.shape[0]]), torch.tensor([cav_graph.shape[0]])
    conn1, edge1 = build_conn_edge(ego_graph)
    conn2, edge2 = build_conn_edge(cav_graph)
    K = pygm.utils.build_aff_mat(
        ego_preds,
        edge1,
        conn1,
        cav_preds,
        edge2,
        conn2,
        n1,
        None,
        n2,
        None,
        edge_aff_fn=edge_affinity_fn,
        node_aff_fn=node_affinity_fn,
    )
    return K, n1, n2


def build_affinity_matrix_v2(
    node_aff_mat: torch.Tensor,
    edge_aff_mat: torch.Tensor,
    graph1_edges: torch.Tensor,
    graph2_edges: torch.Tensor,
) -> torch.Tensor:
    """构建二阶亲和矩阵

    Args:
        node_aff_mat: 节点相似度矩阵，形状 (num_nodes1, num_nodes2)
        edge_aff_mat: 边相似度矩阵，形状 (num_edges1, num_edges2)
        graph1_edges: 图1的边连接关系，形状 (num_edges1, 2)
        graph2_edges: 图2的边连接关系，形状 (num_edges2, 2)
    Returns:
        affinity_matrix: 二阶亲和矩阵，形状 (num_nodes1*num_nodes2, num_nodes1*num_nodes2)
    """
    assert node_aff_mat is not None or edge_aff_mat is not None
    device = edge_aff_mat.device if edge_aff_mat is not None else node_aff_mat.device
    dtype = edge_aff_mat.dtype if edge_aff_mat is not None else node_aff_mat.dtype
    num_nodes1, num_nodes2 = node_aff_mat.shape
    num_edges1, num_edges2 = edge_aff_mat.shape

    # 初始化二阶亲和矩阵K
    affinity_matrix = torch.zeros(
        num_nodes2, num_nodes1, num_nodes2, num_nodes1, dtype=dtype, device=device
    )

    # 处理边的亲和度
    if edge_aff_mat is not None:
        # 构建边的索引矩阵
        edge_indices = _build_edge_indices(
            graph1_edges[:num_edges1], graph2_edges[:num_edges2], num_edges1, num_edges2
        )
        # 填充边的亲和度值
        affinity_matrix[edge_indices] = edge_aff_mat[:num_edges1, :num_edges2].reshape(
            -1
        )

    # 重塑为方阵
    affinity_matrix = affinity_matrix.reshape(
        num_nodes2 * num_nodes1, num_nodes2 * num_nodes1
    )

    # 处理节点的亲和度
    if node_aff_mat is not None:
        diagonal = torch.diagonal(affinity_matrix)
        diagonal[:] = node_aff_mat.t().reshape(-1)

    return affinity_matrix


def _build_edge_indices(
    edges1: torch.Tensor, edges2: torch.Tensor, num_edges1: int, num_edges2: int
) -> tuple[torch.Tensor, ...]:
    """构建边的索引矩阵

    Args:
        edges1: 图1的边，形状 (num_edges1, 2)
        edges2: 图2的边，形状 (num_edges2, 2)
        num_edges1: 图1的边数
        num_edges2: 图2的边数

    Returns:
        edge_indices: 边索引元组 (start_g2, start_g1, end_g2, end_g1)
    """
    combined_edges = torch.cat(
        [edges1.repeat_interleave(num_edges2, dim=0), edges2.repeat(num_edges1, 1)],
        dim=1,
    )

    return (
        combined_edges[:, 2],  # start_g2
        combined_edges[:, 0],  # start_g1
        combined_edges[:, 3],  # end_g2
        combined_edges[:, 1],  # end_g1
    )


def ngm_match(K, n1, n2, network=None, sk_max_iter=20, sk_tau=0.05):
    # n1max, n2max = n1.max(), n2.max()
    # v0 = torch.diag(K[0]).reshape(1, K.shape[1], 1)
    # result = network(K, n1, n2, n1max, n2max, v0, sk_max_iter, sk_tau)
    result = pygm.ngm(K, n1, n2, network=network)
    return result


def gcn_match(K, n1, n2, network):
    return network(K[0], n1, n2).unsqueeze(0)


def gcn_match_v2(ego_preds, cav_preds, network):
    return network(ego_preds, cav_preds)


def gcn_batch_match(K, n1, n2, gcn_model, score_model) -> torch.Tensor:
    """
    :param K: shape (max_size^2,max_size^2)  # 邻接矩阵
    :return:
    """
    max_size = 32
    K_padded = torch.zeros((max_size**2, max_size**2))
    K_padded[: n1 * n2, : n1 * n2] = K

    A = (K_padded != 0).float()
    A = A / (A.sum(dim=1, keepdim=True) + 1e-8)
    W = A * K_padded
    x = torch.zeros((max_size**2, 32), device=K.device)
    x[:, 0] = torch.diagonal(K_padded)
    x = gcn_model(W, x)
    output = score_model(x, n1, n2)
    return output  # shape (n1, n2)


class ScoreLayer(nn.Module):
    def __init__(self, max_size=32):
        super(ScoreLayer, self).__init__()
        self.max_size = max_size
        # 分类器层
        self.classifier = nn.Linear(32, 1)

    def forward(self, x, n1, n2):
        """
        :param x: shape (max_size^2,max_size^2)  # 邻接矩阵
        :return:
        """
        # 分类器输出
        scores = self.classifier(x[: n1 * n2]).view(n2, n1).t()
        scores = torch.softmax(scores, dim=-1)
        return scores
