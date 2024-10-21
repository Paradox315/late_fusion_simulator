import numpy as np
import torch
import pygmtools as pygm
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import stats

from src.utils.tools import (
    compute_joint_dist,
    node_affinity_fn,
    func_timer,
    build_graph,
    build_conn_edge,
    edge_affinity_fn,
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


@func_timer()
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


@func_timer()
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
        with torch.set_grad_enabled(False):
            X = pygm.ngm(K.float(), n1, n2, pretrain="voc")
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
