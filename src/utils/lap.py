import numpy as np
import pygmtools as pygm
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from src.utils.tools import compute_joint_dist, node_affinity_fn


def expand_to_square(matrix, pad_value=0):
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
    new_matrix = np.full((size, size), pad_value)

    # 拷贝原始矩阵的值
    new_matrix[:rows, :cols] = matrix
    return new_matrix


# @func_timer()
def hungarian_match(ego_preds, cav_preds, threshold=0.5):
    dist_mat = cdist(ego_preds, cav_preds, metric=compute_joint_dist)
    ego_ids, cav_ids = linear_sum_assignment(dist_mat, maximize=True)
    matching_indices = np.where(dist_mat[ego_ids, cav_ids] > threshold)
    ego_ids = ego_ids[matching_indices]
    cav_ids = cav_ids[matching_indices]
    return ego_ids, cav_ids


# @func_timer()
def auction_match(ego_preds, cav_preds, threshold=0.5):
    dist_mat = cdist(ego_preds, cav_preds, metric=compute_joint_dist)
    ego_ids, cav_ids = auction(dist_mat, maximize=True)
    matching_indices = np.where(dist_mat[ego_ids, cav_ids] > threshold)
    ego_ids = ego_ids[matching_indices]
    cav_ids = cav_ids[matching_indices]
    return ego_ids, cav_ids


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


# @func_timer()
def graph_based_match(ego_preds, cav_preds):
    """
    :param ego_preds: ego predictions
    :param cav_preds: cav predictions
    :return:
    """
    ego_graph, cav_graph = build_graph(ego_preds), build_graph(cav_preds)
    n1, n2 = np.array([ego_graph.shape[0]]), np.array([cav_graph.shape[0]])
    conn1, edge1 = pygm.utils.dense_to_sparse(ego_graph)
    conn2, edge2 = pygm.utils.dense_to_sparse(cav_graph)
    import functools

    gaussian_aff = functools.partial(
        pygm.utils.gaussian_aff_fn, sigma=1.0
    )  # set affinity function
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
        edge_aff_fn=gaussian_aff,
        node_aff_fn=node_affinity_fn,
    )
    X = pygm.rrwm(K, n1, n2)  # X代表了G1中的每个节点与G2中的每个节点的匹配程度
    match = pygm.hungarian(X)  # 使用匈牙利算法进行匹配
    # Find where match equals 1
    ego_ids, cav_ids = np.where(match == 1)
    return ego_ids, cav_ids
    # dist = [
    #     np.linalg.norm(ego_preds[i][1:3] - cav_preds[j][1:3])
    #     for i, j in zip(ego_ids, cav_ids)
    # ]
    # affinities = stats.zscore(dist)
    # # Create a boolean mask for values with abs(zscore) <= 2
    # mask = np.abs(affinities) <= 1
    #
    # return ego_ids[mask], cav_ids[mask]


def build_graph(preds):
    """
    :param preds: shape(n,7): the predictions of agents, (x, y, w, h, theta, cls, prob)
    :return: graph: shape(n,n): the graph of agents
    """
    n = preds.shape[0]
    return cdist(preds, preds, metric="euclidean")
