import numpy as np


def expand_to_square(matrix, pad_value=0):
    rows, cols = matrix.shape
    if rows == cols:
        return matrix

    # 创建新的方形矩阵
    size = max(rows, cols)
    new_matrix = np.full((size, size), pad_value)

    # 拷贝原始矩阵的值
    new_matrix[:rows, :cols] = matrix
    return new_matrix


def auction_algorithm(cost_matrix, maximize=False, eps=1e-3):
    num_workers, num_tasks = cost_matrix.shape
    if num_workers > num_tasks:
        a, b = auction_algorithm(cost_matrix.T, maximize=maximize, eps=eps)
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
