import numpy as np


def rrwm(
    K: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    n1max,
    n2max,
    x0: np.ndarray,
    max_iter: int,
    sk_iter: int,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    numpy implementation of RRWM algorithm.
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(
        K, n1, n2, n1max, n2max, x0
    )
    # rescale the values in K
    d = K.sum(axis=2, keepdims=True)
    dmax = d.max(axis=1, keepdims=True)
    K = K / (dmax + d.min() * 1e-5)  # d.min() * 1e-5 for numerical reasons
    v = v0
    # v0 初始化为均匀分布1/n1n2
    for i in range(max_iter):
        # random walk
        # K根据v的分布进行加权
        v = np.matmul(K, v)
        last_v = v
        n = np.linalg.norm(v, ord=1, axis=1, keepdims=True)
        # 归一化
        v = v / n

        # reweighted jump
        s = v.reshape((batch_num, n2max, n1max)).transpose((0, 2, 1))
        # s是对v进行膨胀后的结果，beta是膨胀系数，目的是放大相似度接近的节点，使得算法更容易收敛
        s = beta * s / np.amax(s, axis=(1, 2), keepdims=True)
        v = (
            alpha
            * sinkhorn(s, n1, n2, max_iter=sk_iter, batched_operation=True)
            .transpose((0, 2, 1))
            .reshape((batch_num, n1n2, 1))
            + (1 - alpha) * v
        )  # alpha的比例进行随机跳跃，(1-alpha)的比例使用随机游走，sinkhorn算法的目的是归一化s矩阵
        n = np.linalg.norm(v, ord=1, axis=1, keepdims=True)
        v = np.matmul(v, 1 / n)

        if np.linalg.norm((v - last_v).squeeze(axis=-1), ord="fro") < 1e-5:
            break

    return v.reshape((batch_num, n2max, n1max)).transpose((0, 2, 1))
