import math

import numpy as np
import torch
from torch import Tensor


def ctx_base_matching(
    ego_preds,
    cav_preds,
    sigma1=math.radians(3),
    sigma2=5,
    absolute_dis_lim=20,
):
    algo = CBM(sigma1, sigma2, absolute_dis_lim)
    matching = algo(ego_preds, cav_preds)
    return torch.unbind(matching, dim=1)


class CBM:
    # All the transform are defined in right hand coordinates

    def __init__(self, sigma1=math.radians(3), sigma2=3, absolute_dis_lim=20):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.absolute_dis_lim = absolute_dis_lim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        # Input  Ego: torch.Tensor, Cav: torch.Tensor, transform: torch.Tensor
        args = self.check_numpy_to_torch(args)

        ego_preds, cav_preds = args[0], args[1]
        ego_preds, cav_preds = ego_preds.to(self.device), cav_preds.to(self.device)

        # Construct local context
        ctx1, ctx2 = self.build_context(ego_preds), self.build_context(cav_preds)
        self.m, self.n = len(ctx1), len(ctx2)

        # Local matching
        match = self.local_match(ctx1, ctx2)

        # Global matching
        match_result = self.global_match(match, match, ego_preds, cav_preds)

        # Convert matching matrix to the form [[i,j]]
        m = torch.where(match_result > 0)
        matching = torch.hstack((m[0].reshape(-1, 1), m[1].reshape(-1, 1)))

        return matching

    def check_numpy_to_torch(self, args: tuple) -> tuple:
        args_ = ()
        for i, j in enumerate(args):
            if isinstance(j, np.ndarray):
                args_ += (torch.tensor(j, dtype=torch.float32).to(self.device),)
            else:
                args_ += (j.to(self.device),)
        return args_

    def global_match(
        self,
        match: torch.Tensor,
        global_match: torch.Tensor,
        ego: torch.Tensor,
        cav: torch.Tensor,
    ) -> torch.Tensor:
        m, n = self.m, self.n

        # Update the match to ensure each element is matched at most once
        match.clamp_(max=1)

        # Compute the indices of onlooker values
        onlooker_indices = (
            match.sum(dim=1).unsqueeze_(1) + match.sum(dim=2).unsqueeze_(-1) == 2
        ) & (match == 1)
        graph_idx, row_idx, col_idx = torch.where(onlooker_indices)

        # Compute unique indices
        indices = torch.stack([row_idx * n + col_idx, graph_idx]).T
        unique_indices = torch.unique(indices, dim=0)

        # Compute global consensus
        global_consensus = torch.zeros_like(match)
        global_consensus[unique_indices[:, 1]] = torch.sum(
            global_match[unique_indices[:, 0]], dim=0
        )
        global_consensus = global_consensus * (match - onlooker_indices.float()) >= 1

        match[global_consensus] = 1

        # Compute the relative distance and store in dist_mat
        ego_, cav_ = ego.repeat(m * n, 1, 1), cav.repeat(m * n, 1, 1)
        match_indices = torch.where(match)
        rel_dist = (
            ego_[match_indices[0], match_indices[1], 0:2]
            - cav_[match_indices[0], match_indices[2], 0:2]
        )
        dist = torch.norm(rel_dist, dim=1)

        dist_mat = torch.zeros_like(match).to(self.device)
        dist_mat[match_indices] = dist

        # Compute normalized distance and update match where the distance exceeds threshold
        normalized_dist = (
            torch.mean(dist_mat, dim=(1, 2)) * m * n / onlooker_indices.sum(dim=(1, 2))
        )
        match[normalized_dist > self.absolute_dis_lim] = 0

        # Select the match with the maximum count
        max_count_idx = torch.argmax(match.sum(dim=(1, 2)))
        return match[max_count_idx]

    def local_match(self, ctx1: torch.Tensor, ctx2: torch.Tensor) -> Tensor:
        # Set constants
        m, n = self.m, self.n
        sigma1, sigma2 = self.sigma1, self.sigma2

        # Initialize an empty tensor on the device for Match
        match = torch.zeros((m * n, m, n), device=self.device)

        # Expand and reshape context matrices for further computation
        ctx_p = ctx1.unsqueeze(dim=1).expand(-1, n, -1, -1).reshape(m * n, 2, m)
        ctx_q = ctx2.unsqueeze(dim=0).expand(m, -1, -1, -1).reshape(m * n, 2, n)

        # Compute azimuthal distance and constrain its maximum value to 1
        inner_product = torch.matmul(ctx_p.transpose(1, 2), ctx_q)
        norm_product = torch.norm(ctx_p, dim=1).reshape(m * n, -1, 1) * torch.norm(
            ctx_q, dim=1, keepdim=True
        )
        azu_dist = torch.clamp(abs(inner_product / norm_product), max=1)

        # Compute score1 and get its qualified indices
        score1 = torch.acos(azu_dist) / sigma1
        qualified_indices = torch.where(score1 <= 1)

        # Compute 2-norm distance and get its qualified indices
        pairwise_diff = (
            ctx_p[qualified_indices[0], :, qualified_indices[1]]
            - ctx_q[qualified_indices[0], :, qualified_indices[2]]
        )
        score2 = torch.norm(pairwise_diff, dim=1, p=1)
        qualified_indices_for_score2 = torch.where(score2 <= sigma2)[0]

        # Get indices for m and n
        m_indices, n_indices = (
            torch.floor(qualified_indices[0] / n).type(torch.int64),
            qualified_indices[0] % n,
        )

        # Update the match tensor
        match[qualified_indices[0], m_indices, n_indices] = 1
        if len(qualified_indices_for_score2) > 0:
            match[
                qualified_indices[0][qualified_indices_for_score2],
                qualified_indices[1][qualified_indices_for_score2],
                qualified_indices[2][qualified_indices_for_score2],
            ] = 1

        return match

    def build_context(self, ego: torch.Tensor) -> torch.Tensor:
        # construct local context
        # Input: m x 7 tensor: (id, x, y, w, h, heading, cls, probs)
        ego_pos = ego[:, [1, 2, 5]]
        ego_relative_pos = (
            ego_pos[:, :-1] - ego_pos[:, 0:2].reshape(ego_pos.shape[0], 1, -1)
        ).to(self.device)

        # 使用角度创建旋转矩阵
        theta = ego_pos[:, -1]
        cos_theta, sin_theta = theta.cos(), theta.sin()

        # 使用[:,:,None]增加一个维度
        # [None, :, None]生成对角线元素，然后减去不需要的元素生成旋转矩阵
        rotation_mat = torch.diag_embed(
            torch.stack([cos_theta, sin_theta], dim=-1)
        )  # shape: (n, 2, 2)
        rotation_mat[:, 0, 1] -= sin_theta
        rotation_mat[:, 1, 0] += sin_theta
        rotation_mat = rotation_mat.to(self.device)

        # 计算新的相对位置并变换维度,合并简化为一行代码
        return torch.matmul(ego_relative_pos, rotation_mat).transpose(
            1, 2
        )  # shape: (n, 2, n)
