from copy import deepcopy
from typing import List, Tuple, Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.agent import Agent, SimAgent
from src.models.scenario import Scenario, SimScenario


def get_ground_truth(ego: Agent | SimAgent, cav: Agent | SimAgent, env: Scenario):
    return sum(
        1 for _, obj in env.objects.items() if ego.within(obj) and cav.within(obj)
    )


def filter_preds(
    preds: np.ndarray, ego: Agent | SimAgent, cav: Agent | SimAgent
) -> np.ndarray:
    subset = preds[:, 1:3]

    # 创建一个布尔数组表示每个预测是否在 ego 和 cav 的范围内
    within_ego = np.apply_along_axis(ego.within, 1, subset)
    within_cav = np.apply_along_axis(cav.within, 1, subset)

    # 使用布尔数组进行索引
    indices = np.where(within_ego & within_cav)
    return preds[indices]


class Frame:
    def __init__(self, env: Scenario | SimScenario, agents: List[Agent | SimAgent]):
        self.env = env
        self.agents = agents

    def __next__(self):
        agents = deepcopy(self.agents)
        for agent in agents:
            agent.move()
            x, y = agent.position
            if (
                x < 0
                or x > self.env.background.width
                or y < 0
                or y > self.env.background.height
            ):
                return None
        return Frame(self.env, agents)

    def visualize(self):
        # Logic to visualize the frame
        # 创建一个新的图形
        fig, ax = plt.subplots()
        # 在图形上画出环境
        self.env.visualize(ax)
        # 在图形上画出agent
        for agent in self.agents:
            agent.visualize(ax)

        # 显示图形
        plt.show()

    def predict(
        self,
        fuse_method: Callable[[np.ndarray, np.ndarray], Tuple],
        noisy_setting=None,
        threshold=0.6,
    ) -> Dict[str, Dict]:
        """
        :param fuse_method: object fusion method
        :param noisy_setting: object detection noise setting, including position noise, shape noise.
        :param threshold: threshold for object fusion
        :return: prediction results: {ego_id:{cav_id: {correct_preds, false_preds, ego_preds, cav_preds}}}
        """
        if noisy_setting is None:
            noisy_setting = {}
        # Logic to predict the next frame
        predict_results = {}
        for ego in self.agents:
            cav_agents = [agent for agent in self.agents if agent != ego]
            ego_predict_results = self.ego_predict(
                ego, cav_agents, fuse_method, noisy_setting
            )
            predict_results[f"ego{ego.aid}"] = ego_predict_results
        return predict_results

    # TODO: 使用矩阵并行化计算
    def ego_predict(
        self,
        ego_agent: Agent | SimAgent,
        cav_agents: List[Agent | SimAgent],
        fuse_method: Callable[[np.ndarray, np.ndarray], Tuple],
        noisy_setting=None,
    ):
        if noisy_setting is None:
            noisy_setting = {}
        # Logic to predict the next frame
        predict_results = {}
        ego_preds = ego_agent.predict(
            self.env, **noisy_setting
        )  # (id, x, y, w, h, cls, probs)
        if ego_preds is None:
            return predict_results
        for cav in cav_agents:
            if not ego_agent.within(cav):
                continue
            ground_truth = get_ground_truth(ego_agent, cav, self.env)
            if ground_truth == 0:
                continue
            cav_preds = cav.predict(self.env, **noisy_setting)
            if cav_preds is None:
                continue
            ego_preds = filter_preds(ego_preds, ego_agent, cav)
            cav_preds = filter_preds(cav_preds, ego_agent, cav)
            # Logic to fuse the predictions
            ego_ids, cav_ids = fuse_method(ego_preds, cav_preds)

            # Note: Assume ego_preds and cav_preds are 2D numpy arrays, and id is at column 0.
            TP = int(np.sum(ego_preds[ego_ids][:, 0] == cav_preds[cav_ids][:, 0]))
            FP = len(ego_ids) - TP
            FN = ground_truth - TP

            predict_results[f"cav{cav.aid}"] = {
                "accuracy": TP / (TP + FP + FN) if TP + FP + FN > 0 else 0,
                "precision": TP / (TP + FP) if TP + FP > 0 else 0,
                "recall": TP / (TP + FN) if TP + FN > 0 else 0,
                "correct_preds": TP,
                "false_preds": FP,
                "ground_truth": ground_truth,
                "ego_preds": len(ego_preds),
                "cav_preds": len(cav_preds),
            }
        return predict_results
