import collections
from copy import deepcopy
from typing import List, overload

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from src.models.agent import Agent
from src.models.environment import Environment
from src.utils.tools import compute_joint_dist


class Frame:
    def __init__(self, env: Environment, agents: List[Agent]):
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

    def predict(self, fuse_method, noisy_setting=None, threshold=0.5) -> dict:
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
                ego, cav_agents, fuse_method, noisy_setting, threshold
            )
            predict_results[f"ego{ego.aid}"] = ego_predict_results
        return predict_results

    def ego_predict(
        self,
        ego_agent: Agent,
        cav_agents: List[Agent],
        fuse_method,
        noisy_setting=None,
        threshold=0.5,
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
            cav_preds = cav.predict(self.env, **noisy_setting)
            if cav_preds is None:
                continue
            # Logic to fuse the predictions
            dist_mat = cdist(ego_preds, cav_preds, metric=compute_joint_dist)
            ego_ids, cav_ids = fuse_method(dist_mat, maximize=True)
            bools = dist_mat[ego_ids, cav_ids] > threshold
            ego_ids = ego_ids[bools]
            cav_ids = cav_ids[bools]
            correct_preds, false_preds = 0, 0
            for ego_id, cav_id in zip(ego_ids, cav_ids):
                ego_pred = ego_preds[ego_id]
                cav_pred = cav_preds[cav_id]
                # preds[0] is id
                if ego_pred[0] == cav_pred[0]:
                    correct_preds += 1
                else:
                    false_preds += 1
            predict_results[f"cav{cav.aid}"] = {
                "correct_preds": correct_preds,
                "false_preds": false_preds,
                "ego_preds": len(ego_preds),
                "cav_preds": len(cav_preds),
            }
        return predict_results
