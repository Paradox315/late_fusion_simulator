from copy import deepcopy
from typing import List, Tuple, Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from src.models.agent import Agent, SimAgent
from src.models.scenario import Scenario, SimScenario
from src.utils.lap import hungarian_match, build_affinity_matrix


def get_ground_truth(ego: Agent | SimAgent, cav: Agent | SimAgent, env: Scenario):
    """
    :param ego: ego predictions
    :param cav: cav predictions
    :param env: environment
    :return: ground truth of the number of objects in the intersection of ego and cav
    """
    return sum(
        1 for _, obj in env.objects.items() if ego.within(obj) and cav.within(obj)
    )


def filter_preds(
    preds: np.ndarray, ego: Agent | SimAgent, cav: Agent | SimAgent
) -> np.ndarray:
    """
    :param preds: predictions
    :param ego:
    :param cav:
    :return: the subset of predictions that are within the intersection of ego and cav
    """
    subset = preds[:, 1:3]

    # 创建一个布尔数组表示每个预测是否在 ego 和 cav 的范围内
    within_ego = np.apply_along_axis(ego.within, 1, subset)
    within_cav = np.apply_along_axis(cav.within, 1, subset)

    # 使用布尔数组进行索引
    indices = np.where(within_ego & within_cav)
    return preds[indices]


class Frame:
    def __init__(
        self,
        env: Scenario | SimScenario,
        agents: List[Agent | SimAgent],
        noise_setting=None,
    ):
        """
        :param env: environment
        :param agents: agents in the environment
        """
        self.env = env
        self.agents = agents
        self.noise_setting = noise_setting if noise_setting is not None else {}
        self.cache = {
            agent.aid: agent.predict(env, **self.noise_setting) for agent in agents
        }

    def __next__(self):
        """
        :return: next frame
        """
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
        return Frame(self.env, agents, noise_setting=self.noise_setting)

    def visualize(self):
        # Logic to visualize the frame
        fig, ax = plt.subplots()
        # visualize the environment
        self.env.visualize(ax)
        # visualize the agents
        for agent in self.agents:
            agent.visualize(ax)

        plt.show()

    def predict(
        self,
        fuse_method: Callable[[np.ndarray, np.ndarray], Tuple],
    ) -> Dict[str, Dict]:
        """
        :param fuse_method: object fusion method
        :return: prediction results: {ego_id:{cav_id: {correct_preds, false_preds, ego_preds, cav_preds}}}
        """
        # Logic to predict the next frame
        predict_results = {}
        for ego in self.agents:
            cav_agents = [agent for agent in self.agents if agent != ego]
            ego_predict_results = self.ego_predict(ego, cav_agents, fuse_method)
            predict_results[f"ego{ego.aid}"] = ego_predict_results
        return predict_results

    def ego_predict(
        self,
        ego_agent: Agent | SimAgent,
        cav_agents: List[Agent | SimAgent],
        fuse_method: Callable[[np.ndarray, np.ndarray], Tuple],
    ):
        """
        :param ego_agent: the ego agent
        :param cav_agents: cav agents
        :param fuse_method: the method to fuse the predictions
        :return: the prediction results after fusing the predictions from the ego and cav agents
        """

        def get_preds(agent):
            if agent.aid in self.cache:
                return self.cache[agent.aid]
            return agent.predict(self.env, **self.noise_setting)

        # Logic to predict the next frame
        predict_results = {}
        ego_preds = get_preds(ego_agent)
        if ego_preds is None:
            return predict_results
        for cav in cav_agents:
            if not ego_agent.within(cav):
                continue
            ground_truth = get_ground_truth(ego_agent, cav, self.env)
            if ground_truth == 0:
                continue
            cav_preds = get_preds(cav)
            if cav_preds is None:
                continue
            ego_preds = filter_preds(ego_preds, ego_agent, cav)
            cav_preds = filter_preds(cav_preds, ego_agent, cav)
            # Logic to fuse the predictions
            if len(ego_preds) == 1 or len(cav_preds) == 1:
                ego_ids, cav_ids = hungarian_match(ego_preds, cav_preds)
            else:
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

    def generate_data(self):
        data_dict = []
        for ego in self.agents:
            cav_agents = [agent for agent in self.agents if agent != ego]
            ego_data = self.ego_generate_data(
                ego,
                cav_agents,
            )
            if ego_data:
                data_dict.extend(ego_data)
        return data_dict

    def ego_generate_data(
        self, ego_agent: Agent | SimAgent, cav_agents: List[Agent | SimAgent]
    ) -> Optional[List[Tuple]]:
        """
        :param ego_agent: the ego agent
        :param cav_agents: cav agents
        :return: the prediction results before fusing the predictions from the ego and cav agents
        """

        def get_preds(agent):
            if agent.aid in self.cache:
                return self.cache[agent.aid]
            return agent.predict(self.env, **self.noise_setting)

            # Logic to predict the next frame

        ego_preds = get_preds(ego_agent)
        if ego_preds is None:
            return None
        results = []
        for cav in cav_agents:
            if not ego_agent.within(cav):
                continue
            ground_truth = get_ground_truth(ego_agent, cav, self.env)
            if ground_truth == 0:
                continue
            cav_preds = get_preds(cav)
            if cav_preds is None:
                continue
            ego_preds = filter_preds(ego_preds, ego_agent, cav)
            cav_preds = filter_preds(cav_preds, ego_agent, cav)
            if len(ego_preds) <= 1 or len(cav_preds) <= 1:
                continue
            K, n1, n2 = build_affinity_matrix(cav_preds, ego_preds)
            gt = (
                ego_preds[:, 0].reshape(-1, 1) == cav_preds[:, 0].reshape(1, -1)
            ).astype(np.int8)
            results.append((ego_preds, cav_preds, K.numpy(), gt))
        return results
