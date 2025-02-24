import functools
import math
import random
import time
from typing import List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from src.models.agent import SimAgent
from src.models.scenario import SimScenario, ObjectInfo, Road
from src.models.frame import Frame
from src.runner.runner import Simulator
from src.utils.CBM import ctx_base_matching
from src.utils.lap import auction_match, graph_based_match, hungarian_match, history
from src.utils.log import init_ini_log, logger

random.seed(2023)
np.random.seed(2023)
_ = torch.manual_seed(2023)


def init_env():
    env = SimScenario(
        100,
        100,
        [ObjectInfo("car", 100, (3, 4)), ObjectInfo("person", 100, (1, 1))],
        roads=[
            Road(10, math.pi / 4, (0, 0)),
            Road(10, math.pi * 3 / 4, (0, 100)),
        ],
    )
    env.generate_background()
    env.generate_objects()
    return env


def init_agents(env) -> List[SimAgent]:
    agent1 = SimAgent(math.pi / 2, 50, env, 1)
    # agent1.position = (25, 25)
    # agent1.heading = math.pi / 4
    # agent1.speed = (1, 1)
    agent2 = SimAgent(math.pi / 2, 50, env, 1)
    # agent2.position = (75, 75)
    # agent2.heading = math.pi * 5 / 4
    # agent2.speed = (-1, -1)
    agent3 = SimAgent(math.pi / 2, 50, env, 1)
    # agent3.position = (3, 0)
    # agent3.heading = math.pi / 4
    # agent3.speed = (1, 1)
    agent4 = SimAgent(math.pi / 2, 50, env, 1)
    return [agent1, agent2, agent3, agent4]


def init_simulator(env, agents, noise):
    frame = Frame(env, agents, noise_setting=noise)
    simulator = Simulator(1, frame)
    return simulator


def mean_cost_time(history: List, unit="ms"):
    avg = sum(history) / len(history)
    if unit == "ms":
        return avg / 1e6
    elif unit == "s":
        return avg / 1e9
    elif unit == "us":
        return avg / 1e3
    else:
        return avg


noise_setting = [
    {"noise_pos": 0.1, "noise_shape": 0.1, "noise_heading": math.radians(1)},
    {"noise_pos": 2, "noise_shape": 0.1, "noise_heading": math.radians(1)},
]
graph_match_funcs = ["rrwm", "ipfp", "sm", "ngm"]
node_match_funcs = [hungarian_match, auction_match]
if __name__ == "__main__":
    env = init_env()
    agents = init_agents(env)
    for i, noise in enumerate(noise_setting):
        simulator = init_simulator(env, agents, noise)
        # for node_match_func in node_match_funcs:
        #     logger.info(
        #         {
        #             "message": f"starting simulation {i+1} using {node_match_func} for node matching",
        #             "noise_setting": noise,
        #         }
        #     )
        #     start = time.perf_counter_ns()
        #     simulator.run(node_match_func, visualize=False)
        #     end = time.perf_counter_ns()
        #     logger.info(
        #         {
        #             "message": f"simulation {i+1} using {node_match_func} for node matching finished",
        #             "time": (end - start) / simulator.get_records_length() * 1e-6,
        #         }
        #     )
        #     print(f"mean cost time: {mean_cost_time(history)}ms")
        #     history.clear()
        #     simulator.save_simulation_results()
        #     # simulator.save_simulation_results(
        #     #     format="json", path=f"./data/node_match_{node_match_func}_{i}.json"
        #     # )
        for match_func in graph_match_funcs:
            logger.info(
                {
                    "message": f"starting simulation {i+1} using {match_func} for graph matching",
                    "match_func": match_func,
                    "noise_setting": noise,
                }
            )
            start = time.perf_counter_ns()
            simulator.run(
                functools.partial(
                    graph_based_match, associate_func="hungarian", match_func=match_func
                ),
                visualize=False,
            )
            end = time.perf_counter_ns()
            logger.info(
                {
                    "message": f"simulation {i+1} using {match_func} for graph matching finished",
                    "match_func": match_func,
                    "time": (end - start) / simulator.get_records_length() * 1e-6,
                }
            )
            print(f"mean cost time: {mean_cost_time(history)}ms")
            history.clear()
            simulator.save_simulation_results()
        # simulator.save_simulation_results(
        #     format="json", path=f"./data/graph_match_{match_func}_{i}.json"
        # )
    print("finished")
