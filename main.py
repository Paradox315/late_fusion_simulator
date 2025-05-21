import functools
import math
import random
import time
from typing import List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from networks.gcn_net_batch import GCN_Net, ScoreLayer
from src.models.agent import SimAgent
from src.models.scenario import SimScenario, ObjectInfo, Road
from src.models.frame import Frame
from src.runner.runner import Simulator
from src.utils.CBM import ctx_base_matching
from src.utils.lap import auction_match, graph_based_match, hungarian_match, history
from src.utils.log import init_ini_log, logger
from utils.lap import gcn_graph_match

random.seed(2023)
np.random.seed(2023)
_ = torch.manual_seed(2023)


def init_env():
    env = SimScenario(
        100,
        100,
        [ObjectInfo("car", 75, (3, 4)), ObjectInfo("person", 75, (1, 1))],
        roads=[
            Road(10, math.pi / 4, (0, 0)),
            Road(10, math.pi * 3 / 4, (0, 100)),
        ],
    )
    env.generate_background()
    env.generate_objects()
    return env


def init_agents(env, num_agents=4, random_init=False) -> List[SimAgent]:
    """
    Initialize agents for the simulation

    Args:
        env: The simulation environment
        num_agents: Number of agents to create
        random_init: If True, agents will be randomly initialized
                     If False, predefined positions will be used

    Returns:
        List of initialized SimAgent objects
    """
    agents = []

    if not random_init:
        # Use predefined positions and settings
        agent_configs = [
            {"position": (25, 25), "heading": math.pi / 4, "speed": (1, 1)},
            {"position": (75, 75), "heading": math.pi * 5 / 4, "speed": (-1, -1)},
            {"position": (3, 0), "heading": math.pi / 4, "speed": (1, 1)},
            {"position": (0, 3), "heading": math.pi * 5 / 4, "speed": (-1, -1)},
        ]

        # Create agents with predefined configurations
        for i in range(min(num_agents, len(agent_configs))):
            agent = SimAgent(math.pi / 2, 50, env, 1)
            agent.position = agent_configs[i]["position"]
            agent.heading = agent_configs[i]["heading"]
            agent.speed = agent_configs[i]["speed"]
            agents.append(agent)
    else:
        # Create randomly initialized agents
        for _ in range(num_agents):
            agent = SimAgent(
                field_of_view=math.pi / 2,
                distance=50,
                env=env,
                speed=1.0,
                position=None,  # This triggers random initialization
                ori=None,  # This triggers random initialization
            )
            agents.append(agent)

    return agents


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
    # {"noise_pos": 2, "noise_shape": 0.1, "noise_heading": math.radians(3)},
]
graph_match_funcs = ["gcn"]
node_match_funcs = [hungarian_match, auction_match]
if __name__ == "__main__":
    env = init_env()
    agents = init_agents(env, 4, random_init=False)
    # torch.backends.quantized.engine = "qnnpack"
    gcn_model = GCN_Net((32, 32, 32))
    # state_dict = torch.load("checkpoints/gcn_model_20.pth")
    score_model = ScoreLayer()

    for i, noise in enumerate(noise_setting):
        simulator = init_simulator(env, agents, noise)
        # for node_match_func in node_match_funcs:
        #     func_name = node_match_func.__name__
        #     logger.info(
        #         {
        #             "message": f"starting simulation {i+1} using {func_name} for node matching",
        #             "noise_setting": noise,
        #         }
        #     )
        #     start = time.perf_counter_ns()
        #     simulator.run(node_match_func, visualize=False)
        #     end = time.perf_counter_ns()
        #     logger.info(
        #         {
        #             "message": f"simulation {i+1} using {func_name} for node matching finished",
        #             "time": (end - start) / simulator.get_records_length() * 1e-6,
        #         }
        #     )
        #     print(f"mean cost time: {mean_cost_time(history)}ms")
        #     history.clear()
        #     # simulator.save_simulation_results()
        #     simulator.save_simulation_results(
        #         format="json", path=f"./data/node_match_{func_name}_{i}.json"
        #     )
        for match_func in graph_match_funcs:
            logger.info(
                {
                    "message": f"starting simulation {i+1} using {match_func} for graph matching",
                    "match_func": match_func,
                    "noise_setting": noise,
                }
            )
            start = time.perf_counter_ns()
            # simulator.run(
            #     functools.partial(
            #         graph_based_match, associate_func="hungarian", match_func=match_func
            #     ),
            #     visualize=False,
            # )
            gcn_match_func = functools.partial(
                gcn_graph_match,
                gcn_model=gcn_model,
                score_model=score_model,
            )
            simulator.run(gcn_match_func)
            end = time.perf_counter_ns()
            logger.info(
                {
                    "message": f"simulation {i+1} using {match_func} for graph matching finished",
                    "match_func": match_func,
                    "time": (end - start) / simulator.get_records_length() * 1e-6,
                }
            )
            # print(f"mean cost time: {mean_cost_time(history)}ms")
            # history.clear()
            simulator.save_simulation_results()
            # simulator.save_simulation_results(
            #     format="json", path=f"./data/graph_match_{match_func}_{i}.json"
            # )
    print("finished")
