import math
import random
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from src.models.agent import SimAgent
from src.models.scenario import SimScenario, ObjectInfo, Road
from src.models.frame import Frame
from src.runner.runner import Simulator
from src.utils.CBM import ctx_base_matching
from src.utils.lap import auction_match, graph_based_match, hungarian_match

random.seed(2023)
np.random.seed(2023)
_ = torch.manual_seed(2023)
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
frame = Frame(env, [agent1, agent2, agent3, agent4])
simulator = Simulator(2, frame)
noise_setting = [
    {"noise_pos": 0.01, "noise_shape": 0.01, "noise_heading": math.radians(0.01)},
    {"noise_pos": 0.6, "noise_shape": 0.1, "noise_heading": math.radians(3)},
    # {"noise_pos": 0.05, "noise_shape": 0.05, "noise_heading": 0.05},
    # {"noise_pos": 0.1, "noise_shape": 0.1, "noise_heading": 0.1},
    # {"noise_pos": 0.2, "noise_shape": 0.2, "noise_heading": 0.2},
]
for i, noise in enumerate(noise_setting):
    start = time.perf_counter()
    simulator.run(graph_based_match, visualize=False, noise_setting=noise)
    end = time.perf_counter()
    print(f"第{i+1}次仿真耗时{end-start}秒")
    simulator.save_simulation_results(
        format="json", path=f"../data/graph_match_3_{i}.json"
    )
    # simulator.save_simulation_results(format="console")
print("finished")
