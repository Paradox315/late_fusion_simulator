import math
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.models.agent import SimAgent
from src.models.environment import SimEnvironment, ObjectInfo, Road
from src.models.frame import Frame
from src.runner.runner import Simulator

random.seed(2024)
np.random.seed(2024)
env = SimEnvironment(
    100,
    100,
    1,
    [ObjectInfo("car", 100, (3, 4)), ObjectInfo("person", 100, (1, 1))],
)
env.generate_background()
env.background.roads = [Road(10, math.pi / 4, (0, 0))]
env.generate_objects()
agent1 = SimAgent(math.pi * 2 / 3, 50, env, 1)
# agent1.position = (25, 25)
# agent1.heading = math.pi / 4
# agent1.speed = (1, 1)
agent2 = SimAgent(math.pi * 2 / 3, 50, env, 1)
# agent2.position = (75, 75)
# agent2.heading = math.pi * 5 / 4
# agent2.speed = (-1, -1)
agent3 = SimAgent(math.pi * 2 / 3, 50, env, 1)
# agent3.position = (3, 0)
# agent3.heading = math.pi / 4
# agent3.speed = (1, 1)
agent4 = SimAgent(math.pi * 2 / 3, 50, env, 1)
frame = Frame(env, [agent1, agent2, agent3])
simulator = Simulator(10, frame)
results = simulator.run(linear_sum_assignment, visualize=False)
simulator.save_simulation_results(format="json")
