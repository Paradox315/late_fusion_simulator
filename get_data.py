import itertools
import json
import math
import pickle
import random
from typing import List
from tqdm import tqdm
from tqdm import trange
import numpy as np
import torch

from src.models.agent import SimAgent
from src.models.frame import Frame
from src.models.scenario import SimScenario, ObjectInfo, Road
from src.runner.runner import Simulator

random.seed(2023)
np.random.seed(2023)
_ = torch.manual_seed(2023)
dataset_path = "./data/detect_dataset"


def init_env():
    env = SimScenario(
        100,
        100,
        [ObjectInfo("car", 60, (3, 4)), ObjectInfo("person", 60, (1, 1))],
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
    simulator = Simulator(50, frame)
    return simulator


def init_noise_setting():
    noise_pos_values = [0.25, 0.5, 1, 1.25, 1.5, 2]
    noise_shape_values = [0.05, 0.075, 0.1]
    noise_heading_values = list(map(math.radians, [1, 3, 5]))

    noise_setting = [
        {
            "noise_pos": noise_pos,
            "noise_shape": noise_shape,
            "noise_heading": noise_heading,
        }
        for noise_pos, noise_shape, noise_heading in itertools.product(
            noise_pos_values, noise_shape_values, noise_heading_values
        )
    ]
    return noise_setting


def save_data(data, mode="train"):
    file_list = []
    for i in trange(len(data)):
        part_data = data[i]
        fname = f"{mode}/part{i + 1}.pkl"
        file_list.append(fname)
        with open(f"{dataset_path}/{fname}", "wb") as f:
            pickle.dump(part_data, f)
    print("Data saved to", f"{dataset_path}/{mode}")
    with open(f"{dataset_path}/{mode}_parts.json", "w") as f:
        json.dump(file_list, f)


noise_setting = init_noise_setting()
if __name__ == "__main__":
    env = init_env()
    agents = init_agents(env)
    train, val, test = [], [], []
    for noise in tqdm(noise_setting):
        simulator = init_simulator(env, agents, noise)
        train_part, val_part, test_part = simulator.get_dataset()
        train.extend(train_part)
        val.extend(val_part)
        test.extend(test_part)
    print("train:", len(train))
    print("val:", len(val))
    print("test:", len(test))
    save_data(train, "train")
    save_data(val, "validate")
    save_data(test, "test")

    print("finished")
