import math
import unittest

from src.models.agent import SimAgent
from src.models.environment import SimEnvironment, ObjectInfo, Road
from src.models.frame import Frame


class FrameTestCase(unittest.TestCase):
    def test_init(self):
        env = SimEnvironment(
            100,
            100,
            1,
            [ObjectInfo("car", 80, (3, 4)), ObjectInfo("person", 20, (1, 1))],
        )
        env.generate_background()
        env.generate_objects()
        env.background.roads = [Road(10, math.pi / 4, (0, 0))]
        agent1 = SimAgent(math.pi * 2 / 3, 50, env, 1)
        agent1.position = (0, 0)
        agent1.heading = math.pi / 4
        agent2 = SimAgent(math.pi * 2 / 3, 50, env, 1)
        agent2.position = (100, 100)
        agent2.heading = math.pi * 5 / 4
        frame = Frame(env, [agent1, agent2])
        for i in range(10):
            frame = next(frame)

    def test_visualize(self):
        env = SimEnvironment(
            100,
            100,
            1,
            [ObjectInfo("car", 80, (3, 4)), ObjectInfo("person", 20, (1, 1))],
        )
        env.generate_background()
        # env.background.roads = [Road(10, math.pi / 4, (0, 0))]
        env.generate_objects()
        agent1 = SimAgent(math.pi * 2 / 3, 50, env, 1)
        # agent1.position = (0, 0)
        # agent1.heading = math.pi / 4
        agent2 = SimAgent(math.pi * 2 / 3, 50, env, 1)
        # agent2.position = (100, 100)
        # agent2.heading = math.pi * 5 / 4
        frame = Frame(env, [agent1, agent2])
        frame.visualize()


if __name__ == "__main__":
    unittest.main()
