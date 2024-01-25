import unittest

import matplotlib.pyplot as plt

from src.models.agent import SimAgent
from src.models.scenario import *


class AgentTestCase(unittest.TestCase):
    def test_within1(self):
        env = SimScenario(
            100,
            100,
            [ObjectInfo("car", 50, (3, 4)), ObjectInfo("person", 50, (1, 1))],
            roads=[Road(1, math.pi / 4, (0, 0))],
        )
        env.generate_background()
        env.generate_objects()
        agent = SimAgent(math.pi / 2, 5, env, 1, (0, 0), math.pi / 4)
        plt.plot(*agent.sector.exterior.xy)
        plt.show()
        self.assertEqual(True, agent.within((0, 0)))
        self.assertEqual(False, agent.within((5, 5)))
        self.assertEqual(False, agent.within((0, 1)))
        self.assertEqual(False, agent.within((1, 0)))

    def test_within2(self):
        env = SimScenario(
            100,
            100,
            [ObjectInfo("car", 50, (3, 4)), ObjectInfo("person", 50, (1, 1))],
            roads=[Road(1, math.pi / 4, (0, 0))],
        )
        env.generate_background()
        env.generate_objects()
        agent = SimAgent(math.pi / 2, 5, env, 1, (50, 50), math.pi * 3 / 4)
        plt.plot(*agent.sector.exterior.xy)
        plt.show()
        self.assertEqual(True, agent.within((50, 50)))
        self.assertEqual(False, agent.within((55, 55)))
        self.assertEqual(True, agent.within((47, 53)))
        self.assertEqual(False, agent.within((53, 47)))
        self.assertEqual(False, agent.within((50, 55)))
        self.assertEqual(False, agent.within((45, 50)))

    def test_within3(self):
        env = SimScenario(
            100,
            100,
            [ObjectInfo("car", 50, (3, 4)), ObjectInfo("person", 50, (1, 1))],
            roads=[Road(1, math.pi / 4, (0, 0))],
        )
        env.generate_background()
        env.generate_objects()
        agent = SimAgent(math.pi / 3, 5, env, 1, (50, 50), math.pi * 1.25)
        self.assertEqual(True, agent.within((50, 50)))
        self.assertEqual(False, agent.within((55, 55)))
        self.assertEqual(True, agent.within((47, 47)))
        self.assertEqual(False, agent.within((50, 47)))
        self.assertEqual(False, agent.within((47, 50)))

    def test_within4(self):
        env = SimScenario(
            100,
            100,
            [ObjectInfo("car", 50, (3, 4)), ObjectInfo("person", 50, (1, 1))],
            roads=[Road(1, math.pi / 4, (0, 0))],
        )
        env.generate_background()
        env.generate_objects()
        agent = SimAgent(math.pi * 2 / 3, 50, env, 1, (0, 0), math.pi / 4)
        obj_count = 0
        for obj in env.objects.values():
            if agent.within(obj):
                obj_count += 1
        print(obj_count)

    def test_within5(self):
        env = SimScenario(
            100,
            100,
            [ObjectInfo("car", 50, (3, 4)), ObjectInfo("person", 50, (1, 1))],
            roads=[Road(1, math.pi / 4, (0, 0))],
        )
        env.generate_background()
        env.generate_objects()
        agent1 = SimAgent(math.pi * 2 / 3, 50, env, (1, 1), (0, 0), math.pi / 4)
        agent2 = SimAgent(math.pi * 2 / 3, 50, env, (-1, -1), (45, 45), math.pi * 5 / 4)
        self.assertEqual(True, agent1.within(agent2))

    def test_predict(self):
        env = SimScenario(
            100,
            100,
            [ObjectInfo("car", 80, (3, 4)), ObjectInfo("person", 20, (1, 1))],
            roads=[Road(1, math.pi / 4, (0, 0))],
        )
        env.generate_background()
        env.generate_objects()
        agent = SimAgent(math.pi * 2 / 3, 50, env, 1)
        preds = agent.predict(env)
        print("detected objects:", len(preds))


if __name__ == "__main__":
    unittest.main()
