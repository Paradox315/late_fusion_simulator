import unittest

from src.models.agent import SimAgent
from src.models.environment import *


class AgentTestCase(unittest.TestCase):
    def test_within1(self):
        env = SimEnvironment(
            100,
            100,
            1,
            [ObjectInfo("car", 50, (3, 4)), ObjectInfo("person", 50, (1, 1))],
        )
        env.generate_background()
        env.generate_objects()
        env.background.roads = [Road(1, math.pi / 4, (0, 0))]
        agent = SimAgent(math.pi / 3, 5, env, 1)
        agent.position = (0, 0)
        agent.heading = math.pi / 4
        self.assertEqual(True, agent.within((0, 0)))
        self.assertEqual(False, agent.within((5, 5)))
        self.assertEqual(False, agent.within((0, 1)))
        self.assertEqual(False, agent.within((1, 0)))

    def test_within2(self):
        env = SimEnvironment(
            100,
            100,
            1,
            [ObjectInfo("car", 10, (1, 1)), ObjectInfo("person", 10, (2, 3))],
        )
        env.generate_background()
        env.generate_objects()
        env.background.roads = [Road(1, math.pi / 4, (0, 0))]
        agent = SimAgent(math.pi / 3, 5, env, 1)
        agent.position = (50, 50)
        agent.heading = math.pi * 0.75
        self.assertEqual(True, agent.within((50, 50)))
        self.assertEqual(False, agent.within((55, 55)))
        self.assertEqual(True, agent.within((47, 53)))
        self.assertEqual(False, agent.within((53, 47)))
        self.assertEqual(False, agent.within((50, 55)))
        self.assertEqual(False, agent.within((45, 50)))

    def test_within3(self):
        env = SimEnvironment(
            100,
            100,
            1,
            [ObjectInfo("car", 10, (1, 1)), ObjectInfo("person", 10, (2, 3))],
        )
        env.generate_background()
        env.generate_objects()
        env.background.roads = [Road(1, math.pi / 4, (0, 0))]
        agent = SimAgent(math.pi / 3, 5, env, 1)
        agent.position = (50, 50)
        agent.heading = math.pi * 1.25
        self.assertEqual(True, agent.within((50, 50)))
        self.assertEqual(False, agent.within((55, 55)))
        self.assertEqual(True, agent.within((47, 47)))
        self.assertEqual(False, agent.within((50, 47)))
        self.assertEqual(False, agent.within((47, 50)))

    def test_within4(self):
        env = SimEnvironment(
            100,
            100,
            1,
            [ObjectInfo("car", 80, (3, 4)), ObjectInfo("person", 20, (1, 1))],
        )
        env.generate_background()
        env.generate_objects()
        env.background.roads = [Road(10, math.pi / 4, (0, 0))]
        agent = SimAgent(math.pi * 2 / 3, 50, env, 1)
        agent.position = (0, 0)
        agent.heading = math.pi / 4
        obj_count = 0
        within_count = 0
        for obj in env.objects.values():
            x, y, _, _ = obj
            if x**2 + y**2 <= 50**2:
                obj_count += 1
            if agent.within((x, y)):
                within_count += 1
        print(obj_count, within_count)
        self.assertEqual(obj_count, within_count)

    def test_within5(self):
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
        agent1.speed = (1, 1)
        agent2 = SimAgent(math.pi * 2 / 3, 50, env, 1)
        agent2.position = (45, 45)
        agent2.heading = math.pi * 5 / 4
        agent2.speed = (-1, -1)
        self.assertEqual(True, agent1.within(agent2))

    def test_predict(self):
        env = SimEnvironment(
            100,
            100,
            1,
            [ObjectInfo("car", 80, (3, 4)), ObjectInfo("person", 20, (1, 1))],
        )
        env.generate_background()
        env.generate_objects()
        # env.background.roads = [Road(10, math.pi / 4, (0, 0))]
        agent = SimAgent(math.pi * 2 / 3, 50, env, 1)
        # agent.position = (0, 0)
        # agent.heading = math.pi / 4
        preds = agent.predict(env)
        print("detected objects:", len(preds))


if __name__ == "__main__":
    unittest.main()
