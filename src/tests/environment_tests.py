import unittest

from src.models.environment import *


class RoadTestCase(unittest.TestCase):
    def test_Road1(self):
        road = Road(1, 0, (0, 0))
        self.assertEqual(True, road.within((0, 0)))
        self.assertEqual(True, road.within((0, 0.5)))
        self.assertEqual(True, road.within((0, -0.5)))
        self.assertEqual(True, road.within((1, -0.5)))
        self.assertEqual(False, road.within((0, 1)))

    def test_Road2(self):
        road = Road(1, math.pi / 2, (0, 0))
        self.assertEqual(True, road.within((0, 0)))
        self.assertEqual(True, road.within((0.5, 0)))
        self.assertEqual(True, road.within((-0.5, 0)))
        self.assertEqual(True, road.within((-0.5, 1)))
        self.assertEqual(False, road.within((1, 0)))

    def test_Road3(self):
        road = Road(1, math.pi / 4, (0, 0))
        self.assertEqual(True, road.within((0, 0)))
        self.assertEqual(True, road.within((0, 0.5)))
        self.assertEqual(True, road.within((0, -0.5)))
        self.assertEqual(False, road.within((0.6, 0)))
        self.assertEqual(False, road.within((0, 0.6)))
        self.assertEqual(False, road.within((-0.6, 0)))


class BackgroundTestCase(unittest.TestCase):
    def test_Background1(self):
        background = Background(100, 100)
        road = Road(1, 0, (0, 50))
        background.roads = [road]
        self.assertEqual(True, background.within_road((0, 50)))
        self.assertEqual(True, background.within_road((0, 50.5)))
        self.assertEqual(True, background.within_road((0, 49.5)))
        self.assertEqual(True, background.within_road((1, 49.5)))
        self.assertEqual(False, background.within_road((0, 51)))

    def test_Background2(self):
        background = Background(100, 100)
        road1 = Road(1, math.pi / 2, (50, 0))
        road2 = Road(1, 0, (0, 50))
        background.roads = [road1, road2]
        self.assertEqual(True, background.within_road((50, 0)))
        self.assertEqual(True, background.within_road((50.5, 0)))
        self.assertEqual(True, background.within_road((49.5, 0)))
        self.assertEqual(True, background.within_road((49.5, 100)))
        self.assertEqual(False, background.within_road((51, 0)))
        self.assertEqual(True, background.within_road((50, 50)))
        self.assertEqual(False, background.within_road((51, 51)))
        self.assertEqual(False, background.within_road((25, 25)))


class EnvironmentTestCase(unittest.TestCase):
    def test_Environment1(self):
        env = SimEnvironment(
            100,
            100,
            1,
            [ObjectInfo("car", 10, (1, 1)), ObjectInfo("person", 10, (2, 3))],
        )
        env.generate_background()
        env.generate_objects()
        for obj in env.objects.values():
            self.assertEqual(False, env.background.within_road(obj))
        self.assertEqual(20, len(env.objects))


if __name__ == "__main__":
    unittest.main()
