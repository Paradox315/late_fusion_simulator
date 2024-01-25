import math
import unittest

import numpy as np
from shapely import Polygon, Point
from shapely.affinity import rotate
import matplotlib.pyplot as plt
from src.utils.tools import compute_iou, center_to_coord, sector


class IoUTestCase(unittest.TestCase):
    def test_iou(self):
        boxes1 = (0.5, 0.5, 1, 1, 0)
        boxes2 = (0, 0, 1, 1, 0)
        iou = compute_iou(boxes1, boxes2)
        self.assertEqual(iou, 1 / 7)

    def test_rotate(self):
        boxes = np.array((2, 2, 3, 4, math.pi * 0.99))
        loc = center_to_coord(boxes)
        dist1 = math.dist(loc[0], loc[1])
        dist2 = math.dist(loc[1], loc[2])
        self.assertAlmostEqual(dist1 * dist2, 12)

    def test_sector(self):
        sect = sector(Point(0, 0), math.pi, 2 * math.pi, 1)
        print(sect.area)
        plt.plot(*sect.exterior.xy)
        plt.show()


if __name__ == "__main__":
    unittest.main()
