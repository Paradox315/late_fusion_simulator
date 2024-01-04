import collections
import math
import random
from abc import ABC, abstractmethod
from typing import List, overload

import matplotlib.pyplot as plt
import numpy as np

ObjectInfo = collections.namedtuple("ObjectInfo", ["name", "num", "shape"])


class Environment(ABC):
    def __init__(self):
        self.background = None
        self.objects = {}

    @abstractmethod
    def generate_background(self):
        pass

    @abstractmethod
    def generate_objects(self):
        pass


class SimEnvironment(Environment):
    def __init__(
        self, width: int, height: int, roads_num: int, obj_info: List[ObjectInfo]
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.roads_num = roads_num
        self.obj_info = obj_info

    def generate_background(self):
        if self.background is None:
            self.background = Background(self.width, self.height)
            self.background.generate(self.roads_num)

    def generate_objects(self):
        """
        Generate objects based on the given object info
        :return: List of object boxes (xc,yc,w,h)
        """
        id = 1
        for info in self.obj_info:
            for _ in range(info.num):
                obj = self._generate_object(info.shape)
                self.objects[id] = obj
                id += 1

    def _generate_object(self, shape: tuple[float, float]) -> np.ndarray:
        """
        Generate a object with given shape
        :param shape: (w,h)
        :return: object box (xc, yc, w, h)
        """
        w, h = shape
        xc, yc = random.uniform(w / 2, self.width - w / 2), random.uniform(
            h / 2, self.height - h / 2
        )
        box = np.array((xc, yc, w, h))
        while self.background.within_road(box):
            xc, yc = random.uniform(w / 2, self.width - w / 2), random.uniform(
                h / 2, self.height - h / 2
            )
            box = np.array((xc, yc, w, h))
        return box

    def visualize(self, ax):
        self.background.visualize(ax)
        for obj in self.objects.values():
            x, y, w, h = obj
            ax.add_patch(
                plt.Rectangle(
                    (x - w / 2, y - h / 2),
                    w,
                    h,
                    linewidth=1,
                    edgecolor="b",
                    facecolor="none",
                )
            )


class Background:
    def __init__(self, width: int, height: int):
        self.height = height
        self.width = width
        self.roads = []

    def generate(self, roads_num: int):
        for i in range(roads_num):
            road_width = random.uniform(0.15, 0.3) * self.width
            road_direction = random.uniform(0, math.pi)
            road_start_point = (
                random.uniform(road_width / 2, self.width - road_width / 2),
                random.uniform(road_width / 2, self.height - road_width / 2),
            )
            self.roads.append(Road(road_width, road_direction, road_start_point))

    @overload
    def within_road(self, point: tuple[float, float]) -> bool:
        pass

    @overload
    def within_road(self, box: np.ndarray) -> bool:
        pass

    def within_road(self, arg: tuple[float, float] | np.ndarray) -> bool:
        """
        Check if the given point or box is on the road
        :param arg: (x,y) or (xc,yc,w,h)
        :return:
        """
        if isinstance(arg, tuple):
            assert 0 <= arg[0] <= self.width and 0 <= arg[1] <= self.height
            return any(road.within(arg) for road in self.roads)
        elif isinstance(arg, np.ndarray):
            assert arg.shape == (4,)
            xc, yc, w, h = arg
            assert 0 <= xc <= self.width and 0 <= yc <= self.height
            delta_loc = [
                (-w / 2, -h / 2),
                (-w / 2, h / 2),
                (w / 2, -h / 2),
                (w / 2, h / 2),
            ]
            loc = [(xc + x, yc + y) for x, y in delta_loc]
            return any(self.within_road(point) for point in loc)
        else:
            raise TypeError("arg must be tuple[float, float] or np.ndarray")

    def visualize(self, ax):
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        for road in self.roads:
            road.visualize(ax, self.width, self.height)


class Road:
    def __init__(
        self, width: float, direction: float, start_point: tuple[float, float]
    ):
        self.width = width
        self.direction = direction
        self.start_point = start_point

    def within(self, point: tuple[float, float]):
        """
        Check if the given point is on the road
        :param point:(x,y)
        :return:
        """
        x, y = point
        x0, y0 = self.start_point
        if self.direction == math.pi / 2:
            return x0 - self.width / 2 <= x <= x0 + self.width / 2
        slope = math.tan(self.direction)
        return (
            slope * (x - x0) + y0 - self.width / 2
            <= y
            <= slope * (x - x0) + y0 + self.width / 2
        )

    def random_pos(self, w, h):
        x = random.uniform(0, w)
        y = random.uniform(0, h)
        while not self.within((x, y)):
            x = random.uniform(0, w)
            y = random.uniform(0, h)
        return x, y

    def visualize(self, ax, width, height):
        x0, y0 = self.start_point
        if self.direction == math.pi / 2:
            y = np.linspace(0, height, 100)
            ax.fill_betweenx(
                y,
                x0 - self.width / 2,
                x0 + self.width / 2,
                color="gray",
            )
            return
        x = np.linspace(0, width, 100)
        slope = math.tan(self.direction)
        ax.fill_between(
            x,
            slope * (x - x0) + y0 - self.width / 2,
            slope * (x - x0) + y0 + self.width / 2,
            color="gray",
        )


class RealEnvironment(Environment):
    def generate_background(self):
        # Logic to generate the background based on reconstructed data
        pass

    def generate_objects(self):
        # Logic to generate the target based on reconstructed data
        pass
