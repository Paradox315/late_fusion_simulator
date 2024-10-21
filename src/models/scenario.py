import collections
import math
import random
from abc import ABC, abstractmethod
from typing import List, overload

import matplotlib.pyplot as plt
import numpy as np

from src.utils.tools import center_to_coord

ObjectInfo = collections.namedtuple("ObjectInfo", ["name", "num", "shape"])


class Scenario(ABC):
    def __init__(self):
        self.background = None
        self.objects = {}

    @abstractmethod
    def generate_background(self):
        pass

    @abstractmethod
    def generate_objects(self):
        pass


class SimScenario(Scenario):
    def __init__(
        self,
        width: int,
        height: int,
        obj_info: List[ObjectInfo],
        roads_num: int = 0,
        roads=None,
    ):
        """
        :param width: the width of the background(can't be empty)
        :param height: the height of the background(can't be empty)
        :param obj_info: the information of the objects(can't be empty)
        :param roads_num: the number of the roads(default 0, set the roads_num to 0 if you don't want to generate roads)
        :param roads: the roads(can't be empty if roads_num is 0)
        """
        super().__init__()
        self.width = width
        self.height = height
        self.obj_info = obj_info
        self.roads_num = roads_num
        self.roads = roads
        if roads_num == 0 and roads is None:
            raise ValueError("roads_num and roads can't be both empty")

    def generate_background(self):
        """
        Generate the background
        :return:
        """
        if self.background:
            return
        self.background = Background(self.width, self.height, self.roads)
        self.background.generate(self.roads_num)

    def generate_objects(self):
        """
        Generate objects based on the given object info
        :return: List of object boxes (xc,yc,w,h)
        """
        id = 1
        for info in self.obj_info:
            object_generator = self._generate_object(info.shape)
            for _ in range(info.num):
                for obj in object_generator:
                    if not self.background.within_road(obj):
                        self.objects[id] = obj
                        id += 1
                        break

    def _generate_object(self, shape: tuple[float, float]) -> np.ndarray:
        """
        Generate a object with given shape
        :param shape: (w,h)
        :return: object box (xc, yc, w, h, theta)
        """
        w, h = shape
        max_size = math.dist((0, 0), (w, h))
        while True:
            xc, yc = random.uniform(
                max_size / 2, self.width - max_size / 2
            ), random.uniform(max_size / 2, self.height - max_size / 2)
            theta = random.uniform(0, math.pi)
            box = np.array((xc, yc, w, h, theta))
            yield box

    def visualize(self, ax):
        self.background.visualize(ax)
        for obj in self.objects.values():
            x, y, w, h, theta = obj
            ax.add_patch(
                plt.Rectangle(
                    (x - w / 2, y - h / 2),
                    w,
                    h,
                    angle=theta * 180 / math.pi,
                    linewidth=1,
                    edgecolor="b",
                    facecolor="none",
                )
            )


class Background:
    def __init__(self, width: int, height: int, roads=None):
        self.height = height
        self.width = width
        self.roads = roads if roads is not None else []

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
        :param arg: (x, y) or (xc, yc, w, h, theta)
        :return:
        """
        if isinstance(arg, tuple):
            assert 0 <= arg[0] <= self.width and 0 <= arg[1] <= self.height
            return any(road.within(arg) for road in self.roads)
        elif isinstance(arg, np.ndarray):
            assert arg.shape == (5,)
            loc = center_to_coord(arg)
            return any(self.within_road((x, y)) for x, y in loc)
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
        def position_generator(w, h):
            while True:
                x = random.uniform(0, w)
                y = random.uniform(0, h)
                if self.within((x, y)):
                    yield x, y

        for position in position_generator(w, h):
            if self.within(position):
                return position

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


class RealScenario(Scenario):
    def generate_background(self):
        # Logic to generate the background based on reconstructed data
        pass

    def generate_objects(self):
        # Logic to generate the target based on reconstructed data
        pass
