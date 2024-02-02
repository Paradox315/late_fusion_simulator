import math
import random
from typing import overload, Optional

import numpy as np
from matplotlib import patches
from shapely import box
from shapely.affinity import rotate
from shapely.geometry import Point, Polygon

from src.models.scenario import Scenario, SimScenario
from src.utils.tools import sector


class Agent:
    count = 0

    def __init__(
        self,
        position: tuple[float, float],
        ori: float,
        field_of_view: float,
        distance: float,
    ):
        """
        :param position: initial position
        :param ori: move orientation
        :param field_of_view: agent's field of view
        :param distance: agent's detection distance
        """
        Agent.count += 1
        self.aid = Agent.count
        self.position = position
        self.heading = ori
        self.field_of_view = field_of_view
        self.distance = distance


class SimAgent(Agent):
    def __init__(
        self,
        field_of_view: float,
        distance: float,
        env: Scenario,
        speed: float | tuple[float, float] | int = 1.0,
        position: Optional[tuple[float, float]] = None,
        ori: Optional[float] = None,
    ):
        super().__init__(position, ori, field_of_view, distance)
        if position is None or ori is None:
            self._random_init(env)
        if isinstance(speed, float) or isinstance(speed, int):
            self.speed = (
                speed * math.cos(self.heading),
                speed * math.sin(self.heading),
            )
        else:
            self.speed = speed
        # generate the agent's detection area
        sect = sector(
            Point(*self.position),
            self.heading - self.field_of_view / 2,
            self.heading + self.field_of_view / 2,
            self.distance,
        )

        # set the agent's detection area
        self.sector: Polygon = sect

    def __eq__(self, other):
        return self.aid == other.aid

    def _random_init(self, env: Scenario):
        road = random.choice(env.background.roads)
        pos = road.random_pos(env.background.width, env.background.height)
        self.position = pos
        ori = road.direction
        if random.random() > 0.5:
            ori += math.pi
        self.heading = ori

    def move(self, dt=1):
        """
        :param dt: time delta
        :return: agent's next position
        """
        x, y = self.position
        vx, vy = self.speed
        self.position = (x + vx * dt, y + vy * dt)

    @overload
    def within(self, point: tuple):
        """
        Check if the given point is within the agent's field of view
        :param point:
        :return:
        """
        pass

    @overload
    def within(self, agent: Agent):
        """
        Check if the given agent is within the agent's field of view
        :param agent:
        :return:
        """
        pass

    @overload
    def within(self, box: np.ndarray):
        """
        Check if the given box is within the agent's field of view
        :param box:
        :return:
        """
        pass

    def within(self, arg: Agent | np.ndarray | tuple) -> bool:
        if isinstance(arg, np.ndarray):
            if arg.shape == (2,):
                x, y = arg
                return self.sector.contains(Point(x, y))
            x, y, w, h, theta = arg.T
            rect = box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            rotated_rect = rotate(rect, theta, origin=(x, y), use_radians=True)
            return self.sector.covers(rotated_rect)
        elif isinstance(arg, Agent):
            cav = arg
            return self.sector.intersects(cav.sector)
        elif isinstance(arg, tuple):
            x, y = arg
            if arg == self.position:
                return True
            return self.sector.contains(Point(x, y))
        else:
            raise ValueError(f"Invalid argument type: {type(arg)}")

    def predict(
        self,
        env: SimScenario | Scenario,
        noise_pos=0.1,
        noise_shape=0.1,
        noise_heading=0.1,
    ) -> Optional[np.ndarray]:
        """
        Detect the agents in the environment
        :param noise_shape: shape noise
        :param noise_pos: detection center noise
        :param noise_heading: heading noise
        :param env: agent's environment
        :return:
        """
        detected = []  # object box(xc, yc, w, h, theta)
        detected_ids = []
        object_shapes = np.array([info.shape for info in env.obj_info])

        for obj_id, obj in env.objects.items():
            if not self.within(obj):
                continue
            obj = obj.copy()
            obj[:2] += np.random.randn(2) * noise_pos
            close_idx = np.isclose(obj[2:-1], object_shapes).all(axis=1)
            if not close_idx.any():
                continue
            shape = object_shapes[close_idx][0]
            obj[2:-1] = shape + np.random.randn(2) * noise_shape
            obj[-1] += np.random.randn() * noise_heading * math.pi
            detected.append(obj)
            detected_ids.append(obj_id)
        if not detected:
            return None

        preds = np.array(detected)
        pred_ids = np.array(detected_ids)
        probs = np.zeros((preds.shape[0], len(object_shapes)))
        # 添加类别概率分布
        for i, object_shape in enumerate(object_shapes):
            close_idxs = np.isclose(
                preds[:, 2:-1], object_shape, atol=noise_shape * 10
            ).all(axis=1)
            probs[close_idxs, i] = np.random.uniform(0.7, 1, size=close_idxs.sum())
        for prob in probs:
            no_assign_prob = 1 - np.sum(prob)
            if no_assign_prob:
                w = np.random.rand(len(prob[prob == 0]))
                w /= w.sum()
                prob[prob == 0] = w * no_assign_prob
        cls = probs.argmax(axis=1)
        # 添加协同检测(x, y, w, h, theta, cls, prob)
        return np.hstack((pred_ids[:, np.newaxis], preds, cls[:, np.newaxis], probs))

    def visualize(self, ax):
        x, y = self.position
        ax.scatter(x, y, marker="x", c="r")
        ax.arrow(
            x,
            y,
            math.cos(self.heading) * self.distance * 0.25,
            math.sin(self.heading) * self.distance * 0.25,
            head_width=1,
            head_length=1,
            fc="r",
            ec="r",
        )
        ax.add_patch(
            patches.Wedge(
                (x, y),
                self.distance,
                math.degrees(self.heading - self.field_of_view / 2),
                math.degrees(self.heading + self.field_of_view / 2),
                color="g",
                alpha=0.2,
            )
        )
        ax.text(
            x,
            y,
            f"{self.aid}",
            ha="center",
            va="center",
            color="r",
            fontsize="xx-large",
        )
