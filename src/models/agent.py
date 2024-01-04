import math
import random
from typing import overload

import numpy as np
from matplotlib import patches

from src.models.environment import Environment, SimEnvironment


class Agent:
    count = 0

    def __init__(
        self,
        position: tuple[float, float],
        ori: float,
        field_of_view: float,
        distance: float,
    ):
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
        env: Environment,
        speed: float,
    ):
        super().__init__((0, 0), 0, field_of_view, distance)
        road = random.choice(env.background.roads)
        pos = road.random_pos(env.background.width, env.background.height)
        self.position = pos
        ori = road.direction
        # 如果在地图右半边，方向要反过来
        if random.random() > 0.5:
            ori += math.pi
        self.heading = ori
        self.speed = (speed * math.cos(self.heading), speed * math.sin(self.heading))

    def __eq__(self, other):
        return self.aid == other.aid

    def move(self, dt=1):
        x, y = self.position
        vx, vy = self.speed
        self.position = (x + vx * dt, y + vy * dt)

    @overload
    def within(self, point: tuple[float, float]):
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

    def within(self, arg: tuple[float, float] | Agent) -> bool:
        if isinstance(arg, tuple):
            x, y = arg
            x0, y0 = self.position
            dx, dy = x - x0, y - y0
            if dx == 0 and dy == 0:
                return True
            if dx**2 + dy**2 > self.distance**2:
                return False
            angle = math.atan2(dy, dx)
            # 保证角度在0到pi之间
            if angle < 0:
                angle += math.pi
            # 如果目标在agent下方，角度要加pi
            if dy < 0:
                angle += math.pi
            return abs(angle - self.heading) < self.field_of_view / 2
        elif isinstance(arg, Agent):
            cav = arg
            ego_x, ego_y = self.position
            cav_x, cav_y = cav.position
            dx, dy = cav_x - ego_x, cav_y - ego_y
            if dx == 0 and dy == 0:
                return True
            dist = math.sqrt(dx**2 + dy**2)
            if dist > self.distance + cav.distance:
                return False
            max_attempts = 50
            for _ in range(max_attempts):
                theta = random.uniform(
                    self.heading - self.field_of_view / 2,
                    self.heading + self.field_of_view / 2,
                )
                dist = random.uniform(0, self.distance)
                x = ego_x + dist * math.cos(theta)
                y = ego_y + dist * math.sin(theta)
                if cav.within((x, y)):
                    return True
            return False

    def predict(self, env: SimEnvironment, noise_pos=0.1, noise_shape=0.1):
        """
        Detect the agents in the environment
        :param env:
        :return:
        """
        detected = []  # object box(xc, yc, w, h)
        detected_ids = []
        object_shapes = np.array([info.shape for info in env.obj_info])
        for obj_id, obj in env.objects.items():
            if not self.within(tuple[float, float](obj[:2])):
                continue
            obj = obj.copy()
            obj[:2] += np.random.randn(2) * noise_pos
            close_idx = np.isclose(obj[2:], object_shapes).all(axis=1)
            if not close_idx.any():
                continue
            shape = object_shapes[close_idx][0]
            obj[2:] = shape + np.random.randn(2) * noise_shape
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
                preds[:, 2:], object_shape, atol=noise_shape * 10
            ).all(axis=1)
            probs[close_idxs, i] = np.random.uniform(0.7, 1, size=close_idxs.sum())
        for prob in probs:
            no_assign_prob = 1 - np.sum(prob)
            if no_assign_prob:
                w = np.random.rand(len(prob[prob == 0]))
                w /= w.sum()
                prob[prob == 0] = w * no_assign_prob
        cls = probs.argmax(axis=1)
        # 添加协同检测(x, y, w, h, cls, prob)
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
