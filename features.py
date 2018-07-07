from math import atan2, pi
import math
from typing import NamedTuple, Tuple


class MinFeatures(NamedTuple):
    dim : int = 3
    goal : Tuple[int, int] = (9, -1)

    def min_features(self, state):
        features = []
        features.append(self._tip_vertical_dist_from_goal(state))
        features.append(self._tip_horizontal_dist_from_goal(state))
        features.append(self.tip_arm_above_or_below(state))
        return (len(features), features)

    def _tip_vertical_position(self, state):
        u_10_y = state[39]
        l_10_y = state[79]
        avg = float(u_10_y + l_10_y) / float(2)
        return avg

    def _tip_horizontal_position(self, state):
        u_10_x = state[38]
        l_10_x = state[78]
        avg = float(u_10_x + l_10_x) / float(2)
        return avg

    def _tip_vertical_dist_from_goal(self, state):
        tip_vert = self._tip_vertical_position(state)
        _, goal_vert = self.goal

        return (tip_vert - goal_vert)

    def _tip_horizontal_dist_from_goal(self, state):
        tip_hor = self._tip_horizontal_position(state)
        goal_hor, _ = self.goal

        return (tip_hor - goal_hor)

    # def tip_horizontal_dist_from_origin(self, state):
    #     u_1_x = state[2]
    #     l_1_x = state[42]
    #     avg_1 = float(u_1_x + l_1_x) / float(2)
    #
    #     u_10_x = state[38]
    #     l_10_x = state[78]
    #     avg_10 = float(u_10_x + l_10_x) / float(2)
    #     return (avg_10 - avg_1)
    #
    # def tip_vertical_dist_from_origin(self, state):
    #     u_1_y = state[3]
    #     l_1_y = state[43]
    #     avg_1 = float(u_1_y + l_1_y) / float(2)
    #
    #     avg_10 = self.tip_vertical_position(state)
    #     return (avg_10 - avg_1)

    def tip_arm_above_or_below(self, state):
        avg = self._tip_vertical_position(state)

        _, goal_y = self.__goal
        if avg >= goal_y:
            return 1
        else:
            return -1

class Features:
    def __init__(self):
        self.dim = 42
        self.__goal = (9, -1)

    @staticmethod
    def distance(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_features(self, state):
        features = list(state[:2])

        for i in range(2, 42, 2):
            x1 = state[i]
            y1 = state[i+1]
            x2 = state[i+40]
            y2 = state[i+41]

            xg, yg = self.__goal

            if i % 4 == 2:
                x = xg - (x1+x2)/2
                y = yg - (y1+y2)/2
            else:
                x = (x1+x2)/2
                y = (y1+y2)/2

            a = atan2(y1-y2, x1-x2) - pi/2

            sin = math.sin(-a)
            cos = math.cos(-a)

            x_ = x * cos - y * sin
            y_ = y * cos + x * sin

            features.append(x_/9)
            features.append(y_/9)
        return features

    def min_dist(self, state):
        xg, yg = self.__goal
        return min(self.distance(state[42 + i*4], state[42 + i*4 + 1], xg, yg) for i in range(7, 10))




