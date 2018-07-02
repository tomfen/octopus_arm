from math import acos, sqrt, atan2, pi
import math
from numpy import var


class Features:
    def __init__(self):
        self.dim = 42
        self.__goal = (9, -1)

    @staticmethod
    def distance(x1, y1, x2, y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

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
