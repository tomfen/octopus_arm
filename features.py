from math import atan2, pi
import math

class Features:
    def __init__(self):
        self.__food = [9, -1]
        self.__foodLength = self.distanceBetweenPoints(0, 0, self.__food[0], self.__food[1])
        self.dim = 42
        self.__goal = (9, -1)
        pass

    def distanceBetweenPoints(self, x1, y1, x2, y2):
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        d = d ** (0.5)
        return d

    def getFeatures(self, state):
        e= [state[0], state[1]] + [state[i]+state[i+40] for i in range(2, 42, 2)]

        for i in range(1,5):
            e[2+4*i] -= 2*i

        S = list(state[:2])

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

            S.append(x_/9)
            S.append(y_/9)
        return S