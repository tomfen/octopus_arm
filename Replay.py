import numpy as np
import pickle
from glob import iglob
import os


class Replay:

    def __init__(self, actions):
        self.__scores = {}
        self.__data_x = {}
        self.__data_y = {}
        self.__data_w = {}
        self.__actions = actions

    def submit(self, level, data, steps):
        if not data[0]:
            return

        if level in self.__scores:
            score = self.__scores[level]
            if steps <= score:
                self.__add_samples(level, data, steps)
        else:
            self.__add_samples(level, data, steps)

    def __add_samples(self, level, data, steps):
        self.__scores[level] = steps
        self.__data_x[level] = np.concatenate(data[0], 0)

        Y = np.zeros((len(data[1]), self.__actions, 1), np.float)
        W = np.zeros_like(Y)

        for i in range(len(data[1])):
            q, index = data[1][i]
            Y[i][index][0] = q
            W[i][index] = 0.01

        self.__data_y[level] = Y

    def get_training(self):
        return list(self.__data_x.values()), list(self.__data_y.values()), list(self.__data_w.values())

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            replay = pickle.load(file)

        return replay


