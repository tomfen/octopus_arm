import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
from keras.optimizers import SGD

from features import Features


class Agent:
    # name should contain only letters, digits, and underscores (not enforced by environment)
    __name = 'Based_Agent'

    def __init__(self, stateDim, actionDim, agentParams):
        self.__stateDim = stateDim
        self.__actionDim = actionDim
        self.__action = np.random.random(actionDim)
        self.__gamma = 0.9
        self.__decision_every = 3
        self.__step = 1

        self.__features = Features()
        self.__previous_action = None
        self.__previous_out = None
        self.__previous_meta_state = None
        self.__previous_state = None
        self.__best_distance = None

        try:
            self.__net = load_model('net')
        except:
            print('Creating new model')
            self.__net = Sequential([
                Dense(64, batch_size=1, input_dim=self.__features.dim),
                Activation('tanh'),
                Dense(16)
            ])

        self.__net.compile(optimizer=SGD(lr=0.01), loss='mean_squared_error')

    def start(self, state):
        self.__best_distance = 999999

        meta_state = np.asarray(self.__features.getFeatures(state), dtype='float').reshape((1, self.__features.dim))
        out = self.__net.predict_proba([meta_state], batch_size=1)[0]

        best_meta_action = np.argmax(out)

        self.__previous_action = best_meta_action
        self.__previous_meta_state = meta_state
        self.__previous_state = state
        self.__previous_out = out

        self.__meta_to_action(best_meta_action)

        return self.__action

    def step(self, reward, state):
        self.__previous_state = state

        self.__step += 1
        if (self.__step != self.__decision_every) and reward != 10:
            return self.__action
        self.__step = 0

        if self.__features.distMin(state) < self.__best_distance:
            reward += 9
            self.__best_distance = self.__features.distMin(state)

        teach_out = self.__previous_out
        teach_out[self.__previous_action] = reward + self.__gamma * teach_out[self.__previous_action]

        self.__net.fit([self.__previous_meta_state], [teach_out.reshape(1, 16)], verbose=0)

        meta_state = np.asarray(self.__features.getFeatures(state), dtype='float').reshape((1, self.__features.dim))
        out = self.__net.predict_proba([meta_state], batch_size=1)[0]

        best_meta_action = np.argmax(out)

        self.__previous_action = best_meta_action
        self.__previous_meta_state = meta_state
        self.__previous_out = out

        self.__meta_to_action(best_meta_action)

        if reward == 10:
            self.__save_net()

        return self.__action

    def end(self, reward):
        pass

    def cleanup(self):
        pass

    def getName(self):
        return self.__name

    def __meta_to_action(self, meta):
        upper = meta % 4
        lower = meta // 4

        self.__action[:] = 0

        if upper == 1:
            self.__action[0:15:3] = 1
        elif upper == 2:
            self.__action[1:15:3] = 1
        elif upper == 3:
            self.__action[2:15:3] = 1

        if lower == 1:
            self.__action[15:30:3] = 1
        elif lower == 2:
            self.__action[16:30:3] = 1
        elif lower == 3:
            self.__action[17:30:3] = 1

    def __save_net(self):
        self.__net.save('net')
