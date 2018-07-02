import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.models import load_model
from keras.optimizers import SGD

from Replay import Replay
from features import Features


# noinspection PyBroadException
class Agent:
    # name should contain only letters, digits, and underscores (not enforced by environment)
    __name = 'Based_Agent'

    def __init__(self, stateDim, actionDim, agentParams):
        self.__stateDim = stateDim
        self.__actionDim = actionDim
        self.__action = np.random.random(actionDim)
        self.__step = 0

        self.__alpha = 0.001
        self.__gamma = 0.9
        self.__decision_every = 6
        self.__explore_probability = 0.2
        self.__max_replay_samples = 20

        self.__features = Features()
        self.__previous_action = None
        self.__current_out = None
        self.__previous_out = None
        self.__previous_meta_state = None
        self.__previous_state = None

        self.__test = agentParams[0] if agentParams else None
        self.__exploit = False

        self.__segments = 2
        self.__actions = 3**self.__segments

        try:
            self.__net = load_model('net')
        except:
            print('Creating new model')
            self.__net = Sequential([
                Dense(50, activation='elu', input_dim=self.__features.dim),
                Dense(30, activation='elu'),
                Dense(self.__actions),
                Reshape((self.__actions, 1))
            ])

        self.__net.compile(optimizer=SGD(lr=self.__alpha), loss='mean_squared_error', sample_weight_mode='temporal')

        try:
            self.__replay = Replay.load('replay')
        except Exception as a:
            self.__replay = Replay(self.__actions)

        self.__replay_X = []
        self.__replay_Y = []

    def start(self, state):
        self.__previous_state = state

        self.__choose_action(state)

        self.__previous_out = self.__current_out

        return self.__action

    def step(self, reward, state):
        self.__previous_state = state

        self.__step += 1
        if self.__step % self.__decision_every != 0:
            return self.__action

        self.__choose_action(state)

        if not self.__exploit:
            max_q = self.__current_out[np.argmax(self.__current_out)]
            self.__update_q(reward - self.__features.min_dist(state) / 100, max_q)

        self.__previous_out = self.__current_out

        return self.__action

    def end(self, reward):
        if not self.__exploit:
            self.__update_q(reward, reward)
            self.__replay.submit(self.__test, (self.__replay_X, self.__replay_Y), self.__step)
            self.__net.save('net')
            self.__replay.save('replay')

    def cleanup(self):
        pass

    def getName(self):
        return self.__name

    def __choose_action(self, state):
        meta_state = np.asarray(self.__features.get_features(state), dtype='float').reshape((1, self.__features.dim))
        out = self.__net.predict_proba([meta_state], batch_size=1)[0].flatten()

        self.__current_out = out

        if self.__exploit or self.__explore_probability < np.random.random():
            # take best action
            action = np.argmax(out)
        else:
            # take random action
            action = np.random.randint(0, self.__actions)

        self.__previous_action = action
        self.__previous_meta_state = meta_state

        self.__meta_to_action(action)

    def __update_q(self, reward, max_q):
        teach_out = self.__previous_out
        teach_out[self.__previous_action] = reward + self.__gamma * max_q

        # sampling from infinite stream
        if len(self.__replay_X) < self.__max_replay_samples:
            self.__replay_X.append(self.__previous_meta_state)
            self.__replay_Y.append((teach_out[self.__previous_action], self.__previous_action))

        elif np.random.random() < self.__max_replay_samples/self.__step:
            to_replace = np.random.randint(0, self.__max_replay_samples)
            self.__replay_X[to_replace] = self.__previous_meta_state
            self.__replay_Y[to_replace] = (teach_out[self.__previous_action], self.__previous_action)

        self.__net.fit([self.__previous_meta_state], [teach_out.reshape(1, self.__actions, 1)], verbose=0)

        replay_x, replay_y, replay_w = self.__replay.get_training()
        if replay_x:
            data = list(zip(replay_x, replay_y, replay_w))
            np.random.shuffle(data)
            for x, y, w in data:
                self.__net.fit([x], [y], sample_weight=[w], verbose=0)

    def __meta_to_action(self, meta):

        self.__action[:] = 0

        for segment in range(self.__segments):
            segment_action = meta % 3

            muscle_start = 30 * segment // self.__segments
            muscle_stop = 30 * (segment+1) // self.__segments

            if segment_action == 0:
                self.__action[muscle_start:muscle_stop:3] = 1

            if segment_action == 1:
                self.__action[muscle_start+1:muscle_stop:3] = 1

            if segment_action == 2:
                self.__action[muscle_start+2:muscle_stop:3] = 1

            meta //= 3
