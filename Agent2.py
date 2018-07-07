from collections import namedtuple
from itertools import product

from typing import NamedTuple, List

import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.models import load_model
from keras.optimizers import SGD

from Replay import Replay
from features import Features, MinFeatures

np.set_printoptions(precision=3, suppress=True)

class MuscleGroup(NamedTuple):
    dorsal : float = 0.0
    transverse : float = 0.0
    venrtal : float = 0.0


class ThreeActionsMapper(NamedTuple):
    #action 0 -> bend downwards
    #action 1 -> bend upwards
    #action 2 -> extend
    segments_count : int = 2
    action_vector_size = 30
    action_vector : np.array = np.zeros(action_vector_size)
    possible_actions_count : int = 3
    actions_dict = None

    def get_actions_dict(self):
        all_configurations = list(product(list(range(0,self.possible_actions_count)), repeat=self.segments_count))

        actions_dict = {}
        for idx, conf in enumerate(all_configurations):
            actions_dict[idx] = conf

        self.actions_dict = actions_dict
        return actions_dict

    # def get_segments(self, action_vector):
    #     muscle_groups = []
    #     for group_idx in range(0, 10):
    #         group_start = group_idx*3
    #         muscle_groups.append(MuscleGroup(action_vector[group_start], action_vector[group_start+1], action_vector[group_start+2]))
    #
    #     expected_segment_size = int(self.action_vector_size / self.segments_count)
    #     segments = []
    #     segment = []
    #     for m in muscle_groups:
    #         if len(segment)  < expected_segment_size:
    #             segment.append(m)
    #         else:
    #             segments.append(segment)
    #             segment = []
    #     return  segments

    def __to_action_vector(self, segments : List[MuscleGroup]):

        action_vector =  np.zeros(self.action_vector_size)

        for idx, m_g in enumerate(segments):
            action_vector[idx] = m_g.dorsal
            action_vector[idx+1] = m_g.transverse
            action_vector[idx+2] = m_g.venrtal

        return action_vector


    def create_action(self, action_idx : int):
        action_dict = None
        if self.actions_dict is None:
            action_dict = self.get_actions_dict()
        else:
            action_dict = self.actions_dict

        action_sequence = action_dict[action_idx]

        muscle_groups = []
        for  muscle_group_val in action_sequence:
            if muscle_group_val == 0:
                muscle_groups.append(MuscleGroup(-1,0,1))
            elif muscle_group_val == 1:
                muscle_groups.append(MuscleGroup(1, 0, -1))
            elif muscle_group_val == 2:
                muscle_groups.append(MuscleGroup(0,1,0))
        return self.__to_action_vector(muscle_groups)







# noinspection PyBroadException
class Agent:
    # name should contain only letters, digits, and underscores (not enforced by environment)
    __name = 'second'
    __net_name = "second_net.data"

    partFactory = namedtuple("Part", ["upper_x", "upper_y", "u_x_velocity", "u_y_velocity",
                                      "lower_x", "lower_y", "l_x_velocity", "l_y_velocity"])


    def describe_parts(self, state):
        parts = []
        if type(state) != list:
            state_as_list = state.flatten().tolist()
        else:
            state_as_list = state
        for idx in range(1, 11):
            lower = 2 + (idx - 1) * 4
            upper = lower + 40
            p = self.partFactory(
                state_as_list[lower + 0],
                state_as_list[lower + 1],
                state_as_list[lower + 2],
                state_as_list[lower + 3],
                state_as_list[upper + 0],
                state_as_list[upper + 1],
                state_as_list[upper + 2],
                state_as_list[upper + 3],
            )
            parts.append(p)

        return parts


    def __init__(self, stateDim, actionDim, agentParams):
        self.__stateDim = stateDim
        self.__actionDim = actionDim
        self.__action = np.random.random(actionDim)
        self.__step = 0

        self.__alpha = 0.001
        self.__gamma = 0.9
        self.__decision_every = 6
        self.__explore_probability = 0.3
        self.__max_replay_samples = 20

        self.__features = MinFeatures()
        self.__previous_action = None
        self.__current_out = None
        self.__previous_out = None
        self.__previous_meta_state = None
        self.__previous_state = None

        self.__test = agentParams[0] if agentParams else None
        self.__exploit = False

        self.__segments = 2
        self.__action_space_size = 3 ** self.__segments

        self.__action_mapper = ThreeActionsMapper()
        try:
            self.__net = load_model(self.__net_name)
        except:
            print('Creating new model')
            self.__net = Sequential([
                Dense(50, activation='elu', input_dim=self.__features.dim),
                Dense(30, activation='elu'),
                Dense(self.__action_space_size, activation='linear'),
            ])

        self.__net.compile(optimizer=SGD(lr=self.__alpha), loss='mean_squared_error', sample_weight_mode='temporal')

        try:
            self.__replay = Replay.load('replay.data')
        except Exception as a:
            self.__replay = Replay(self.__action_space_size)

        self.__replay_X = []
        self.__replay_Y = []

    def start(self, state):
        self.__choose_action(state)

        return self.__action

    def step(self, reward, state):

        described = self.describe_parts(state)

        self.__choose_action(state)
        return self.__action

    def end(self, reward):
        if not self.__exploit:
            self.__update_q(reward, reward)
            self.__replay.submit(self.__test, (self.__replay_X, self.__replay_Y), self.__step)
            self.__net.save(self.__net_name)
            self.__replay.save('replay')

    def cleanup(self):
        pass

    def getName(self):
        return self.__name

    def __choose_action(self, state):

        if self.__exploit or self.__explore_probability < np.random.random():
            # take best action
            feature_vector = self.__features.min_features(state)
            q_values_of_actions = self.__net.predict(feature_vector)
            action = np.argmax(q_values_of_actions)
        else:
            # take random action
            action = np.random.randint(0, self.__action_space_size)

        self.__action = self.__action_mapper.create_action(action)

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

        self.__net.fit([self.__previous_meta_state], [teach_out.reshape(1, self.__action_space_size, 1)], verbose=0)

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
