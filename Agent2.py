import datetime
import json
import math
import os
import random
import sqlite3
from collections import namedtuple
from itertools import product

from typing import NamedTuple, List, Tuple

import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.models import load_model
from keras.optimizers import SGD

from Replay import Replay
from config import get_counter
from features import Features, MinFeatures

np.set_printoptions(precision=3, suppress=True)

def default(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError



class HistoryTuple(NamedTuple):
    previous_raw_state : np.array
    previous_feature_vector : np.array
    action : np.array
    action_idx : int
    next_raw_state : np.array
    next_feature_vector : np.array
    reward : float
    started_at : int = datetime.datetime.utcnow().timestamp()

    def to_json(self):
        to_be_serialized = HistoryTuple(previous_raw_state=self.previous_raw_state,
                                        previous_feature_vector=self.previous_feature_vector,
                                        action=self.action.tolist(),
                                        action_idx=self.action_idx,
                                        next_raw_state=self.next_raw_state,
                                        next_feature_vector=self.next_feature_vector,
                                        reward=self.reward,
                                        started_at=self.started_at
                                        )

        return json.dumps(to_be_serialized, default=default)

    @staticmethod
    def from_unpacked(unpacked_list):
        json_element = unpacked_list
        previous_raw_state = json_element[0]
        previous_feature_vector = json_element[1]
        action = json_element[2]
        action_idx = json_element[3]
        next_raw_state = json_element[4]
        next_feature_vector = json_element[5]
        reward = json_element[6]
        started_at = json_element[7]
        correct =  HistoryTuple(previous_raw_state=previous_raw_state,
                                        previous_feature_vector=previous_feature_vector,
                                        action=action,
                                        action_idx=action_idx,
                                        next_raw_state=next_raw_state,
                                        next_feature_vector=next_feature_vector,
                                        reward=reward,
                                        started_at=started_at
                                        )
        return correct

    @staticmethod
    def from_json(json_string):
        json_element = json.loads(json_string)
        return HistoryTuple.from_unpacked(json_element)



class MuscleGroup(NamedTuple):
    dorsal : float = 0.0
    transverse : float = 0.0
    venrtal : float = 0.0


def get_actions_dict(possible_actions_count, segments_count):
    all_configurations = list(product(list(range(0, possible_actions_count)), repeat=segments_count))

    actions_dict = {}
    for idx, conf in enumerate(all_configurations):
        actions_dict[idx] = conf

    return actions_dict

class ThreeActionsMapper(NamedTuple):
    #action 0 -> bend downwards
    #action 1 -> bend upwards
    #action 2 -> extend
    segments_count : int = 2
    action_vector_size = 30
    action_vector : np.array = np.zeros(action_vector_size)
    possible_actions_count : int = 3
    actions_dict = get_actions_dict(possible_actions_count, segments_count)




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

    def __to_action_vector(self, segments : List[MuscleGroup]) -> np.array:

        action_vector =  np.zeros(self.action_vector_size)

        for idx, m_g in enumerate(segments):
            offset = 3*idx
            action_vector[offset] = m_g.dorsal
            action_vector[offset+1] = m_g.transverse
            action_vector[offset+2] = m_g.venrtal

        return action_vector


    def create_action_vector(self, action_idx : int):

        action_dict = self.actions_dict

        action_sequence = action_dict[action_idx]
        segment_length = int(10 / len(action_sequence))
        muscle_groups = []
        for  muscle_group_val in action_sequence:
            for i in range(0, segment_length):
                if muscle_group_val == 0:
                    muscle_groups.append(MuscleGroup(-1,0,1))
                elif muscle_group_val == 1:
                    muscle_groups.append(MuscleGroup(1, 0, -1))
                elif muscle_group_val == 2:
                    muscle_groups.append(MuscleGroup(0,1,0))
        return self.__to_action_vector(muscle_groups)


def namedtuple_factory(cursor, row):
    """
    Usage:
    con.row_factory = namedtuple_factory
    """
    fields = [col[0] for col in cursor.description]
    Row = namedtuple("Row", fields)
    return Row(*row)




# noinspection PyBroadException
class Agent:
    # name should contain only letters, digits, and underscores (not enforced by environment)
    __name = 'second'
    _generation = 0
    _counter = get_counter()
    __upper_net_name = "lower_net_generation_{}.data".format(_generation)
    __lower_net_name = "upper_net_generation_{}.data".format(_generation)

    partFactory = namedtuple("Part", ["upper_x", "upper_y", "u_x_velocity", "u_y_velocity",
                                      "lower_x", "lower_y", "l_x_velocity", "l_y_velocity"])

    def custom_reward(self, reward, state):
      return reward + (1/math.sqrt(self.__features._tip_horizontal_dist_from_goal(state) ** 2 + self.__features._tip_vertical_dist_from_goal(state) **2))

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

    # def load_latest_upper_net(self):
    #     local_path = __file__
    #     files = os.listdir(local_path)
    #     nets = [f for f in files if "upper_net" in f]
    #     if len(nets):
    #
    #
    # def try_to_load_upper(self, counter):
    #     try:
    #         self.__upper_net = load_model("upper_net_{}_.data".format((self._counter)))
    #         return True
    #     except:
    #        return False
    #
    # def try_to_load_lower(self, counter):
    #     lower_name = "lower_net_{}_.data".format((self._counter - 1))
    #     try:
    #         self.__lower_net = load_model(lower_name)
    #         return True
    #     except:
    #        return False

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

        self.__last_history_tuple = None
        self.__current_map_category = None

        self._net_dict = {}
        self._lower_net = None
        self._upper_net = None

        try:
            self._lower_net = load_model(self.__lower_net_name)
        except:
            print('{} not found, creating new lower model'.format(self.__lower_net_name))
            self._lower_net = Sequential([
                Dense(50, activation='elu', input_dim=self.__features.dim),
                Dense(30, activation='elu'),
                Dense(self.__action_space_size, activation='linear'),
            ])

        self._lower_net.compile(optimizer=SGD(lr=self.__alpha), loss='mean_squared_error')

        try:
            self._upper_net = load_model(self.__upper_net_name)
        except:
            print('{} not found, creating new upper model'.format(self.__upper_net_name))
            self._upper_net = Sequential([
                Dense(50, activation='elu', input_dim=self.__features.dim),
                Dense(30, activation='elu'),
                Dense(self.__action_space_size, activation='linear'),
            ])

        self._upper_net.compile(optimizer=SGD(lr=self.__alpha), loss='mean_squared_error')

        # if self.try_to_load_upper(self._counter -1):
        #     print("found upper_net_{}".format(self._counter - 1))
        #     pass
        # else:
        #     if self.try_to_load_upper(self._counter):
        #         print("found_upper_net_{}".format(self._counter))
        #         pass
        #     else:
        #         print("did not found upper_net_{}".format(self._counter))
        #         upper_name = "upper_net_{}_.data".format((self._counter))
        #         print('Creating new model, {} not found'.format(upper_name))
        #         self.__upper_net = Sequential([
        #             Dense(50, activation='elu', input_dim=self.__features.dim),
        #             Dense(30, activation='elu'),
        #             Dense(self.__action_space_size, activation='linear'),
        #         ])
        #
        #     self.__upper_net.compile(optimizer=SGD(lr=self.__alpha), loss='mean_squared_error')
        #
        # if self.try_to_load_lower(self._counter - 1):
        #
        #     pass
        # else:
        #     print("did not found lower_net_{}".format(self._counter - 1))
        #     if self.try_to_load_lower(self._counter):
        #         pass
        #     else:
        #         print("did not found lower_net_{}".format(self._counter))
        #         lower_name = "lower_net_{}_.data".format((self._counter))
        #         print('Creating new model, {} not found'.format(lower_name))
        #         self.__upper_net = Sequential([
        #             Dense(50, activation='elu', input_dim=self.__features.dim),
        #             Dense(30, activation='elu'),
        #             Dense(self.__action_space_size, activation='linear'),
        #         ])
        #
        #     self.__lower_net.compile(optimizer=SGD(lr=self.__alpha), loss='mean_squared_error')

    def start(self, state):
        self._net_dict[-1] = self._lower_net
        self._net_dict[1] = self._upper_net
        conn = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "history"))
        conn.row_factory = namedtuple_factory
        conn.execute('''
            create table if not exists sarsa
(
	entry_id INTEGER
		primary key
		 autoincrement,
	data_json text not null,
	started_at float default 0 not null,
	upper_or_lower int default 0 not null
);''')


        conn.execute('''create index if not exists sarsa_started_at_index
            on sarsa (started_at)''')


        conn.execute('''create index if not exists sarsa_upper_or_lower_index
            on sarsa (upper_or_lower)
        ;''')


        conn.commit()
        self.__conn = conn

        # self.train() debug

        self.__current_map_category = self.__features.tip_arm_above_or_below(state)
        if self.__current_map_category == 1:
            print("should be using upper")
        if self.__current_map_category == -1:
            print("should be using lower")

        action, action_idx = self.__choose_action(state)

        self.__last_history_tuple = HistoryTuple(previous_raw_state=state,
                                                 previous_feature_vector=self.__features.min_features(state),
                                                 action=action,
                                                 action_idx=action_idx,
                                                 next_raw_state=None,
                                                 next_feature_vector=None,
                                                 reward=None,
                                                 )

        return self.__action

    def step(self, reward, state):

        #described = self.describe_parts(state) # for debug

        previous_tuple : HistoryTuple = self.__last_history_tuple
        updated_history_tuple = HistoryTuple(
            previous_raw_state=previous_tuple.previous_raw_state,
            previous_feature_vector=previous_tuple.previous_feature_vector,
            action=previous_tuple.action,
            action_idx=previous_tuple.action_idx,
            next_raw_state=state,
            next_feature_vector=self.__features.min_features(state),
            reward=self.custom_reward(reward, state),
            started_at=previous_tuple.started_at
        )
        data = updated_history_tuple.to_json()

        self.__conn.execute('''insert into sarsa (data_json, started_at, upper_or_lower) values (json('{}'), {}, {})'''.format (data, updated_history_tuple.started_at, self.__current_map_category))
        self.__conn.commit()

        action, action_idx = self.__choose_action(state)
        self.__last_history_tuple = HistoryTuple(previous_raw_state=state,
                                                 previous_feature_vector=self.__features.min_features(state),
                                                 action=action,
                                                 action_idx=action_idx,
                                                 next_raw_state=None,
                                                 next_feature_vector=None,
                                                 reward=None,
                                                 started_at=previous_tuple.started_at)
        return self.__action

    def end(self, reward):

        for i in range(0, 5):
            self.train(self.__current_map_category)


        if not self.__exploit:
            self._lower_net.save(self.__lower_net_name)
            self._upper_net.save(self.__upper_net_name)

            # self._lower_net.save("_after_level_{}".format(self.__test))
            # self._upper_net.save("_after_level_{}".format(self.__test))

    def cleanup(self):
        pass

    def getName(self):
        return self.__name

    def get_past_moves(self, category, how_far_into_past=5000, how_many=200) -> List[HistoryTuple]:
        conn = self.__conn

        past = list(conn.execute('''
            select
            -- entry_id,
            data_json
            --started_at
            from sarsa
            where upper_or_lower = {}
            order by entry_id desc
            limit {}

        '''.format(category, how_far_into_past)))

        past = [json.loads(p[0]) for p in past]
        #unpacked = [p[0] for p in past]
        tuples = [HistoryTuple.from_unpacked(p) for p in past]
        return random.sample(tuples, min(how_many, len(tuples)))


    def train(self, category, how_far_into_past=2000, how_many=500):

        tuples = self.get_past_moves(category, how_far_into_past, how_many)

        batch_inputs = []
        batch_targets = []
        for previous_raw_state, previous_feature_vector, action, action_idx, next_raw_state, next_feature_vector, reward, started_at in tuples:


            predicted_qs_of_old_state = self._net_dict[self.__current_map_category].predict(np.array(previous_feature_vector).reshape(1, self.__features.dim))
            predicted_qs_of_new_state = self._net_dict[self.__current_map_category].predict(np.array(next_feature_vector).reshape(1, self.__features.dim))

            best_future_action_idx = np.argmax(predicted_qs_of_new_state)

            target = reward + self.__gamma * predicted_qs_of_new_state[0][best_future_action_idx]

            updated_qs_of_old_state = predicted_qs_of_old_state
            updated_qs_of_old_state[0][action_idx] = target

            batch_inputs.append(previous_feature_vector)
            batch_targets.append(updated_qs_of_old_state)

        self._net_dict[self.__current_map_category].fit(np.vstack([np.array(el) for el in batch_inputs]), np.vstack([np.array(el) for el in batch_targets]), batch_size=how_many, verbose=0)





    def __choose_action(self, state) -> Tuple[np.array, int]:

        if self.__exploit or self.__explore_probability < np.random.random():
            # take best action
            feature_vector = self.__features.min_features(state)
            q_values_of_actions = self._net_dict[self.__current_map_category].predict(np.array(feature_vector).reshape(1, self.__features.dim))
            action_idx = np.argmax(q_values_of_actions)
        else:
            # take random action
            action_idx = np.random.randint(0, self.__action_space_size)

        action_selected = self.__action_mapper.create_action_vector(action_idx)
        self.__action = action_selected
        return (action_selected, action_idx)

    # def __update_q(self, reward, max_q):
    #     teach_out = self.__previous_out
    #     teach_out[self.__previous_action] = reward + self.__gamma * max_q
    #
    #     # sampling from infinite stream
    #     if len(self.__replay_X) < self.__max_replay_samples:
    #         self.__replay_X.append(self.__previous_meta_state)
    #         self.__replay_Y.append((teach_out[self.__previous_action], self.__previous_action))
    #
    #     elif np.random.random() < self.__max_replay_samples/self.__step:
    #         to_replace = np.random.randint(0, self.__max_replay_samples)
    #         self.__replay_X[to_replace] = self.__previous_meta_state
    #         self.__replay_Y[to_replace] = (teach_out[self.__previous_action], self.__previous_action)
    #
    #     self.__net.fit([self.__previous_meta_state], [teach_out.reshape(1, self.__action_space_size, 1)], verbose=0)
    #
    #     replay_x, replay_y, replay_w = self.__replay.get_training()
    #     if replay_x:
    #         data = list(zip(replay_x, replay_y, replay_w))
    #         np.random.shuffle(data)
    #         for x, y, w in data:
    #             self.__net.fit([x], [y], sample_weight=[w], verbose=0)
