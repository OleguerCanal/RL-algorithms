from abc import ABC, abstractmethod
import numpy as np
import copy
import random
import itertools

class Problem1B(MDP):
    def __init__(self):
        self.map = np.ones((7, 8))
        self.map[0:4, 2] = 0
        self.map[5, 1:7] = 0
        self.map[5, 1:7] = 0
        self.map[6, 4] = 0
        self.map[1:4, 5] = 0
        self.map[2, 6:8] = 0

    def get_states(self):
        player_states = list(np.where(self.map == 1))
        minotaur_states = list(np.where(self.map < 2))
        return list(itertools.product(player_states, minotaur_states))

    def get_actions(self, state = None):
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def get_transition_prob(self, next_state, state, action):
        if next_state[0] != tuple(np.array(state[0]) + np.array(action)):
            return 0
        for i in range(2):
            if next_state[i][0] < 0 or next_state[i][0] >= self.map.shape[0] or\
                next_state[i][1] < 0 or next_state[i][1] >= self.map.shape[1]:
                return 0
        if self.map[next_state[0]] == 0:
            return 0
        if state[0] == (6, 6):
            return 0
        # if state[0] == (0, 0) or state[0] == (6, 0) or state[0] == (0, 7) or state[0] == (6, 7):
        return 0.25


    def get_reward(self, state, action):
        new_state = tuple(np.array(state) + np.array(action))
        if new_state[0] < 0 or new_state[0] >= self.map.shape[0] or\
            new_state[1] < 0 or new_state[1] >= self.map.shape[1]:
            return float('-inf')
        if self.map[new_state] == 0:
            return float('-inf')
        if state[0] == state[1]:
            return float('-inf')
        return -1