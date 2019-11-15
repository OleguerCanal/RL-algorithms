from abc import ABC, abstractmethod
import numpy as np
import copy
import random
import itertools
from abstract_mdp import MDP
from matplotlib import pyplot as plt

class Problem1B(MDP):
    def __init__(self):
        self.map = np.ones((7, 8))
        self.map[0:4, 2] = 0
        self.map[5, 1:7] = 0
        self.map[5, 1:7] = 0
        self.map[6, 4] = 0
        self.map[1:4, 5] = 0
        self.map[2, 6:8] = 0

        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

    def get_states(self):
        player_states = tuple(np.swapaxes(np.where(self.map == 1), 0, 1))
        minotaur_states = tuple(np.swapaxes(np.where(self.map < 2), 0, 1))
        states =  list(itertools.product((player_states), (minotaur_states)))
        tuple_states = []
        for state in states:
            tuple_states.append((tuple(state[0]), tuple(state[1])))
        return tuple_states

    def get_actions(self, state):
        possible_actions = []
        for action in self.actions:
            if self.__valid_action(state[0], action, player = True):
                possible_actions.append(action)
        return possible_actions

    def __valid_action(self, position, action, player = True):
        next_pos = tuple(np.array(position) + np.array(action))
        if next_pos[0] < 0 or next_pos[0] >= self.map.shape[0] or\
            next_pos[1] < 0 or next_pos[1] >= self.map.shape[1]:
            return False
        if player and self.map[next_pos] == 0:
            return False
        return True

    def get_transitions(self, state, action):
        player_pos = tuple(np.array(state[0]) + np.array(action))
        if player_pos == (6, 5):
            return None, 0
        minotaur_positions = []
        for action in self.actions:
            if self.__valid_action(state[1], action, player=False):
                mino_pos = tuple(np.array(state[1]) + np.array(action))
                minotaur_positions.append(mino_pos)
        prob = 1./len(minotaur_positions)
        for mino_pos in minotaur_positions:
            new_state = (player_pos,  mino_pos)
            yield new_state, prob

    def get_reward(self, state, action):
        player_pos = tuple(np.array(state[0]) + np.array(action))
        new_state = (tuple(player_pos), tuple(state[1]))
        if tuple(new_state[0]) == tuple(new_state[1]):
            return -1000
        if tuple(new_state[0]) == (6, 5):
            return 0
        return -1

    def get_heat_map(self, values_dict, minotaur_pos):
        minotaur_pos = tuple(minotaur_pos)
        heatmap = np.zeros(self.map.shape)
        for state in values_dict:
            if state[1] == minotaur_pos:
                heatmap[state[0]] = values_dict[state]
        return heatmap

if __name__ == "__main__":
    problem = Problem1B()

    states = problem.get_states()
    initial_values = {state : 0 for state in states}

    value_hist = problem.dynamic_programming(initial_values = initial_values, T = 20)
    # value_hist = problem.value_iteration(initial_values = initial_values)

    mino_states = tuple(np.swapaxes(np.where(problem.map < 2), 0, 1))
    mino_states = [(3, 1), (2, 3), (4, 7)]
    for mino_pos in mino_states:
        heatmap = problem.get_heat_map(value_hist[-1], mino_pos)
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.show()