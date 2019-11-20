from abc import ABC, abstractmethod
import numpy as np
import copy
import random
import itertools
from abstract_mdp import MDP
from matplotlib import pyplot as plt
from tqdm import tqdm

class Problem1B(MDP):
    def __init__(self):
        self.map = np.ones((5, 8))
        # self.map[0:4, 2] = 0
        # self.map[5, 1:7] = 0
        # self.map[5, 1:7] = 0
        # self.map[6, 4] = 0
        # self.map[1:4, 5] = 0
        # self.map[2, 6:8] = 0
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        self.start = (0, 0)
        self.goal = (2, 7)

        # Internal control variables
        self.__dead_state = "dead"
        self.__free_state = "free"
        self.__die_action = "die"
        self.__escape_action = "escape"

    def get_states(self):
        player_states = tuple(np.swapaxes(np.where(self.map == 1), 0, 1))
        minotaur_states = tuple(np.swapaxes(np.where(self.map < 2), 0, 1))
        states =  list(itertools.product((player_states), (minotaur_states)))
        tuple_states = []
        for state in states:
            tuple_states.append((tuple(state[0]), tuple(state[1])))
        tuple_states.append(self.__dead_state)
        tuple_states.append(self.__free_state)
        return tuple_states
    
    def __valid_action(self, position, action, player = True):
        if action == self.__escape_action or action == self.__die_action:
            return True
        next_pos = tuple(np.array(position) + np.array(action))
        if next_pos[0] < 0 or next_pos[0] >= self.map.shape[0] or\
            next_pos[1] < 0 or next_pos[1] >= self.map.shape[1]:
            return False
        if player and self.map[next_pos] == 0:
            return False
        return True

    def get_actions(self, state):
        if state[0] == self.goal:
            return [self.__escape_action]
        if (state[0] == state[1]): # Dead by minotaur
           return [self.__die_action]
        if state == self.__dead_state:
           return []
        if state == self.__free_state:
           return []
        possible_actions = []
        for action in self.actions:
            if self.__valid_action(state[0], action, player = True):
                possible_actions.append(action)
        return possible_actions

    def __mino_positions(self, mino_pos):
        minotaur_positions = []
        for action in self.actions:
            if self.__valid_action(mino_pos, action, player=False):
                new_mino_pos = tuple(map(sum, zip(mino_pos, action)))
                minotaur_positions.append(new_mino_pos)
        prob = 1./len(minotaur_positions)
        return minotaur_positions, prob

    def get_transitions(self, state, action):
        if action == self.__die_action:
            return [(self.__dead_state, 1)]
        if action == self.__escape_action:
            return [(self.__free_state, 1)]
        player_pos = tuple(np.array(state[0]) + np.array(action))
        minotaur_positions, prob = self.__mino_positions(state[1])
        trans = []
        for mino_pos in minotaur_positions:
            new_state = (player_pos,  mino_pos)
            trans.append((new_state, prob))
        return trans

    def get_reward(self, state, action):
        if state == self.__dead_state:
            return -2
        if state == self.__free_state:
            # print("asking reward of free state")
            return 0
        if action == self.__die_action:
            return -100
        if action == self.__escape_action:
            return 10
        return -1

    def get_heat_map(self, values_dict, minotaur_pos):
        minotaur_pos = tuple(minotaur_pos)
        heatmap = np.zeros(self.map.shape)
        for state in values_dict:
            if state[1] == minotaur_pos:
                heatmap[state[0]] = values_dict[state]
        return heatmap

    def generate_game(self, values, deadline):
        cur_state = (self.start, self.goal) #player in start, mino in goal
        policy = self.get_policy(values)
        T = 0
        states = [cur_state]
        while True:
            if cur_state[0] == cur_state[1]:
                return states, "Minotaur"
            if cur_state[0] == self.goal:
                return states, "Win"
            if T == deadline:
                return states, "Deadline"
            action = policy[cur_state]
            new_player_pos = tuple(map(sum, zip(cur_state[0], action)))
            # gen random mino move
            possible_mino_positions, prob = self.__mino_positions(cur_state[1])
            new_mino_pos = random.choice(possible_mino_positions)
            cur_state = (new_player_pos, new_mino_pos)
            states.append(cur_state)
            T = T + 1

if __name__ == "__main__":
    problem = Problem1B()
    states = problem.get_states()
    initial_values = {state : 0 for state in states}

    # value_hist = problem.dynamic_programming(initial_values = initial_values, T = 20)
    value_hist = problem.value_iteration(initial_values = initial_values, gamma = 0.95)

    mino_pos = (4, 1)
    heatmap = problem.get_heat_map(value_hist[-1], mino_pos)
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.show()


    # time_steps = 20

    # # print("Computing policy ...")
    # # value_hist = problem.dynamic_programming(initial_values = initial_values, T = time_steps)
    # policy = problem.get_policy(value_hist[-1])

    # n_games = 1000
    # print("Generating {} games ...".format(n_games))
    # endgames = {"Minotaur":0, "Deadline":0, "Win":0}
    # for i in tqdm(range(n_games)):
    #     states, result = problem.generate_game(value_hist[-1], deadline = time_steps)
    #     endgames[result] += 1
    # print("Wins: {}, Minotaur: {}, Deadline: {}".format(endgames["Win"],\
    #         endgames["Minotaur"], endgames["Deadline"]))
