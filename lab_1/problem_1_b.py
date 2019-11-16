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
        self.start = (0, 0)
        self.goal = (6, 5)

    def get_states(self):
        player_states = tuple(np.swapaxes(np.where(self.map == 1), 0, 1))
        minotaur_states = tuple(np.swapaxes(np.where(self.map < 2), 0, 1))
        states =  list(itertools.product((player_states), (minotaur_states)))
        tuple_states = []
        for state in states:
            tuple_states.append((tuple(state[0]), tuple(state[1])))
        return tuple_states
    
    def __valid_action(self, position, action, player = True):
        next_pos = tuple(np.array(position) + np.array(action))
        if next_pos[0] < 0 or next_pos[0] >= self.map.shape[0] or\
            next_pos[1] < 0 or next_pos[1] >= self.map.shape[1]:
            return False
        if player and self.map[next_pos] == 0:
            return False
        return True

    def get_actions(self, state):
        #if state == self.goal:
        #    return []
        #if (state[0] == state[1]):
        #    return []
        possible_actions = []
        for action in self.actions:
            if self.__valid_action(state[0], action, player = True):
                possible_actions.append(action)
        return possible_actions

    def get_transitions(self, state, action):
        player_pos = tuple(np.array(state[0]) + np.array(action))
        minotaur_positions, prob = self.__mino_positions(state[1])
        for mino_pos in minotaur_positions:
            new_state = (player_pos,  mino_pos)
            yield new_state, prob

    def __mino_positions(self, mino_pos):
        minotaur_positions = []
        for action in self.actions:
            if self.__valid_action(mino_pos, action, player=False):
                new_mino_pos = tuple(map(sum, zip(mino_pos, action)))
                minotaur_positions.append(new_mino_pos)
        prob = 1./len(minotaur_positions)
        return minotaur_positions, prob


    def get_reward(self, state, action):
        if tuple(state[0]) == (6, 5):
            return 0
        if state[0] == state[1]:
            return -float('inf')
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
                print("Eaten by minotaur")
                return states, 0
            if cur_state[0] == self.goal:
                print("Win!")
                return states, 1
            if T == deadline:
                print("Deadline")
                return states, 0
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
    # for state in states:
    #     rew = max([problem.get_reward(state, a) for a in problem.get_actions(state)])
    #     initial_values[state] = rew

    value_hist = problem.dynamic_programming(initial_values = initial_values, T = 15)
    policy = problem.get_policy(value_hist[-1])
    mino_pos = (2,3)

    wins = 0
    tot = 0
    for i in range(100):
        states, result = problem.generate_game(value_hist[-1], 30)
        wins += result
        tot += 1
    print("Wins: {} out of {}".format(wins, tot))

#   states, result = problem.generate_game(value_hist[-1], 30)
#   for state in states:
#        print(state)
