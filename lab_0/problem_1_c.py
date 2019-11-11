import numpy as np
import matplotlib.pyplot as plt
from abstract_mdp import MDP

class Problem1C(MDP):
    def __init__(self):
        self.map = np.ones((6, 7))
        self.map[0:3, 2] = 0
        self.map[4, 1:6] = 0

        self.terminals = [(5, 5)]

    def get_states(self):
        states = []
        for state, _ in np.ndenumerate(self.map):
            if self.map[tuple(state)] == 1:
                states.append(state)
        return states

    def get_actions(self, state = None):
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def get_transition_prob(self, next_state, state, action): # Transitions
        if next_state != tuple(np.array(state) + np.array(action)):
            return 0
        if state in self.terminals:
            return 0
        if next_state[0] < 0 or next_state[0] >= self.map.shape[0] or\
            next_state[1] < 0 or next_state[1] >= self.map.shape[1]:
            return 0
        else:
            return self.map[next_state]

    def get_reward(self, state, action):
        new_state = tuple(np.array(state) + np.array(action))
        if new_state[0] < 0 or new_state[0] >= self.map.shape[0] or\
            new_state[1] < 0 or new_state[1] >= self.map.shape[1]:
            return float('-inf')
        if self.map[new_state] == 0:
            return float('-inf')
        return -1


if __name__ == "__main__":
    problem = Problem1C()
    value_hist = problem.value_iteration(initial_values = np.zeros((6, 7)))

    for value in value_hist:
        plt.imshow(value, cmap='hot', interpolation='nearest')
        plt.show()