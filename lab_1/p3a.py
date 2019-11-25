import numpy as np
from tqdm import tqdm
import random
import copy
from matplotlib import pyplot as plt
import pickle

from rl_core import *

def get_valid_action_id(pos, actions):
    def is_valid(action):
        if pos[0] + action[0] < 0 or pos[0] + action[0] > 3 or\
            pos[1] + action[1] < 0 or pos[1] + action[1] > 3:
            return False
        return True
    valid_actions = []
    for action_id, action in enumerate(actions):
        if is_valid(action):
            valid_actions.append(action_id)
    return valid_actions


class State():
    bank = np.array([1, 1])
    thief_actions = [np.array(action)\
            for action in [(0, 0), (1, 0), (-1, 0), (0, -1), (0, 1)]]
    police_actions = [np.array(action)\
            for action in [(1, 0), (-1, 0), (0, -1), (0, 1)]]

    # Precompute thief actions for every thief pos:
    valid_thief_actions = []
    for i in range(4):
        temp = []
        for j in range(4):
            temp.append(get_valid_action_id((i, j), thief_actions))
        valid_thief_actions.append(temp)
    
    # Precumpute police actions for every police pos:
    valid_police_actions = []
    for i in range(4):
        temp = []
        for j in range(4):
            temp.append(get_valid_action_id((i, j), police_actions))
        valid_police_actions.append(temp)

    def __init__(self, state=None, thief=(0, 0), police=(3, 3)):
        if state is None:
            self.thief = np.array(thief)
            self.police = np.array(police)
        else:
            self.thief = copy.deepcopy(state.thief)
            self.police = copy.deepcopy(state.police)

    def get_coord(self):
        return self.thief[0], self.thief[1], self.police[0], self.police[1]

    def step(self, action_id):
        self.thief += State.thief_actions[action_id]
        self.police += State.police_actions[random.choice(\
            State.valid_police_actions[self.police[0]][self.police[1]])]

        reward = 0
        if np.array_equal(self.thief, State.bank):
            reward += 1
        if np.array_equal(self.thief, self.police):
            reward -= 10
        return reward

    def __str__(self):
        return "Thief: " + str(self.thief) + "\n" + "Police: " + str(self.police)

    def __eq__(self, other):
        return np.array_equal(self.thief, other.thief) and np.array_equal(self.police, other.police)

if __name__ == "__main__":
    agent = Agent()
    # agent.load("models/1e4.npy")

    initial_state = State(None, thief = (0, 0), police = (3, 3))

    agent.q_train(initial_state, epochs = 1e6)
    # agent.save("models/1e6.npy")
    
    agent.Q.plot((3, 3))
    agent.plot_convergence()

    policy = Policy(agent.Q)
    policy.plot((3, 3))

    # print(agent.Q.Q[:, :, :, :, 0])
    
    n_games = 100
    greedy = 0
    uniform = 0
    for n in tqdm(range(n_games)):
        g, u = agent.test(initial_state, T=20)
        greedy += g
        uniform += u

    print("Greedy average reward: " + str(float(greedy)/n_games))
    print("Uniform averge reward: " + str(float(uniform)/n_games))
