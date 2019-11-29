import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from tqdm import tqdm
import time
import os

from  infinite_mdp import *

map = np.ones((3, 6))

class State:
    player_actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    mino_actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    start_pos = (0, 0)
    police_station = (1, 2)
    banks = [(0, 0), (0, 5), (2, 0), (2, 5)]

    def __init__(self, coord=(0, 0, 1, 2)):
        self.player = (coord[0], coord[1])
        self.mino = (coord[2], coord[3])
    
    def get_coord(self):
        return self.player[0], self.player[1], self.mino[0], self.mino[1]

    def get_actions(self):
        if np.array_equal(self.player, self.mino):
            return []
        actions = []
        for action in State.player_actions:
            next_pos = np.array(self.player) + np.array(action)
            if next_pos[0] < 0 or next_pos[0] >= map.shape[0] or\
                next_pos[1] < 0 or next_pos[1] >= map.shape[1]:
                continue
            actions.append(action)
        return actions

    def get_transitions(self, action):
        next_state = State(self.get_coord())
        next_state.player = tuple(np.array(next_state.player) + np.array(action))

        if np.array_equal(self.player, self.mino):
            return [(State(0, 0, 1, 2), 1)]

        possible_states = []
        for mino_action in State.mino_actions:
            # Discard actions that would make police get further
            if np.array_equal(mino_action, (1, 0)) and self.player[0] - self.mino[0] < 0:
                continue
            if np.array_equal(mino_action, (-1, 0)) and self.player[0] - self.mino[0] > 0:
                continue
            if np.array_equal(mino_action, (0, 1)) and self.player[1] - self.mino[1] < 0:
                continue
            if np.array_equal(mino_action, (0, -1)) and self.player[1] - self.mino[1] > 0:
                continue
            # Discard actions that would make police exit board
            next_pos = np.array(next_state.mino) + np.array(mino_action)
            if next_pos[0] < 0 or next_pos[0] >= map.shape[0] or\
                next_pos[1] < 0 or next_pos[1] >= map.shape[1]:
                continue
            possible_state = State(next_state.get_coord())
            possible_state.mino = tuple(next_pos)
            possible_states.append(possible_state)

        prob = 1./len(possible_states)

        result = []
        for a in possible_states:
            result.append((a, prob))
        return result

    def reward(self):
        if np.array_equal(self.player, self.mino):
            return -50
        for bank in State.banks:
            if np.array_equal(self.player, bank):
                return 10
        return 0

    def __str__(self):
        return "Player:" + str(self.player) + ", " + "Mino:" + str(self.mino)

if __name__ == "__main__":
    lambdas = np.linspace(0, 0.9, 9)
    # lambdas = [0.8]
    # initial_values = []

    for lamb in lambdas:
        print("Lambda:" + str(lamb))
        value = np.zeros((map.shape[0], map.shape[1], map.shape[0], map.shape[1]))
        value[0, 0, :, :] = 10
        value[0, -1, :, :] = 10
        value[-1, 0, :, :] = 10
        value[-1, -1, :, :] = 10
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                value[i, j, i, j] = -50
        value = value_iteration_infinite(value = value, lambd = lamb)
        np.save("../models/p2_lamb_" + str(lamb) + ".png", value)
        # print(value[0, 0, 1, 2])
        # initial_values.append(value[0, 0, 1, 2])
        # np.save("../figures/p2/initial_vals.npy", initial_values)
        directory = "../figures/p2_policy/lamb_" + str(lamb) + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        count = 0
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                plot(value, (i, j), True, path = directory, ind = count)
                count += 1
    # initial_values = np.load("../figures/p2/initial_vals.npy")
    # lambdas = np.array(lambdas, dtype=np.float64)
    # vals = np.array(initial_values, dtype=np.float64)
    # print(initial_values)
    # print(lambdas)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.plot(list(lambdas), list(vals), '.b-')
    # plt.xlabel("Lambda")
    # plt.ylabel("Initial state value")
    # plt.title("Initial state value for different lambda values")
    # plt.show()


    # print("Trained!")
    # initial_state = State()
    # states, deaths, reward = generate_game(value, initial_state, 20)
    # # print("Deaths: " + str(deaths))
    # # print("Reward: " + str(reward))

    # i = 0
    # for s, r in states:
    #     heatmap = np.zeros(map.shape)
    #     heatmap[s.banks[0][0]][s.banks[0][1]] = 1
    #     heatmap[s.banks[1][0]][s.banks[1][1]] = 1
    #     heatmap[s.banks[2][0]][s.banks[2][1]] = 1
    #     heatmap[s.banks[3][0]][s.banks[3][1]] = 1
    #     heatmap[s.player[0]][s.player[1]] = 0.6
    #     heatmap[s.mino[0]][s.mino[1]] = 0.3
    #     cmap = colors.ListedColormap(['white', 'red', 'green', 'blue'])
    #     plt.imshow(heatmap, cmap=cmap, interpolation='nearest')
    #     plt.title("Reward: " + str(r))
    #     plt.savefig("../figures/p2b_run_test/fig" + str(i) + ".png")
    #     plt.plot()
    #     # plt.pause(0.75)
    #     i += 1
