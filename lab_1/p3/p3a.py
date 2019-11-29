import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import colors

from p3 import *
from rl_core import *

if __name__ == "__main__":
    initial_state = State(None, thief = (0, 0), police = (3, 3))

    agent = Agent()
    agent.load("../models/1e7.npy")
    # agent.q_train(initial_state, epochs = 1e5)
    # agent.save("../models/q_1e7.npy")
    
    # agent.plot_convergence()
    # agent.Q.plot((3, 3))

    policy = Policy(agent.Q)
    for i in range(4):
        for j in range(4):
            policy.plot((i, j), save = True, path = "../figures/p3a_policy", ind=4*i+j)

    # _, _, states, rewards = agent.test(initial_state, T=25)
    # i = 0
    # for s, reward in zip(states, rewards):
    #     heatmap = np.zeros((4, 4))
    #     cmap = colors.ListedColormap(['white', 'red', 'blue', 'green'])
    #     heatmap[s.bank[0]][s.bank[1]] = 0.6
    #     heatmap[s.police[0]][s.police[1]] = 0.3
    #     heatmap[s.thief[0]][s.thief[1]] = 1
    #     plt.imshow(heatmap, cmap=cmap, interpolation='nearest')
    #     plt.title("Reward: " + str(reward))
    #     plt.savefig("../figures/p3a_example/fig" + str(i) + ".png")
    #     plt.plot()
    #     plt.pause(0.75)
    #     i += 1


    # n_games = 1000
    # greedy = 0
    # uniform = 0
    # for n in tqdm(range(n_games)):
    #     g, u, _, _ = agent.test(initial_state, T=100)
    #     greedy += g
    #     uniform += u

    # print("Greedy average reward: " + str(float(greedy)/n_games))
    # print("Uniform averge reward: " + str(float(uniform)/n_games))