import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from p3 import *
from rl_core import *

def plot_conv():
    path = "../models/sarsa_1e6_conv_0.01.npy"
    plt.semilogx(np.load(path), label = "Epsilon = 0.01")

    path = "../models/sarsa_1e6_conv_0.1.npy"
    plt.semilogx(np.load(path), label = "Epsilon = 0.1")

    path = "../models/sarsa_1e6_conv_0.5.npy"
    plt.semilogx(np.load(path), label = "Epsilon = 0.5")

    path = "../models/sarsa_1e6_conv_0.7.npy"
    plt.semilogx(np.load(path), label = "Epsilon = 0.7")

    path = "../models/sarsa_1e6_conv_1.npy"
    plt.semilogx(np.load(path), label = "Epsilon = 1")

    plt.title("Initial state value")
    plt.xlabel("Update")
    plt.ylabel("State Value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # plot_conv()

    initial_state = State(None, thief = (0, 0), police = (3, 3))

    epsilons = [0.01, 0.1, 0.5, 0.7]
    epsilons = [1]
    for epsilon in epsilons:
        agent = Agent()
        # agent.load("../models/sarsa_1e7.npy")
        agent.sarsa_train(initial_state, epochs = 1e6, epsilon = epsilon)
        agent.save("../models/sarsa_1e6.npy")
    
        agent.plot_convergence()

    # policy = Policy(agent.Q)
    # for i in range(4):
    #     for j in range(4):
    #         policy.plot((i, j), save = False, path = "figures/p3b/")

    # Test against uniform policy
    n_games = 100
    greedy = 0
    uniform = 0
    for n in tqdm(range(n_games)):
        g, u = agent.test(initial_state, T=100)
        greedy += g
        uniform += u

    print("Greedy average reward: " + str(float(greedy)/n_games))
    print("Uniform averge reward: " + str(float(uniform)/n_games))


