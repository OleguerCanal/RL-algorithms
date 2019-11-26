import numpy as np
from tqdm import tqdm

from p3 import *
from rl_core import *

if __name__ == "__main__":
    initial_state = State(None, thief = (0, 0), police = (3, 3))

    agent = Agent()
    # agent.load("../models/sarsa_1e7.npy")
    agent.sarsa_train(initial_state, epochs = 1e5, epsilon = 0.5)
    # agent.save("../models/sarsa_1e7.npy")
    
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
        g, u = agent.test(initial_state, T=20)
        greedy += g
        uniform += u

    print("Greedy average reward: " + str(float(greedy)/n_games))
    print("Uniform averge reward: " + str(float(uniform)/n_games))