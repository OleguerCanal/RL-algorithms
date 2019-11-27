import numpy as np
from tqdm import tqdm

from p3 import *
from rl_core import *

if __name__ == "__main__":
    initial_state = State(None, thief = (0, 0), police = (3, 3))

    agent = Agent()
    agent.load("../models/1e7.npy")
    # agent.q_train(initial_state, epochs = 1e5)
    # agent.save("../models/q_1e7.npy")
    
    agent.plot_convergence()
    # agent.Q.plot((3, 3))

    # policy = Policy(agent.Q)
    # for i in range(4):
    #     for j in range(4):
    #         policy.plot((i, j), save = False, path = "figures/p3b/")

    n_games = 100
    greedy = 0
    uniform = 0
    for n in tqdm(range(n_games)):
        g, u = agent.test(initial_state, T=100)
        greedy += g
        uniform += u

    print("Greedy average reward: " + str(float(greedy)/n_games))
    print("Uniform averge reward: " + str(float(uniform)/n_games))