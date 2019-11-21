import numpy as np
from tqdm import tqdm
import random
import copy
from matplotlib import pyplot as plt

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
    
    def __init__(self, state=None, thief=(0, 0), police=(3, 3)):
        if state is None:
            self.thief = np.array(thief)
            self.police = np.array(police)
        else:
            self.thief = state.thief
            self.police = state.police

    def get_coord(self):
        return self.thief[0], self.thief[1], self.police[0], self.police[1]

    def step(self, action_id):
        self.thief += State.thief_actions[action_id]
        self.police += State.police_actions[random.choice(\
                get_valid_action_id(self.police, State.police_actions))]

        reward = 0
        if np.array_equal(self.thief, State.bank):
            reward += 1
        if np.array_equal(self.thief, self.police):
            reward -= 10
        return reward

    # def __string__(self):
    #     print(self.thief)
    #     print(self.police)

class Quality():
    def __init__(self):
        self.Q = np.zeros((4, 4, 4, 4, 5, 2))
        self.Q[:, :, :, :, :, 1] = 1
        # print(self.Q)

    def update(self, state, action_id, value):
        # if np.isnan(value) or value != 0:
        #     print("kfjsbfbjfbvfvfvvf")
        a, b, c, d = state.get_coord()
        self.Q[a, b, c, d, action_id, 0] = value
        self.Q[a, b, c, d, action_id, 1] += 1

        # if value != 0:
        #     print(value)
        #     print(self.Q[a, b, c, d, action_id, 0])
        #     print(self.Q[a, b, c, d, action_id, 1])
        #     print((a, b, c, d))
        #     print(action_id)

    def get(self, state, action_id):
        a, b, c, d = state.get_coord()
        # print(self.Q[a, b, c, d, action_id, 0])
        return self.Q[a, b, c, d, action_id, 0]
    
    def get_count(self, state, action_id):
        a, b, c, d = state.get_coord()
        return self.Q[a, b, c, d, action_id, 1]
    
    def get_best_action_val(self, state):
        a, b, c, d = state.get_coord()
        return np.max(self.Q[a, b, c, d, :, 0])
        # print(m)
        # print(self.Q[a, b, c, d, :, 0])
        # SystemError()

    def best_action(self, state):
        a, b, c, d = state.get_coord()
        return np.argmax(self.Q[a, b, c, d, :, 0])

    def converged(self, old, tol = 1e-5):
        return np.max(np.abs(self.Q[:, :, :, :, 0] - old.Q[:, :, :, :, 0])) < tol

    def show(self, police):
        heatmap = np.zeros((4, 4))
        for index, val in np.ndenumerate(self.Q[:, :, :, :, 0]):
            if index[2] == police[0] and index[3] == police[1]:
                state = State(None, (index[0], index[1], (index[2], index[3])))
                best_action_id = self.best_action(state)
                heatmap[index[0], index[1]] = best_action_id
        
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.show()

class Policy():
    def __init__(self, Q=None, epsilon = 0.1):
        self.Q = Q
        self.epsion = epsilon
    
    def uniform(self, state):
        # print(get_valid_action(state.thief, state.thief_actions))
        return random.choice(\
            get_valid_action_id(state.thief, state.thief_actions))

    def greedy(self, state):
        return self.Q.best_action(state)

    def epsilon_greedy(self, state):
        if random.uniform(0, 1) < 0.1:
            return self.uniform(state)
        else:
            return self.greedy(state)

class Agent():
    def __init__(self):
        self.Q = Quality()
        self.mu = Policy()

        self.lamb = 0.8

    def __alpha(self, state, action):
        return 1./(self.Q.get_count(state, action) ** (2/3))

    def update(self, state, action, reward, next_state, step = 1):
        value = self.Q.get(state, action) + self.__alpha(state, action)*\
            (reward + self.lamb*self.Q.get_best_action_val(next_state) - self.Q.get(state, action))
        self.Q.update(state, action, value)

    def train(self, initial_state, epochs = 1e8, steps = 100):
        for epoch in tqdm(range(int(epochs))):
            state = initial_state
            for step in range(steps):
                old_Q = copy.deepcopy(self.Q)

                action = self.mu.uniform(state)
                next_state = State(state)
                reward = next_state.step(action)

                self.update(state, action, reward, next_state)

                state = next_state

            # if self.Q.converged(old_Q):
            #     print("Iterations: " + str(epoch))
            #     break


    def test(self, initial_state, T = 20):
        greedy_state = initial_state
        uniform_state = initial_state
        
        pi = Policy(self.Q)
        greedy_reward = 0
        uniform_reward = 0
        for i in range(T):
            greedy_action = pi.greedy(greedy_state)
            greedy_next_state = State(greedy_state)
            greedy_reward += greedy_next_state.step(greedy_action)

            uniform_action = pi.uniform(uniform_state)
            uniform_next_state = State(uniform_state)
            uniform_reward += uniform_next_state.step(uniform_action)

        # print("Greedy total reward: " + str(greedy_reward))
        # print("Uniform totl reward: " + str(uniform_reward))
        return greedy_reward, uniform_reward


if __name__ == "__main__":
    agent = Agent()
    initial_state = State(None, thief = (0, 0), police = (3, 3))
    agent.train(initial_state, 10000)
    
    for i in range(4):
        for j in range(4):
            police = (i, j)
            agent.Q.show(police)

    # print(agent.Q.Q[:, :, :, :, 0])
    
    # n_games = 100
    # greedy = 0
    # uniform = 0
    # for n in tqdm(range(n_games)):
    #     g, u = agent.test(initial_state, T=20)
    #     greedy += g
    #     uniform += u

    # print("Greedy average reward: " + str(float(greedy)/n_games))
    # print("Uniform averge reward: " + str(float(uniform)/n_games))

    # initial_state = State()
    # for action in range(4):
    #     print(agent.Q.get(initial_state, action))
