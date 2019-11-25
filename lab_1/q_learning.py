import numpy as np
from tqdm import tqdm
import random
import copy
from matplotlib import pyplot as plt
import pickle

from p3a import State

class Quality():
    def __init__(self):
        self.table = np.zeros((4, 4, 4, 4, 5, 2))
        self.table[:, :, :, :, :, 1] = 1

    def update(self, state, action_id, value):
        self.table[state.get_coord()][action_id][0] = value
        self.table[state.get_coord()][action_id][1] += 1

    def get(self, state, action_id):
        return self.table[state.get_coord()][action_id][0]
    
    def get_count(self, state, action_id):
        return self.table[state.get_coord()][action_id][1]

    def get_best_action_val(self, state):
        return np.max(self.table[state.get_coord()][:][0])

    def best_action(self, state):
        return np.argmax(self.table[state.get_coord()][:][0])

    def reset_counters(self):
        self.table[:, :, :, :, :, 1] = 1

    def converged(self, old, tol = 1e-5):
        return np.max(np.abs(self.table[:, :, :, :, 0] - old.Q[:, :, :, :, 0])) < tol

    def show(self, police):
        heatmap = np.zeros((4, 4))
        for index, val in np.ndenumerate(self.table[:, :, :, :, 0]):
            if index[2] == police[0] and index[3] == police[1]:
                state = State(None, (index[0], index[1], (index[2], index[3])))
                best_action_id = self.get(state, 0)
                heatmap[index[0], index[1]] = best_action_id

        # fig, axs = plt.subplots(2, 2)
        plt.subplot(3, 3, 5)
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')

        heatmap = np.zeros((4, 4))
        for index, val in np.ndenumerate(self.table[:, :, :, :, 0]):
            if index[2] == police[0] and index[3] == police[1]:
                state = State(None, (index[0], index[1], (index[2], index[3])))
                best_action_id = self.get(state, 1)
                heatmap[index[0], index[1]] = best_action_id
        plt.subplot(3, 3, 8)
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')

        heatmap = np.zeros((4, 4))
        for index, val in np.ndenumerate(self.table[:, :, :, :, 0]):
            if index[2] == police[0] and index[3] == police[1]:
                state = State(None, (index[0], index[1], (index[2], index[3])))
                best_action_id = self.get(state, 2)
                heatmap[index[0], index[1]] = best_action_id        
        plt.subplot(3, 3, 2)
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')

        heatmap = np.zeros((4, 4))
        for index, val in np.ndenumerate(self.table[:, :, :, :, 0]):
            if index[2] == police[0] and index[3] == police[1]:
                state = State(None, (index[0], index[1], (index[2], index[3])))
                best_action_id = self.get(state, 3)
                heatmap[index[0], index[1]] = best_action_id
        
        plt.subplot(3, 3, 4)
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')

        heatmap = np.zeros((4, 4))
        for index, val in np.ndenumerate(self.table[:, :, :, :, 0]):
            if index[2] == police[0] and index[3] == police[1]:
                state = State(None, (index[0], index[1], (index[2], index[3])))
                best_action_id = self.get(state, 4)
                heatmap[index[0], index[1]] = best_action_id
        plt.subplot(3, 3, 6)        
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.show()

    def save(self, name):
        np.save(name, self.table)
        print("Model saved!")

    def load(self, name):
        self.table = np.load(name)

class Policy():
    def __init__(self, Q=None):
        self.Q = Q
    
    def uniform(self, state):
        return random.choice(\
            state.valid_thief_actions[state.thief[0]][state.thief[1]])

    def greedy(self, state):
        return self.Q.best_action(state)

    def epsilon_greedy(self, state, epsilon = 0.1):
        if random.uniform(0, 1) < self.epsilon:
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
        q = self.Q.get(state, action)
        value = q + self.__alpha(state, action)*\
            (reward + self.lamb*self.Q.get_best_action_val(next_state) - q)
        self.Q.update(state, action, value)

    def train(self, initial_state, epochs = 1e8, steps = 100):
        for epoch in tqdm(range(int(epochs))):
            state = initial_state
            # old_Q = copy.deepcopy(self.Q)
            self.Q.reset_counters()
            
            for step in range(int(steps)):

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
