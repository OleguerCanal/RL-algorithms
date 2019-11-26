import numpy as np
from tqdm import tqdm
import random
import copy
from matplotlib import pyplot as plt
from matplotlib import colors
import pickle

from p3 import State

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
        valid_actions = state.valid_thief_actions[state.thief[0]][state.thief[1]]
        val = self.table[state.get_coord()][valid_actions[0]][0]
        for action in valid_actions:
            val = max(val, self.table[state.get_coord()][action][0])
        return val

    def best_action(self, state):
        valid_actions = state.valid_thief_actions[state.thief[0]][state.thief[1]]
        best_action = valid_actions[0]
        max_val = self.get(state, valid_actions[0])
        for action in valid_actions:
            val = self.get(state, action)
            if val > max_val:
                max_val = val
                best_action = action
        return best_action

    def reset_counters(self):
        self.table[:, :, :, :, :, 1] = 1

    def converged(self, old, tol = 1e-5):
        return np.amax(np.abs(self.table[:, :, :, :][0] - old.table[:, :, :, :][0])) < tol

    def plot(self, police):
        heatmap = np.zeros((4, 4, 5))
        for index, val in np.ndenumerate(self.table[:, :, :, :, 0]):
            if index[2] == police[0] and index[3] == police[1]:
                state = State(None, thief = (index[0], index[1]), police=police)
                for action_id in range(5):
                    heatmap[index[0], index[1], action_id] = self.get(state, action_id)

        # fig, axs = plt.subplots(2, 2)
        plt.subplot(3, 3, 5)
        plt.imshow(heatmap[:, :, 0], cmap='hot', interpolation='nearest')

        plt.subplot(3, 3, 8)
        plt.imshow(heatmap[:, :, 1], cmap='hot', interpolation='nearest')

        plt.subplot(3, 3, 2)
        plt.imshow(heatmap[:, :, 2], cmap='hot', interpolation='nearest')

        plt.subplot(3, 3, 4)
        plt.imshow(heatmap[:, :, 3], cmap='hot', interpolation='nearest')

        plt.subplot(3, 3, 6)        
        plt.imshow(heatmap[:, :, 4], cmap='hot', interpolation='nearest')
        plt.show()

    def save(self, name):
        np.save(name, self.table)
        print("Model saved!")

    def load(self, name):
        self.table = np.load(name)
        print("Model " + name + " loaded!")

class Policy():
    def __init__(self, Q=None):
        self.Q = Q
    
    def uniform(self, state):
        return random.choice(\
            state.valid_thief_actions[state.thief[0]][state.thief[1]])

    def greedy(self, state):
        return self.Q.best_action(state)

    def epsilon_greedy(self, state, epsilon = 0.1):
        if random.uniform(0, 1) < epsilon:
            return self.uniform(state)
        else:
            return self.greedy(state)
    
    def plot(self, police, save = False, path = ""):
        x = []
        y = []
        ax = []
        ay = []
        for index, val in np.ndenumerate(self.Q.table[:, :, :, :][0]):
            if index[2] == police[0] and index[3] == police[1] and index[4] == 0:
                s = State(None, thief=(index[0], index[1]), police=police)
                best_action_value = self.Q.get_best_action_val(s)
                for action_id in range(len(s.thief_actions)):
                    if abs(self.Q.get(s, action_id) - best_action_value) < 0.1:
                        best_action = s.thief_actions[action_id]
                        x.append(index[1])
                        y.append(index[0])
                        ax.append(int(best_action[1]))
                        ay.append(-int(best_action[0]))

        heatmap = np.zeros((4, 4))
        cmap = colors.ListedColormap(['white', 'red', 'green'])
        heatmap[police] = 0.3
        heatmap[s.bank[0]][s.bank[1]] = 0.6
        plt.imshow(heatmap, cmap=cmap, interpolation='nearest')

        plt.quiver(x, y, ax, ay)
        if save:
            plt.savefig(path + "/policy" + str(police[0]) + str(police[1]) + "png")
        plt.show()


class Agent():
    def __init__(self):
        self.Q = Quality()
        self.mu = Policy(self.Q)

        self.lamb = 0.8
        self.initial_state_value = []

    def __alpha(self, state, action):
        return 1./(np.power(self.Q.get_count(state, action), 2./3.))

    def __q_update(self, state, action, reward, next_state):
        q = self.Q.get(state, action)
        alpha = self.__alpha(state, action)
        value = q + alpha*(reward + self.lamb*self.Q.get_best_action_val(next_state) - q)
        self.Q.update(state, action, value)

    def q_train(self, initial_state, epochs = 1e8):
        state = copy.deepcopy(initial_state)
        for t in tqdm(range(int(epochs))):
            old_Q = copy.deepcopy(self.Q)  # To check convergence

            if state == initial_state:  # Save initial state value (to plot convergence)
                self.initial_state_value.append(self.Q.get_best_action_val(state))

            action = self.mu.uniform(state)
            next_state = copy.deepcopy(state)
            reward = next_state.step(action)
            self.__q_update(state, action, reward, next_state)
            state = next_state

            # if self.Q.converged(old_Q):
            #     print("Iterations: " + str(t))
            #     break

    def __sarsa_update(self, state, action, reward, state_next, action_next):
        q = self.Q.get(state, action)
        alpha = self.__alpha(state, action)
        value = q + alpha*(reward + self.lamb*self.Q.get(state_next, action_next) - q)
        self.Q.update(state, action, value)

    def sarsa_train(self, initial_state, epochs = 1e5, epsilon = 0.1):
        state = copy.deepcopy(initial_state)
        action = self.mu.epsilon_greedy(state, epsilon=epsilon)

        for t in tqdm(range(int(epochs))):
            self.mu.Q = self.Q  # Update Q function in policy
            old_Q = copy.deepcopy(self.Q)  # To check convergence

            if state == initial_state:  # Save initial state value (to plot convergence)
                self.initial_state_value.append(self.Q.get_best_action_val(state))

            state_next = copy.deepcopy(state)
            reward = state_next.step(action)
            action_next = self.mu.epsilon_greedy(state_next, epsilon=epsilon)
            self.__sarsa_update(state, action, reward, state_next, action_next)
            state = state_next
            action = action_next

            # if self.Q.converged(old_Q):
            #     print("Iterations: " + str(t))
            #     break

    def test(self, initial_state, T = 20):
        greedy_state = copy.deepcopy(initial_state)
        uniform_state = copy.deepcopy(initial_state)
        
        pi = Policy(self.Q)
        greedy_reward = 0
        uniform_reward = 0
        for i in range(T):
            # Greedy policy
            greedy_action = pi.greedy(greedy_state)
            greedy_next_state = copy.deepcopy(greedy_state)
            greedy_reward += greedy_next_state.step(greedy_action)
            greedy_state = greedy_next_state
            
            # Uniform policy
            uniform_action = pi.uniform(uniform_state)
            uniform_next_state = copy.deepcopy(uniform_state)
            uniform_reward += uniform_next_state.step(uniform_action)
            uniform_state = uniform_next_state
        return greedy_reward, uniform_reward

    def save(self, name):
        self.Q.save(name)
        name = name.replace(".npy", "_conv.npy")
        np.save(name, self.initial_state_value)

    def load(self, name):
        self.Q.load(name)
        name = name.replace(".npy", "_conv.npy")
        try:
            self.initial_state_value = np.load(name)
        except:
            print("Load error! No convergence was saved")

    def plot_convergence(self):
        plt.plot(self.initial_state_value)
        plt.show()
