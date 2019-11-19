import numpy as np
from tqdm import tqdm
import random
import copy

def get_valid_action(pos, actions):
    def is_valid(action):
        if pos[0] + action[0] < 0 or pos[0] + action[0] > 3 or\
            pos[1] + action[1] < 0 or pos[1] + action[1] > 3:
            return False
        return True
    valid_actions = []
    for action in actions:
        if is_valid(action):
            valid_actions.append(action)
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

    def step(self, action):
        self.thief += np.array(action)
        self.police += random.choice(\
            get_valid_action(self.police, State.police_actions))

        reward = 0

        if np.array_equal(self.thief, State.bank):
            reward += 1
        if np.array_equal(self.thief, self.police):
            reward -= 10
        return reward

    def __string__(self):
        print(self.thief)
        print(self.police)

    def __equal__(self, other):
        return self.thief == other.thief and self.police == other.police

    def __hash__(self):
        return hash(tuple(self.thief)) + hash(1000*tuple(self.police))


class Quality():
    def __init__(self):
        self.Q = {}

    def update(self, state, action, value):
        action = tuple(action)
        if state not in self.Q:
            self.Q[state] = {}
        if action not in self.Q[state]:
            self.Q[state][action] = (value, 0)
        self.Q[state][action] = (value, self.Q[state][action][1] + 1)

    def get(self, state, action):
        action = tuple(action)
        if state not in self.Q:
            return 0
        if tuple(action) not in self.Q[state]:
            return 0
        return self.Q[state][action][0]
    
    def get_count(self, state, action):
        action = tuple(action)
        if state not in self.Q:
            return 1
        if tuple(action) not in self.Q[state]:
            return 1
        return self.Q[state][action][1]
    
    def get_best_action_val(self, state):
        if state not in self.Q:
            return 0
        return np.argmax(self.Q[state])

    def best_action(self, state):
        if state not in self.Q:
            print("State never seen, taking random action")
            pol = Policy()
            return pol.uniform(state)
        return max(self.Q[state], key=self.Q[state].get)

    def converged(self, old, tol = 1e-5):
        for state in self.Q:
            for action in self.Q[state]:
                if abs(self.get(state, action) - old.get(state, action)) > tol:
                    return False
        return True


class Policy():
    def __init__(self, Q=None, epsilon = 0.1):
        self.Q = Q
        self.epsion = epsilon
    
    def uniform(self, state):
        # print(get_valid_action(state.thief, state.thief_actions))
        return random.choice(\
            get_valid_action(state.thief, state.thief_actions))

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

    def train(self, initial_state, max_iter = 1e10):
        state = initial_state

        for it in tqdm(range(int(max_iter))):
            old_Q = copy.deepcopy(self.Q)

            action = self.mu.uniform(state)
            next_state = State(state)
            reward = next_state.step(action)

            self.update(state, action, reward, next_state)

            state = next_state

            # if self.Q.converged(old_Q):
            #     print("Iterations: " + str(it))
            #     break


    def test(self, initial_state, T = 20):
        greedy_state = initial_state
        uniform_state = initial_state
        
        pi = Policy(self.Q)
        greedy_reward = 0
        uniform_reward = 0
        for i in tqdm(range(T)):
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
    

    
    n_games = 100
    greedy = 0
    uniform = 0
    for n in tqdm(range(n_games)):
        g, u = agent.test(initial_state, T=20)
        greedy += g
        uniform += u

    print("Greedy average reward: " + str(float(greedy)/n_games))
    print("Uniform averge reward: " + str(float(uniform)/n_games))