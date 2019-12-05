import gym
import numpy as np
import random

from cartpole_dqn import DQNAgent

def evaluate_random(env, tests_num = 100):
    rewards = np.zeros(tests_num)
    i = 0
    while i < tests_num:
        done = False
        state = env.reset()
        while not done:
            action = random.randrange(2)
            _, reward, done, _ = env.step(action)
            rewards[i] += reward
        i += 1
    return np.mean(rewards)

def evaluate_federico(env, tests_num = 100):
    rewards = np.zeros(tests_num)
    i = 0
    while i < tests_num:
        done = False
        state = env.reset()
        while not done:
            action = 0
            if state[2] > 0:
                action = 1
            state, reward, done, _ = env.step(action)
            rewards[i] += reward
        i += 1
    return np.mean(rewards)

def evaluate_agent(env, agent, tests_num = 100):
    rewards = np.zeros(tests_num)
    i = 0
    while i < tests_num:
        done = False
        state = env.reset()
        while not done:
            action = agent.get_action(np.reshape(state, [1, 4]))
            # action = (action + 1)%2
            state, reward, done, _ = env.step(action)
            rewards[i] += reward
        i += 1
    return np.mean(rewards)


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    agent = DQNAgent(state_size = env.observation_space.shape[0],
                  action_size = env.action_space.n)

    agent.load("test1")  # Load NN weights

    tests_num = 100
    random_value = evaluate_random(env = env, tests_num = tests_num)
    federico_value = evaluate_federico(env = env, tests_num = tests_num)
    agent_value = evaluate_agent(env = env, agent = agent, tests_num = tests_num)
    
    print("Random value:", random_value)
    print("Federico value:", federico_value)
    print("Agent value:", agent_value)


