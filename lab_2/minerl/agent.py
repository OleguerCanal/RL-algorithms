import gym
import minerl
import logging

logging.basicConfig(level=logging.DEBUG)

env = gym.make('MineRLNavigateDense-v0')

obs  = env.reset()
done = False
net_reward = 0

while not done:
    env.render()
    
    action = env.action_space.noop()

    # action['camera'] = [-0.1, 0]
    action['back'] = 0
    action['forward'] = 1
    action['jump'] = 0
    action['attack'] = 1

    obs, reward, done, info = env.step(
        action)

    net_reward += reward
    print("Total reward: ", net_reward)