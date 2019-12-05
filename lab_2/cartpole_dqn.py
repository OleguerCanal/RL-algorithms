import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from tqdm import tqdm
from tqdm import trange
from time import sleep

try:
    from tests import *
except:
    pass

EPISODES = 100 # Maximum number of episodes

# DQN Agent for the Cartpole
# Q function approximation with NN, experience replay, and target network
class DQNAgent:
    # Constructor for the agent (invoked when DQN is first called in main)
    def __init__(self, state_size, action_size):
        self.check_solve = False	# If True, stop if you satisfy solution confition
        self.render = False        # If you want to see Cartpole learning, then change to True

        # Get size of state and action
        self.state_size = state_size
        self.action_size = action_size

       #  Modify here

        # Set hyper parameters for the DQN. Do not adjust those labeled as Fixed.
        self.discount_factor = 0.95
        self.learning_rate = 0.005
        self.epsilon = 0.02 # Fixed
        self.batch_size = 32 # Fixed
        self.memory_size = 1000
        self.train_start = 1000 # Fixed
        self.target_update_frequency = 1

        # Number of test states for Q value plots
        self.test_state_no = 10000

        # Create memory buffer using deque
        self.memory = deque(maxlen=self.memory_size)

        # Create main network and target network (using build_model defined below)
        self.model = self.build_model()
        self.target_model = self.build_model()

        # Initialize target network
        self.update_target_model()

    # Approximate Q function using Neural Network
    # State is the input and the Q Values are the output.
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Edit the Neural Network model here
    # Tip: Consult https://keras.io/getting-started/sequential-model-guide/
    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        # model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # After some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    def get_action(self, state):
        if random.uniform(0, 1) > self.epsilon:
            actions = self.model.predict(state)
            action = np.argmax(actions)
        else:
            action = random.randrange(self.action_size)
        return action

    #Save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # Add sample to the end of the list

    # Sample <s,a,r,s'> from replay memory
    def train_model(self):
        if len(self.memory) < self.train_start: # Do not train if not enough memory
            return
        batch_size = min(self.batch_size, len(self.memory)) # Train on at most as many samples as you have in memory
        mini_batch = random.sample(self.memory, batch_size) # Uniformly sample the memory buffer
        # Preallocate network and target network input matrices.
        update_input = np.zeros((batch_size, self.state_size)) # batch_size by state_size two-dimensional array (not matrix!)
        update_target = np.zeros((batch_size, self.state_size)) # Same as above, but used for the target network
        action, reward, done = [], [], [] # Empty arrays that will grow dynamically

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0] #Allocate s(i) to the network input array from iteration i in the batch
            action.append(mini_batch[i][1]) #Store a(i)
            reward.append(mini_batch[i][2]) #Store r(i)
            update_target[i] = mini_batch[i][3] #Allocate s'(i) for the target network array from iteration i in the batch
            done.append(mini_batch[i][4])  #Store done(i)

        target = self.model.predict(update_input) #Generate target values for training the inner loop network using the network model
        target_val = self.target_model.predict(update_target) #Generate the target values for training the outer loop target network

        #Q Learning: get maximum Q value at s' from target network
        for i in range(self.batch_size):
            target[i][action[i]] = reward[i]
            if not done[i]:
                target[i][action[i]] = reward[i] + self.discount_factor*np.max(target_val[i])

        #Train the inner loop network
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
        return
    
    # Plots the score per episode as well as the maximum q value per episode, averaged over precollected states.
    def plot_data(self, episodes, scores, max_q_mean):
        pylab.figure(0)
        pylab.plot(episodes, max_q_mean, 'b')
        pylab.xlabel("Episodes")
        pylab.ylabel("Average Q Value")
        pylab.savefig("figures/qvalues.png")

        pylab.figure(1)
        pylab.plot(episodes, scores, 'b')
        pylab.xlabel("Episodes")
        pylab.ylabel("Score")
        pylab.savefig("figures/scores.png")

    # Load model params
    def save(self, name):
        path = "models/" + name
        self.model.save_weights(path + ".h5")
        # self.target_model.save_weights(path + "_target.h5")

    # Load network weights
    def load(self, name):
        path = "models/" + name
        self.model.load_weights(path + ".h5")
        self.target_model.load_weights(path + ".h5")
        # self.target_model.load_weights(path + "_target.h5")


def sample_states(agent):
    test_states = np.zeros((agent.test_state_no, state_size))
    done = True
    for i in range(agent.test_state_no):
        if done:
            done = False
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            test_states[i] = state
        else:
            action = random.randrange(action_size)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            test_states[i] = state
            state = next_state
    return test_states


if __name__ == "__main__":
    # For CartPole-v0, maximum episode length is 200
    env = gym.make('CartPole-v0') # Generate Cartpole-v0 environment object from the gym library
    # Get state and action sizes from the environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create agent, see the DQNAgent __init__ method for details
    agent = DQNAgent(state_size, action_size)

    # Collect test states for plotting Q values using uniform random policy
    test_states = sample_states(agent)  # Sample random states for plotting
    max_q = np.zeros((EPISODES, agent.test_state_no))
    max_q_mean = np.zeros((EPISODES,1))
    
    scores, episodes = [], [] # Create dynamically growing score and episode counters
    t = trange(EPISODES, desc='Bar desc', leave=True)
    for e in t:
        done = False
        score = 0
        state = env.reset() # Initialize/reset the environment
        state = np.reshape(state, [1, state_size]) # Reshape state so that to a 1 by state_size two-dimensional array ie. [x_1,x_2] to [[x_1,x_2]]
        # Compute Q values for plotting
        tmp = agent.model.predict(test_states)
        max_q[e][:] = np.max(tmp, axis=1)
        max_q_mean[e] = np.mean(max_q[e][:])

        while not done:
            if agent.render:
                env.render() # Show cartpole animation

            # Get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size]) # Reshape next_state similarly to state

            # Save sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            # Training step
            agent.train_model()
            score += reward # Store episodic reward
            state = next_state # Propagate state

            if done:
                # At the end of very episode, update the target network
                if e % agent.target_update_frequency == 0:
                    agent.update_target_model()
                # Plot the play time for every episode
                scores.append(score)
                episodes.append(e)

                t.set_description("Score: " + str(np.round(score)) + ", Q: " + str(np.round(max_q_mean[e][0])))
                t.refresh() # to show immediately the update

                #  if the mean of scores of last 100 episodes is bigger than 195
                #  stop training
                if agent.check_solve:
                    if np.mean(scores[-min(100, len(scores)):]) >= 195:
                        print("solved after", e-100, "episodes")
                        agent.plot_data(episodes,scores,max_q_mean[:e+1])
                        sys.exit()
    agent.save("test1")
    agent.plot_data(episodes,scores,max_q_mean)

    print("Simulating environments...")
    agent_value = evaluate_agent(env = env, agent = agent, tests_num = 100)
    print("Agent value:", agent_value)

