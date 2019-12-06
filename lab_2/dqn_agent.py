from collections import deque
import copy
import gym
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential, model_from_json
import numpy as np
import pylab
import random
import sys
from tqdm import tqdm, trange

class DQNAgent:
    ''' Deep QN Agent with experience replay and target network
    '''
    def __init__(self, environment, parameters):
        # Set parameters TODO(oleguer): Pass as kwargs??
        self.discount_factor = parameters["discount_factor"] 
        self.learning_rate = parameters["learning_rate"]
        self.epsilon = parameters["epsilon"]
        self.batch_size = parameters["batch_size"]
        self.memory_size = parameters["memory_size"]
        self.train_start = parameters["train_start"]
        self.target_update_frequency = parameters["target_update_frequency"]

        # Private vars
        self.__environment = environment
        self.__state_size = environment.observation_space.shape[0]
        self.__action_size = environment.action_space.n
        self.__memory = deque(maxlen=self.memory_size)  # Memory buffer

        # Load model
        if "model" in parameters and parameters["model"] is not None:
            self.model = copy.deepcopy(parameters["model"])
            self.target_model = copy.deepcopy(parameters["model"])
        else:
            self.model = self.build_model() 
            self.target_model = self.build_model()

        # Initialize target network TODO(oleguer): Not sure this is doing anything
        self.__update_target_model()

    def build_model(self):
        '''NN used for Q learning
        '''
        model = Sequential()
        model.add(Dense(32, input_dim=self.__state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        # model.add(Dense(16, activation='relu'))
        model.add(Dense(self.__action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        # model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def save(self, name):
        '''Save QNN model and params to models/ folder
        '''
        path = "models/" + name
        model_json = self.model.to_json()
        with open(path + "_architecture.json", 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights(path + ".h5")
        # self.target_model.save_weights(path + "_target.h5")

    def load(self, name):
        '''Load saved QNN
        '''
        path = "models/" + name
        json_file = open(path + "_architecture.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = copy.deepcopy(model_from_json(loaded_model_json))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        self.target_model = copy.deepcopy(model_from_json(loaded_model_json))
        self.target_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        self.model.load_weights(path + ".h5")
        self.target_model.load_weights(path + ".h5")

    def train(self, name, episode_num = 1000, solved_score = None, test_states_num = 1000):
        '''Train agent in loaded environment
        '''
        print("Training model...")
        test_states = self.__sample_states(test_states_num)  # Sample random states for plotting
        max_q_mean = np.zeros((episode_num,1))
        
        scores, episodes = [], [] # Create dynamically growing score and episode counters
        t = trange(episode_num, desc='Bar desc', leave=True)
        for e in t:
            done = False
            score = 0
            state = self.__environment.reset() # Initialize/reset the environment

            # Follow training
            max_q_mean[e] = np.mean(np.max(self.model.predict(test_states), axis=1)) # Get Q values to

            while not done:
                action = self.__get_epsilon_greedy_action(state)
                next_state, reward, done, info = self.__environment.step(action)
                self.__append_sample(state, action, reward, next_state, done)
                self.__train_model()  # Update model
                score += reward
                state = next_state

                if done:
                    self.save(name)  # Save model after each epoch
                    if e % self.target_update_frequency == 0:  # Update target weights
                        self.__update_target_model()

                    # Tracking metrics  TODO(oleguer): Use tensorboard for this
                    scores.append(score)
                    episodes.append(e)
                    t.set_description("Score: " + str(np.round(score)) + ", Q: " + str(np.round(max_q_mean[e][0])))
                    t.refresh()

                    if solved_score:  # Stop training if last 100 scores > solved_score
                        if np.mean(scores[-min(100, len(scores)):]) >= solved_score:
                            print("Solved after", e-100, "episodes")
                            self.__plot_data(episodes,scores,max_q_mean[:e+1])
                            break
        self.__plot_data(episodes, scores, max_q_mean)

    def test(self, tests_num, render = False):
        rewards = np.zeros(tests_num)
        i = 0
        while i < tests_num:
            done = False
            state = self.__environment.reset()
            while not done:
                if render:
                    self.__environment.render()
                action = self.get_greedy_action(state)
                state, reward, done, _ = self.__environment.step(action)
                rewards[i] += reward
            i += 1
        mean_reward = np.mean(rewards)
        print("Test mean reward: " + str(mean_reward))
        return mean_reward

    def get_greedy_action(self, state):
        ''' Returns greedy action
        '''
        state = np.reshape(state, [1, self.__state_size])  # Reshape to tensorflow notation 
        actions = self.model.predict(state)
        return np.argmax(actions)

    # Private functions
    def __update_target_model(self):
        '''After some time interval update the target model to be same with model
        '''
        self.target_model.set_weights(self.model.get_weights())

    def __get_epsilon_greedy_action(self, state):
        ''' Returns an epsilon-greedy action
        '''
        state = np.reshape(state, [1, self.__state_size])  # Reshape to tensorflow notation 
        if random.uniform(0, 1) > self.epsilon:
            actions = self.model.predict(state)
            return np.argmax(actions)
        else:
            return random.randrange(self.__action_size)

    def __append_sample(self, state, action, reward, next_state, done):
        '''Save sample <s,a,r,s'> to the replay memory
        '''
        self.__memory.append((state, action, reward, next_state, done)) # Add sample to the end of the list

    def __train_model(self):
        '''Samples self.batch_size from replay memory and updates NN in one pass
        '''
        if len(self.__memory) < self.train_start: # Do not train if not enough memory
            return
        batch_size = min(self.batch_size, len(self.__memory)) # Train on at most as many samples as you have in memory
        mini_batch = random.sample(self.__memory, batch_size) # Uniformly sample the memory buffer
        # Preallocate network and target network input matrices.
        update_input = np.zeros((batch_size, self.__state_size)) # batch_size by __state_size two-dimensional array (not matrix!)
        update_target = np.zeros((batch_size, self.__state_size)) # Same as above, but used for the target network
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
    
    def __plot_data(self, episodes, scores, max_q_mean):
        """Very crappy plotting
        """
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

    def __sample_states(self, test_states_num):
        '''Applies random policy to sample states, used to evaluate Q average value
        '''
        test_states = np.zeros((test_states_num, self.__state_size))
        done = True
        for i in range(test_states_num):
            if done:
                done = False
                state = env.reset()
                test_states[i] = state
            else:
                action = random.randrange(self.__action_size)
                next_state, reward, done, info = env.step(action)
                test_states[i] = state
                state = next_state
        return test_states


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    parameters = {
        "discount_factor": 0.95,
        "learning_rate": 0.005,
        "memory_size": 1000,
        "target_update_frequency": 1,
        "epsilon": 0.02, # Fixed
        "batch_size": 32,  # Fixed
        "train_start": 1000, # Fixed
        "model": None,  # Pass custom model (None => defined by build_model)
    }

    agent = DQNAgent(environment = env, parameters = parameters)
    agent.train(name = "test1", episode_num=1000)
    # agent.load(name = "test1")
    average_score = agent.test(tests_num=5, render = True)



