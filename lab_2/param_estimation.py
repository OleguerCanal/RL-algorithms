import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import gym
from gaussian_process import GaussianProcess, save, load
from dqn_agent import DQNAgent, generate_experiment_name
import itertools as it

def get_model(nlayers, nunits, lr):
    model = Sequential()
    model.add(Dense(int(nunits), input_dim=4, activation='sigmoid',
                    kernel_initializer='he_uniform'))
    for i in range(int(nlayers)):
        model.add(Dense(int(nunits), activation='sigmoid', kernel_initializer='he_uniform'))
    model.add(Dense(2, activation='linear', kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    return model
    

def eval_agent(point):
    # TODO: (Federico) This is a mess, find a better way to convert parameters
    # point = [df, lr, memsize, updatefreq, nlayers, nunits]
    model = get_model(point[4], point[5], point[1])
    parameters = {"discount_factor":point[0],
                  "learning_rate": point[1],
                  "memory_size":point[2],
                  "target_update_frequency":point[3],
                  "train_start":1000,
                  "epsilon":0.02,
                  "batch_size":32,
                  "env":env,
                  "full_model":model,
                  "model":None #TODO merge full_model and model in a single function that
                               # takes aslo the number of layers. I didn't do it because
                               # there may be lot of usages in the code I'm not aware of
                 }
    print('Evaluating at '+str(parameters)+', nlayers: '+str(point[4])+
          ', nunits: '+str(point[5]))
    agent = DQNAgent(parameters)
    res = agent.train(generate_experiment_name(parameters), episode_num=1000, solved_score=195)
    print('Evaluation result: '+str(res))
    return res

save_dir = 'gp_saves'
env = gym.make('CartPole-v0')
discount_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
discount_factors.reverse()
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2]
memory_sizes = [1000, 10000, 100000]
update_frequencies = [1, 10, 50, 100]
n_layers = [0, 1, 2]
n_units = [4, 8, 16, 32, 64]

param_space = [discount_factors, learning_rates, memory_sizes,
               update_frequencies, n_layers, n_units]

gp = GaussianProcess(space_dim=len(param_space), length_scale=100)
# Uncomment this to start from saved values
#known_points, known_values = load('saved_evaluation')
#gp.add_points(known_points, known_values)
#eval_point = gp.most_likely_max(param_space)
eval_point = [discount_factors[0],
              learning_rates[0],
              memory_sizes[0],
              update_frequencies[0],
              n_layers[0],
              n_units[0]]

while True:
    val = eval_agent(eval_point)
    gp.add_points([eval_point], [val])
    save(save_dir,gp.known_points, gp.known_values)
    candidate_max = gp.most_likely_max(param_space)
    eval_point = candidate_max
