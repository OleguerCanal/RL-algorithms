from dqn_agent import DQNAgent, perform_experiment
import gym
import numpy as np
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.models import Sequential, model_from_json

def get_model1(input_size, output_size, lr):
    model = Sequential()
    model.add(Dense(16, input_dim=input_size, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(output_size, activation='linear',
                    kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    return model

def get_model2(input_size, output_size, lr):
    model = Sequential()
    model.add(Dense(64, input_dim=input_size, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(output_size, activation='linear',
                    kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    return model

def get_model3(input_size, output_size, lr):
    model = Sequential()
    model.add(Dense(16, input_dim=input_size, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(16, input_dim=input_size, activation='relu',
                kernel_initializer='he_uniform'))
    model.add(Dense(output_size, activation='linear',
                    kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    return model

if __name__ == "__main__":
    models = [get_model1, get_model2, get_model3]
    memory_sizes = [1000, 10000, 100000]
    discount_factors = [0.8, 0.9, 0.99]
    learning_rates = [0.001, 0.005, 0.05]
    target_update_frequencies = [1, 10, 50]

    scores = []
    for model in models:

        env = gym.make('CartPole-v0')

        parameters = {
            "env" : env,
            "discount_factor": 0.9,
            "learning_rate": 0.002,
            "memory_size": 10000,
            "target_update_frequency": 2,
            "epsilon": 0.02, # Fixed
            "batch_size": 32,  # Fixed
            "train_start": 1000, # Fixed
            "model": model
        }

        experiment_name = generate_experiment_name(parameters)
        print(experiment_name)
        
        agent = DQNAgent(parameters = parameters)
        agent.train(name = experiment_name, episode_num = 5000, solved_score = 195)
        # agent.load(name = experiment_name)
        average_score = agent.test(tests_num=100, render = False)
        # scores.append(average_score)
        del agent
        del env
        del parameters

    # np.save("metrics/test_scores.npy", scores)
