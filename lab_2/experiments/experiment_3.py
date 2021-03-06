from dqn_agent import DQNAgent, perform_experiment
import gym
import numpy as np
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.models import Sequential, model_from_json

def get_model1(input_size, output_size, lr):
    model = Sequential()
    model.add(Dense(32, input_dim=input_size, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(output_size, activation='linear',
                    kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    return model

if __name__ == "__main__":
    models = [get_model1]
    discount_factors = [0.99]
    learning_rates = [0.003]
    memory_sizes = [10000]
    update_frequencies = [1, 10, 50]

    perform_experiment(models = models,
                        discount_factors = discount_factors,
                        learning_rates = learning_rates,
                        memory_sizes = memory_sizes,
                        update_frequencies = update_frequencies,
                        episodes = 1000)