from dqn_agent import DQNAgent, generate_experiment_name
import gym
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential, model_from_json


def get_model(input_size, output_size, lr):
    ''' Model definition (parameters will be filled automatically by caller)
    '''

    model = Sequential()
    model.add(Dense(16, input_dim=input_size, activation='relu',
                    kernel_initializer='he_uniform'))
    # model.add(Dense(16, activation='relu'))
    model.add(Dense(output_size, activation='linear',
                    kernel_initializer='he_uniform'))
    # model.summary()
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    return model


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    # # env = gym.make('MountainCar-v0')

    parameters = {
        "env" : env,
        "discount_factor": 0.8,
        "learning_rate": 0.005,
        "memory_size": 1000,
        "target_update_frequency": 1,
        "epsilon": 0.02, # Fixed
        "batch_size": 32,  # Fixed
        "train_start": 1000, # Fixed
        "model": get_model
    }

    experiment_name = generate_experiment_name(parameters)
    print(experiment_name)
    
    agent = DQNAgent(parameters = parameters)
    agent.train(name = experiment_name, episode_num = 10000)
    # agent.load(name = experiment_name)

    average_score = agent.test(tests_num=1, render = True)