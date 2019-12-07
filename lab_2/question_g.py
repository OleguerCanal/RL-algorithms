from dqn_agent import DQNAgent, generate_experiment_name
import gym
from keras.layers import Dense
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
    model.add(Dense(32, input_dim=input_size, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(output_size, activation='linear',
                    kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    return model

def get_model22(input_size, output_size, lr):
    model = Sequential()
    model.add(Dense(50, input_dim=input_size, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(output_size, activation='linear',
                    kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    return model

def get_model3(input_size, output_size, lr):
    model = Sequential()
    model.add(Dense(16, input_dim=input_size, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(output_size, activation='linear',
                    kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    return model

def get_model4(input_size, output_size, lr):
    model = Sequential()
    model.add(Dense(16, input_dim=input_size, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(output_size, activation='linear',
                    kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    return model

def get_model5(input_size, output_size, lr):
    model = Sequential()
    model.add(Dense(8, input_dim=input_size, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(output_size, activation='linear',
                    kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    return model


if __name__ == "__main__":
    models = [get_model1, get_model2, get_model22, get_model3, get_model4, get_model5]
    # TODO(oleguer): Test also different loss (binnary cross entropy could work better) 

    for model in models:
        env = gym.make('CartPole-v0')

        parameters = {
            "env" : env,
            "discount_factor": 0.95,
            "learning_rate": 0.005,
            "memory_size": 1000,
            "target_update_frequency": 1,
            "epsilon": 0.02, # Fixed
            "batch_size": 32,  # Fixed
            "train_start": 1000, # Fixed
            "model": model
        }

        experiment_name = generate_experiment_name(parameters)
        print(experiment_name)
        
        agent = DQNAgent(parameters = parameters)
        agent.train(name = experiment_name, episode_num = 10000)
        # agent.load(name = experiment_name)
        # average_score = agent.test(tests_num=1, render = True)
        del agent
        del env
        del parameters
