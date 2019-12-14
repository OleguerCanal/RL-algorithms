from dqn_agent import DQNAgent, perform_experiment, generate_experiment_name
import gym
import numpy as np
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.models import Sequential, model_from_json
from plotting import smooth
from typing import List
import matplotlib.pyplot as plt
import os
import pandas as pd


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
    discount_factors = [0.8, 0.9, 0.99]
    learning_rates = [0.001, 0.005, 0.05]
    memory_sizes = [1000, 10000, 100000]
    update_frequencies = [1]

    path = "csvs/question_h"
    smoothing_factor = 0.995

    # wrt = "df"
    # wrt = "lr"
    # wrt = "ms"
    for wrt in ["df", "lr", "ms"]:

        files = np.array(os.listdir(os.fsencode(path)))
        file_names = [os.fsdecode(name) for name in files]
        file_names_short = [filename.replace("run-CartPole-v0_", "").split("_2019")[0] for filename in file_names]
        print(file_names_short)
        env = gym.make('CartPole-v0')
        
        fig, ax = plt.subplots(3,3)
        for model in models:
            for i, df in enumerate(discount_factors):
                for j, lr in enumerate(learning_rates):
                    for k, ms in enumerate(memory_sizes):
                        for uf in update_frequencies:
                            parameters = {
                                "env" : env,
                                "discount_factor": df,
                                "learning_rate": lr,
                                "memory_size": ms,
                                "target_update_frequency": uf,
                                "epsilon": 0.02, # Fixed
                                "batch_size": 32,  # Fixed
                                "train_start": 1000, # Fixed
                                "model": model
                            }
                            namee = generate_experiment_name(parameters, "")
                            namee = namee.replace("CartPole-v0_", "").split("_2019")[0]

                            indx = 0
                            indy = 0
                            title = ""
                            if wrt == "df":
                                indx = j
                                indy = k
                                label = "df:" + str(df)
                            if wrt == "lr":
                                indx = i
                                indy = k
                                label = "lr:" + str(lr)
                            if wrt == "ms":
                                indx = i
                                indy = j
                                label = "ms:" + str(ms)

                            try:
                                filename = file_names[file_names_short.index(namee)]
                                dataframe = pd.read_csv(os.path.join(path, filename))
                                values = dataframe["Value"].tolist()
                                smoothed_values = smooth(values, smoothing_factor)
                                times = np.array(dataframe["Wall time"])[:len(smoothed_values)]/6000.
                                times -= times[0]  # Make it relative
                                name = filename.replace("run-CartPole-v0_", "").split("_2019")[0]
                                name = name.replace("_", " ").replace("-", ":")
                                # Plot in corresponding axes
                                ax[indy][indx].plot(times, smoothed_values, label=label)
                                # ax[indx][indy].set_title(title)
                                
                            except Exception as e:
                                print(e)
                                # print("Mising:")
                                # print(filename)
        # plt.set_xlabel('Relative Time [hours]')
        # plt.set_ylabel('Score [Points]')

        # Set titles
        if wrt == "df":
            x_name = "lr = "
            y_name = "ms = "
            x_vals = learning_rates
            y_vals = memory_sizes
        if wrt == "lr":
            x_name = "df = "
            y_name = "ms = "
            x_vals = discount_factors
            y_vals = memory_sizes
        if wrt == "ms":
            x_name = "df = "
            y_name = "lr = "
            x_vals = discount_factors
            y_vals = learning_rates
        
        ax[0][0].set_ylabel(y_name + str(y_vals[0]), fontsize = 12)
        ax[1][0].set_ylabel(y_name + str(y_vals[1]), fontsize = 12)
        ax[2][0].set_ylabel(y_name + str(y_vals[2]), fontsize = 12)
        ax[0][0].set_title(x_name + str(x_vals[0]), fontsize = 12)
        ax[0][1].set_title(x_name + str(x_vals[1]), fontsize = 12)
        ax[0][2].set_title(x_name + str(x_vals[2]), fontsize = 12)
        

        xlim = (0, 1.6)
        ylim = (0, 200)

        # Setting the values for all axes.
        plt.legend()
        fig.set_size_inches(10, 6)
        for a in ax:
            plt.setp(a, xlim=xlim, ylim=ylim)
        fig.savefig("figures/"+path.split("/")[-1] + "_" + wrt)


        # plt.show()

