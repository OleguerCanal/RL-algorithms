from typing import List
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
#     last = scalars[0]  # First value in the plot (first timestep)
#     smoothed = list()
#     for point in scalars:
#         smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
#         smoothed.append(smoothed_val)                        # Save it
#         last = smoothed_val                                  # Anchor the last smoothed value
#     return smoothed

def smooth(values, alpha, epsilon = 0):
   if not 0 < alpha < 1:
      raise ValueError("out of range, alpha='%s'" % alpha)
   if not 0 <= epsilon < alpha:
      raise ValueError("out of range, epsilon='%s'" % epsilon)
   result = [None] * len(values)
   for i in range(len(result)):
       currentWeight = 1.0
       numerator     = 0
       denominator   = 0
       for value in values[i::-1]:
           numerator     += value * currentWeight
           denominator   += currentWeight
           currentWeight *= alpha
           if currentWeight < epsilon: 
              break
       result[i] = numerator / denominator
   return result

if __name__ == "__main__":
    path = "csvs/question_j"
    # smoothing_factor = 0.993
    smoothing_factor = 0.1

    fig, ax = plt.subplots(1,1)
    # fig, (ax1, ax2) = plt.subplots(1,1, figsize=(10,4), sharey=True, dpi=120)
    files = np.array(os.listdir(os.fsencode(path)))
    # files = files[[2, 1, 4, 3, 0]]  # g_q
    # files = files[[2, 0, 1, 3, 4]]  # g
    # files = files[[0, 3, 1, 2]]  # i
    for file in files:
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, filename))
            values = df["Value"].tolist()
            smoothed_values = smooth(values, smoothing_factor)
            # times = np.array(df["Wall time"])[:len(smoothed_values)]/
            times = np.array(df["Wall time"])[:len(smoothed_values)]/6000.
            times -= times[0]  # Make it relative
            name = filename.replace("run-CartPole-v0_", "").split("_2019")[0]
            name = name.replace("_", " ").replace("-", ":")
            ax.plot(times, smoothed_values,\
                label=name)
    # Title, X and Y labels, X and Y Lim
    ax.set_title('')
    plt.ylim((0, 200))
    plt.xlim((0, 0.07))
    ax.set_xlabel('Relative Time [hours]')
    ax.set_ylabel('Score [Points]')
    plt.legend()
    fig.savefig("figures/"+path.split("/")[-1])
    plt.show()