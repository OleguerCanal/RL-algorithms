import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def policy_b(sequence):
    if sequence[0]:
        return 1.
    else:
        return float(np.sum(sequence))/len(sequence)

def policy_c(sequence):
    for t in range(len(sequence)):
        n = np.sum(sequence[0:t])
        if n > float(t)/2:
            return float(n)/t
    return float(np.sum(sequence))/len(sequence)

if __name__ == "__main__":

    c = []
    b = []
    for T in tqdm(range(100, 2000, 100)):
        games = 3000
        policy_b_average = 0
        policy_c_average = 0
        for _ in range(games):
            seq = np.where(np.random.uniform(size = T) < 0.5, 0, 1)
            policy_b_average += policy_b(seq)/float(games)
            policy_c_average += policy_c(seq)/float(games)

        b.append(policy_b_average)    
        c.append(policy_c_average)    
        # print("Policy b: " + str(policy_b_average))
        # print("Policy c: " + str(policy_c_average))
    
    plt.plot(b, color = 'b')
    plt.plot(c, color = 'r')
    plt.show()