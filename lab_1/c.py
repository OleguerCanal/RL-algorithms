import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from b import State

map = np.ones((7, 8))
map[0:4, 2] = 0
map[5, 1:7] = 0
map[5, 1:7] = 0
map[6, 4] = 0
map[1:4, 5] = 0
map[2, 6:8] = 0

c_rewards = {"Win": 10000, "Minotaur":-10000, "Step": -1}

def value_iteration_infinite(value, lambd):
    max_iters = 10000
    tol = 1e-5
    cont = 0

    for iter in tqdm(range(max_iters)):
        new_value = np.zeros(value.shape)
        for coords, _ in np.ndenumerate(value[:, :, :, :]):
            state = State(coords, c_rewards)
            actions = state.get_actions()
            if actions == []:
                new_value[coords] = state.reward()
                continue
            new_value[coords] = \
                    max(state.reward() + lambd * \
                        sum(p * value[next_state.get_coord()] \
                        for next_state, p in state.get_transitions(action)) \
                                for action in actions)
 
        delta = np.linalg.norm(np.matrix.flatten(value) - np.matrix.flatten(new_value))
        value = new_value
        cont += 1
        if delta < tol * (1 - lambd) / lambd:
            print("Converged after "+str(cont)+" iterations")
            print("Delta: "+str(delta))
            return value

def policy(state, values):
    best_action = None
    max_found = -float('inf')
    for action in state.get_actions():
        val = state.reward()
        for next_state, p in state.get_transitions(action):
            xp, yp, xm, ym = next_state.get_coord()
            val += p*values[xp, yp, xm, ym]
        if val >= max_found:
            max_found = val
            best_action = action
    if state.get_actions() == []:
        print(state)
        print("Error")
    return best_action

def generate_game(values, initial_state, deadline):
    state = initial_state
    states = []

    for T in range(deadline):
        states.append(state)
        if state.dead():
            return states, "Minotaur"
        if state.free():
            return states, "Win"

        action = policy(state, values)
        transitions = state.get_transitions(action)
        state = transitions[np.random.choice(range(len(transitions)))][0]
    return states, "Deadline"

def test_geometric(deadlines, n_games):
        value = np.zeros((map.shape[0], map.shape[1], map.shape[0], map.shape[1]))
        value[6, 5, :, :] = 1
        value[6, 5, 6, 5] = 0
        value = value_iteration_infinite(value, 29./30.)

        initial_state = State(rewards=c_rewards)
        xp, yp, xm, ym = initial_state.get_coord()
        
        wins = []
        for t in deadlines:
            endgames = {"Minotaur":0, "Deadline":0, "Win":0}
            for i in range(n_games):
                states, result = generate_game(value, initial_state, t)
                endgames[result] += 1
            print("Time: {}, Wins: {}, Minotaur: {}, Deadline: {}".format(t, endgames["Win"],\
                    endgames["Minotaur"], endgames["Deadline"]))
            wins.append(float(endgames["Win"])/n_games)
        return wins


if __name__ == "__main__":
    deadlines = [16, 17, 18, 19, 20, 22, 24, 25, 30, 35, 40]
    n_games = 100
    wins = test_geometric(deadlines, n_games)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(deadlines, wins, '.b-', label='Percentage of wins over '+str(n_games)+' games')
    ax1.legend()
    plt.title('Number of wins for increasing deadlines with discounted rewards, lambda = 29/30')
    plt.plot(deadlines, wins)
    plt.show()

