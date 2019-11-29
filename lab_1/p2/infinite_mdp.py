import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from tqdm import tqdm

from b import State, map

def value_iteration_infinite(value, lambd):
    max_iters = 10000
    tol = 1e-5
    cont = 0

    for iter in tqdm(range(max_iters)):
        new_value = np.zeros(value.shape)
        for coords, _ in np.ndenumerate(value[:, :, :, :]):
            state = State(coords)
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
            print("Converged after " + str(cont) + " iterations")
            print("Delta: " + str(delta))
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

    deaths_count = 0
    reward_count = 0
    for T in range(deadline):
        if np.array_equal(state.player, state.mino):
            deaths_count += 1
        states.append((state, reward_count))
        reward_count += state.reward()
        action = policy(state, values)
        transitions = state.get_transitions(action)
        state = transitions[np.random.choice(range(len(transitions)))][0]
    return states, deaths_count, reward_count

def test_geometric(deadlines, n_games, lambd):
        value = np.zeros((map.shape[0], map.shape[1], map.shape[0], map.shape[1]))
        value[6, 5, :, :] = 1
        value[6, 5, 6, 5] = 0
        value = value_iteration_infinite(value, lambd)

        initial_state = State()
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

def plot(values, police, save = False, path = "", ind = 0):
    x = []
    y = []
    ax = []
    ay = []
    for index, val in np.ndenumerate(values[:, :, :, :]):
        if index[2] == police[0] and index[3] == police[1]:
            s = State((index[0], index[1], police[0], police[1]))
            best_action = policy(s, values)
            if best_action is not None:
                x.append(index[1])
                y.append(index[0])
                ax.append(int(best_action[1]))
                ay.append(-int(best_action[0]))

    s = State((0, 0, 0, 0))
    heatmap = np.zeros(map.shape)
    heatmap[s.banks[0][0]][s.banks[0][1]] = 1
    heatmap[s.banks[1][0]][s.banks[1][1]] = 1
    heatmap[s.banks[2][0]][s.banks[2][1]] = 1
    heatmap[s.banks[3][0]][s.banks[3][1]] = 1
    heatmap[police[0]][police[1]] = 0.5
    cmap = colors.ListedColormap(['white', 'red', 'blue'])
    plt.imshow(heatmap, cmap=cmap, interpolation='nearest')

    plt.quiver(x, y, ax, ay, scale=2.5, scale_units='x')
    if save:
        plt.savefig(path + "/fig" + str(ind) + ".png")
    plt.plot()
    plt.pause(0.25)
    plt.close()