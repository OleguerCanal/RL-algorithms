import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from tqdm import tqdm

map = np.ones((7, 8))
map[0:4, 2] = 0
map[5, 1:7] = 0
map[5, 1:7] = 0
map[6, 4] = 0
map[1:4, 5] = 0
map[2, 6:8] = 0

class State:
    goal = (6, 5)
    player_actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    mino_actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

    def __init__(self, coord=(0, 0, 6, 5)):
        self.player = (coord[0], coord[1])
        self.mino = (coord[2], coord[3])
    
    def get_coord(self):
        return self.player[0], self.player[1], self.mino[0], self.mino[1]

    def dead(self):
        return self.player[0] == self.mino[0] and self.player[1] == self.mino[1]
    
    def free(self):
        return self.player[0] == State.goal[0] and self.player[1] == State.goal[1]

    def get_actions(self):
        if self.free():
            return []
        if self.dead():
            return []
        actions = []
        for action in State.player_actions:
            next_pos = np.array(self.player) + np.array(action)
            if next_pos[0] < 0 or next_pos[0] >= map.shape[0] or\
                next_pos[1] < 0 or next_pos[1] >= map.shape[1] or\
                map[next_pos[0], next_pos[1]] == 0 or\
                map[self.player[0], self.player[1]] == 0:
                continue
            actions.append(action)
        return actions

    def get_transitions(self, action):
        next_state = State(self.get_coord())
        next_state.player = tuple(np.array(next_state.player) + np.array(action))

        possible_states = []
        for mino_action in State.mino_actions:
            next_pos = np.array(next_state.mino) + np.array(mino_action)
            if next_pos[0] < 0 or next_pos[0] >= map.shape[0] or\
                next_pos[1] < 0 or next_pos[1] >= map.shape[1]:
                continue
            possible_state = State(next_state.get_coord())
            possible_state.mino = tuple(next_pos)
            possible_states.append(possible_state)

        prob = 1./len(possible_states)

        result = []
        for a in possible_states:
            result.append((a, prob))
        return result

    def reward(self):
        if self.dead():
            return 0
        if self.free():
            return 1
        return 0

    def __str__(self):
        return "Player:" + str(self.player) + ", " + "Mino:" + str(self.mino)

def policy(state, T, values):
    best_action = None
    max_found = state.reward()
    for action in state.get_actions():
        val = state.reward()
        #val = -float('-inf')
        for next_state, p in state.get_transitions(action):
            xp, yp, xm, ym = next_state.get_coord()
            val += p*values[xp, yp, xm, ym, T + 1]
        if val >= max_found:
            max_found = val
            best_action = action
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

        action = policy(state, T, values)
        transitions = state.get_transitions(action)
        state = transitions[np.random.choice(range(len(transitions)))][0]
    return states, "Deadline"

def get_heat_map(values, minotaur_pos, T):
    minotaur_pos = tuple(minotaur_pos)
    heatmap = np.zeros(map.shape)

    for index, val in np.ndenumerate(values[:, :, :, :, T]):
        if index[2] == minotaur_pos[0] and index[3] == minotaur_pos[1]:
            heatmap[index[0], index[1]] = val
    return heatmap

def get_policy_arrows(values, mino_pos, T):
    mino_pos = tuple(mino_pos)
    x = []
    y = []
    ax = []
    ay = []

    for player_pos in np.c_[np.where(map == 1)]:
        player_pos = tuple(player_pos)
        if player_pos == (6, 5) or player_pos == mino_pos:
            continue
        state = State(tuple([player_pos[0], player_pos[1], mino_pos[0], mino_pos[1]]))
        action = policy(state, T, values)
        x.append(player_pos[1])
        y.append(player_pos[0])
        ax.append(action[1])
        ay.append(-action[0])
    return x, y, ax, ay


def value_iteration(value, T):
    for T in tqdm(range(T-1, -1, -1)):
        for index, _ in np.ndenumerate(value[:, :, :, :, T]):
            state = State(index)
            max_found = state.reward()
            for action in state.get_actions():
                val = 0
                for next_state, p in state.get_transitions(action):
                    xp, yp, xm, ym = next_state.get_coord()
                    val += p*value[xp, yp, xm, ym, T+1]
                max_found = max(max_found, val)

            xp, yp, xm, ym = index
            value[xp, yp, xm, ym, T] = max_found
    return value

def train_and_test(deadlines = [20]):
    theoretical_successes = []
    experiment_successes = []

    for T in deadlines:
        print("Deadline: " + str(T))

        value = np.zeros((map.shape[0], map.shape[1], map.shape[0], map.shape[1], T))
        value[6, 5, :, :, T-1] = 1
        value[6, 5, 6, 5, T-1] = 0
        value = value_iteration(value, T-1)

        initial_state = State()
        xp, yp, xm, ym = initial_state.get_coord()
        print("Escape prob: " + str(value[xp, yp, xm, ym, 0]))

        n_games = 1000
        endgames = {"Minotaur":0, "Deadline":0, "Win":0}
        for i in range(n_games):
            states, result = generate_game(value, initial_state, T)
            endgames[result] += 1
        print("Wins: {}, Minotaur: {}, Deadline: {}".format(endgames["Win"],\
                endgames["Minotaur"], endgames["Deadline"]))
            
        theoretical_successes.append(value[xp, yp, xm, ym, 0])
        experiment_successes.append(endgames["Win"]/n_games)

    print(deadlines)
    print(theoretical_successes)
    print(experiment_successes)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(deadlines, theoretical_successes, '.b-', label='theoretical')
    ax1.plot(deadlines, experiment_successes, '.r-', label='experiment')
    plt.legend(loc='upper left')
    plt.show() 
    
if __name__ == "__main__":
    #train_and_test([20])
    
    T = 20
    value = np.zeros((map.shape[0], map.shape[1], map.shape[0], map.shape[1], T))
    value[6, 5, :, :, T-1] = 1
    value[6, 5, 6, 5, T-1] = 0
    value = value_iteration(value, T-1)

    x, y, ax, ay = get_policy_arrows(value, (4, 3), 1)
    cmap = colors.ListedColormap(['black','red','green', 'white'])
    heatmap = map
    heatmap[4, 3] = 0.3
    heatmap[6, 5] = 0.6
    plt.imshow(heatmap, cmap=cmap, interpolation='nearest')
    plt.quiver(x, y, ax, ay)
    plt.show()
    # initial_State = State()
    # steps, result = generate_game(value, initial_State, T)
    # for id, step in enumerate(steps):
    #     print(id)
    #     print(step)
    # print(result)

    # for t in range(T-1, 0, -1):
    #     heatmap = get_heat_map(value, (6, 5), t)
    #     plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    #     plt.show()
