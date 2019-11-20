import numpy as np
from matplotlib import pyplot as plt
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
    mino_actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, coord=(0, 0, 6, 5)):
        self.player = (coord[0], coord[1])
        self.mino = (coord[2], coord[3])
    
    def get_coord(self):
        return self.player[0], self.player[1], self.mino[0], self.mino[1]

    def __dead(self):
        return self.player[0] == self.mino[0] and self.player[1] == self.mino[1]
    
    def __free(self):
        return self.player[0] == State.goal[0] and self.player[1] == State.goal[1]

    def get_actions(self):
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
        if self.__dead():
            return -10000
        if self.__free():
            return 1
        return 0

def get_heat_map(values, minotaur_pos, T):
    minotaur_pos = tuple(minotaur_pos)
    heatmap = np.zeros(map.shape)

    for index, val in np.ndenumerate(value[:, :, :, :, T]):
        if index[2] == minotaur_pos[0] and index[3] == minotaur_pos[1]:
            heatmap[index[0], index[1]] = val
    return heatmap


def value_iteration(value, T):
    for T in tqdm(range(T-1, 0, -1)):
        for index, _ in np.ndenumerate(value[:, :, :, :, T]):
            state = State(index)
            max_found = 0
            for action in state.get_actions():
                val = state.reward()
                
                for next_state, p in state.get_transitions(action):
                    xp, yp, xm, ym = next_state.get_coord()
                    val += p*value[xp, yp, xm, ym, T+1]
                max_found = max(max_found, val)

            xp, yp, xm, ym = index
            value[xp, yp, xm, ym, T] = max_found
    return value


if __name__ == "__main__":
    T = 20

    value = np.zeros((map.shape[0], map.shape[1], map.shape[0], map.shape[1], T))
    value[6, 5, :, :, T-1] = 1
    value[6, 5, 6, 5, T-1] = 0
    value = value_iteration(value, T-1)

    mino_pos = (4, 2)
    for t in range(19, 0, -1):
        heatmap = get_heat_map(value, mino_pos, t)
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.show()

