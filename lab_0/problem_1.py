import matplotlib.pyplot as plt
import numpy as np
import copy

map = np.zeros((6, 7))
reward_matrix = np.zeros((6, 7))

def get_model():
    map = np.ones((6, 7))
    map[0:3, 2] = 0
    map[4, 1:6] = 0
    start = (0, 0)
    goal = (5, 5)

    reward_matrix = -np.ones((6, 7))
    # reward_matrix[start] = 0
    reward_matrix[goal] = 0
    return map, reward_matrix, start, goal

def trans(state_prima, state, a):
    if state_prima != tuple(np.array(state) + np.array(a)):
        return 0
    if state_prima[0] < 0 or state_prima[0] >= map.shape[0] or\
        state_prima[1] < 0 or state_prima[1] >= map.shape[1]:
        return 0
    else:
        return map[state_prima]

def get_reward(state, action):
    new_state = tuple(np.array(state) + np.array(action))
    if new_state[0] < 0 or new_state[0] >= map.shape[0] or\
        new_state[1] < 0 or new_state[1] >= map.shape[1]:
        return float('-inf')
    return reward_matrix[new_state]

def get_states():
    for state, _ in np.ndenumerate(map):
        if map[tuple(state)] == 1:
            yield state

if __name__ == "__main__":
    map, reward_matrix, start, goal = get_model()
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    T = 10
    Us = [reward_matrix]
    state = goal
    for t in reversed(range(2, T)):
        new_u = np.zeros(map.shape)
        for state_t_1 in get_states():
            max_found = float('-inf')
            for action in actions:
                val = get_reward(state_t_1, action) #TODO(oleguer): Review this
                for j in get_states():
                    print(j)
                    val += trans(j, state_t_1, action)*Us[-1][j]
                max_found = max(max_found, val)
            new_u[state_t_1] = max_found
        Us.append(new_u)

    for U in Us:
        plt.imshow(U, cmap='hot', interpolation='nearest')
        plt.show()