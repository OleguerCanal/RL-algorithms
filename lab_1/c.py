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

def value_iteration_infinite(value, lambd):
    max_iters = 10000
    tol = 1e-10
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


if __name__ == "__main__":

    lambdas = np.arange(1, 30, 1)/ 30
    wins = []
    for lambd in lambdas:
        deadlines = [100000000]
        n_games = 1000
        n_wins = test_geometric(deadlines, n_games, lambd)[0]
        print(""+str(n_wins)+" for lambda = "+str(lambd))
        wins.append(n_wins)


    plt.plot(lambdas, wins, '.b-', label='Percentage of wins over '+str(n_games)+' games')
    plt.ylim(0, 1)
    plt.legend(prop={'size': 15})
    plt.title('Number of wins for decreasing values of lambda')
    plt.show()

