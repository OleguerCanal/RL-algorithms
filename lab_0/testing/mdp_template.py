import numpy as np

class MDP:
    def __init__(self):
        self.goal = None
        pass

    # Model definition:
    def get_states(self):
        pass

    def get_transition_prob(self, state_prima, state, action):
        pass

    def get_reward(self, state, action):
        pass

    def get_actions(self):
        pass

    # Solvers
    def value_iteration(self, T = -1, discount_factor = 0.8):
        #TODO(oleguer):
        pass

    def policy_iteration(self, T = -1, discount_factor = 0.8):
        actions = self.get_actions()

        discount = 0.5
        Us = [reward_matrix]
        while True:
            new_u = np.zeros(map.shape)
            for state_t_1 in get_states():
                if state_t_1 == self.goal:
                    continue
                max_found = float('-inf')
                for action in actions:
                    val = get_reward(state_t_1, action) #TODO(oleguer): Review this
                    for j in get_states():
                        val += discount*self.get_transition_prob(j, state_t_1, action)*Us[-1][j]
                    max_found = max(max_found, val)
                new_u[state_t_1] = max_found
                new_u[goal] = 0
            if np.array_equal(new_u, Us[-1]):
                break
            Us.append(new_u)
