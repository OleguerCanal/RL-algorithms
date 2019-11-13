from abc import ABC, abstractmethod
import numpy as np
import copy
 
class MDP(ABC):
    def __init__(self):
        super().__init__()
    
    # Model methods:
    @abstractmethod
    def get_states(self): # States
        # TODO: Maybe this should be a variable states instead of a method
        # @property
        pass

    @abstractmethod
    def get_actions(self, state = None): # Actions (might depend on state)
        # TODO: Maybe this should be a variable states instead of a method
        pass

    @abstractmethod
    def get_transition_prob(self, next_state, state, action): # Transitions
        pass

    @abstractmethod
    def get_reward(self, state, action): # Rewards
        pass


    def value_iteration(self, initial_values, T = -1, discount = 0.8):
        ''' Performs value iteration algorithm to defined MDP
            returns list of state values after each iteration
        '''
        actions = self.get_actions()
        states = self.get_states()

        Us = [initial_values]
        while True:
            new_u = copy.deepcopy(Us[-1])
            for state in states:
                max_found = float('-inf')
                for action in actions:
                    val = self.get_reward(state, action)
                    for j in states:
                        val += discount*\
                            self.get_transition_prob(j, state, action)*Us[-1][j]
                    max_found = max(max_found, val)
                new_u[state] = max_found
            if np.array_equal(new_u, Us[-1]):
                break
            Us.append(new_u)
        return Us

    def get_policy(self, value_mapping, discount = 0.8):
        #TODO: return new best policy given value_mapping
        # obs: value mapping can either be: 
        # np array of values with indexes as states
        # map to state value  (more genereal)
        pass

    def policy_evaluation(self, policy, discount = 0.8):
        #TODO: Implement policy evaluation algorithm (return value mapping)
        pass

    def policy_iteration(self):
        #TODO: Implement policy iteration using policy_evaluation and policy_imporvement
        # Return policy and 
        pass