from abc import ABC, abstractmethod
import numpy as np
import copy
import random
 
class MDP(ABC):
    epsilon = 1e-5

    def __init__(self):
        self.epsilon = 1e-2
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
    def get_transitions(self, state, action): # Transitions
        pass

    @abstractmethod
    def get_reward(self, state, action): # Rewards
        pass

    def __equal_dicts(self, dict1, dict2, tol):
        for key in dict1.keys():
            if dict1[key] - dict2[key] > tol:
                return False
        return True

    def __iteration(self, initial_values, T, gamma):
        states = self.get_states()

        Us = [initial_values]
        tol = self.epsilon*(1 - gamma)/gamma

        while T > 0:
            T -= 1
            new_u = copy.deepcopy(Us[-1])
            for state in states:
                max_found = -100
                possible_actions = self.get_actions(state)
                for action in possible_actions:
                    val = self.get_reward(state, action)
                    transitions = self.get_transitions(state, action)
                    # print("val: " + str(val))
                    # print("transitions: " + str(transitions))
                    # a = input()
                    for j, p in transitions:
                        val += gamma*p*Us[-1][j]
                    max_found = max(max_found, val)
                new_u[state] = max_found
            if self.__equal_dicts(Us[-1], new_u, tol):
                break
            Us.append(new_u)
        return Us


    def value_iteration(self, initial_values, gamma = 0.8):
        ''' Performs value iteration algorithm to defined MDP
            returns list of state values after each iteration
        '''
        Us = self.__iteration(initial_values, np.inf, gamma)
        return Us

    def dynamic_programming(self, initial_values, T):
        ''' Performs value iteration algorithm to defined MDP
            returns list of state values after each iteration
        '''
        Us = self.__iteration(initial_values, T, 1)
        return Us

    def get_policy(self, value_mapping, discount = 0.8):
        #TODO(federico): FIX THIS NOW!!! >:-(
        pass

    def policy_evaluation(self, policy, discount = 1):
        # Return value mapping: map state->value
        states = get_states()
        actions = get_actions()
        T = 20

        # Initialize each state with reward given by action specified by policy
        value_mapping = {state: get_reward(state, policy[state]) for state in states}

        for t in range(T):
            new_value_mapping = {state: get_reward(state, policy[state]) for state in states}
            for state in states:
                new_value_mapping[state] += discount *\
                    sum(get_transition_prob(new_state, state, policy[state]) * value_mapping[new_state] for new_state in states)
            value_mapping = new_value_mapping
        return value_mapping



    def policy_iteration(self):
        # return best policy as a map {state: action} and also {state: value}
        states = get_states()
        actions = get_actions()
        # Initialize random policy
        policy = {state: random.choice(actions) for state in states}

        while True:
            changed = False
            values = policy_evaluation(policy) # values is a map (state, value)
            for state in values:
                best_action, best_action_val = (None, float('-inf'))
                for action in actions:
                    action_value = sum(values[new_state] * p for new_state, p in get_action_result(action))
                    if best_action_val > action_value:
                        best_action = action
                        best_action_val = action_value
                if policy[state] != best_action:
                    policy[state] = best_action
                    changed = True
            if not changed:
                break
        return policy, values
    
    def get_action_result(self, state, action):
        # return list of tuples (new_state, probability) that represent
        # the possible new states with their probability
        # TODO (federico) should this return a map instead? 
        states = get_states()
        return [(new_state, get_transition_prob(new_state, state, action)) for new_state in states]
