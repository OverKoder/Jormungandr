import numpy as np

from .action_selection import EpsilonGreedy

class Jormungandr():
    """
    Implements the agent of our game, the snake Jormungandr.
    """

    def __init__(self, alpha, action_selector: EpsilonGreedy, environment, checkpoint: str):

        # Q function
        if checkpoint is not None:
            self.Q = np.load(checkpoint)
        else:
            self.Q = np.zeros((2,2,2,2,2,2,2,2,2,2,2,3))

        # Action selector
        self.action_selector = action_selector

        # Alpha or step size
        self.alpha = alpha

        # Environment the agent interacts with
        self.environment = environment

    def get_Q(self, state):
        """
        Returns Q(S)
        """

        # Get the state
        state = list(state.values())

        return self.Q[*state]
    
    def get_V(self, state):
        """
        Returns V(S)
        """
        return self.V[*state]
    
    def select_action(self, state):
        """
        Selects action, according to an epsilon-greedy policy.
        """

        return self.action_selector.get_action(self.get_Q(state))

    
    def update_Sarsa(self, state, action, reward, next_state, next_action, terminal):
        """
        Updates Q according to Sarsa update rule
        Q(S,A) = Q(S,A) + alpha * [R + Q(S',A') - Q(S,A)]
        """
        
        # If we are on a terminal state, we don't have S' and A', therefore we update only with the current state
        if terminal:
            self.Q[*state.values()][action] = self.Q[*state.values()][action] + self.alpha * (reward - self.Q[*state.values()][action])
        
        else:
            self.Q[*state.values()][action] = self.Q[*state.values()][action] + self.alpha * (reward + self.Q[*next_state.values()][next_action] - self.Q[*state.values()][action])

        return 

    def update_QLearning(self, state, action, reward, next_state, terminal):
        """
        Updates Q according to Q-Learning update rule
        Q(S,A) = Q(S,A) + alpha * [R + max_a Q(S',a) - Q(S,A)]
        """

        if terminal:
            self.Q[*state.values()][action] = self.Q[*state.values()][action] + self.alpha * (reward - self.Q[*state.values()][action])

        else:
            self.Q[*state.values()][action] = self.Q[*state.values()][action] + self.alpha * (reward + self.Q[*next_state.values()][self.action_selector.get_greedy_action(self.get_Q(next_state))] - self.Q[*state.values()][action])
        return 
    
    def update_n_step_sarsa(self, state, action, gain):
        """
        Updates Q according to n_step_sarsa update rule
        Q(S,A) = Q(S,A) + alpha * [G - Q(S,A)]
        """
        self.Q[*state.values()][action] = self.Q[*state.values()][action] + self.alpha * (gain - self.Q[*state.values()][action])

        return 
    
    def update_n_step_offpolicy(self, state, action, sampling_ratio, gain):
        """
        Updates Q according to n_step_offpolicy update rule
        Q(S,A) = Q(S,A) + alpha * sampling_ratio * [G - Q(S,A)]
        """
        self.Q[*state.values()][action] = self.Q[*state.values()][action] + self.alpha * sampling_ratio * (gain - self.Q[*state.values()][action])

        return 