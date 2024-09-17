import random
import numpy as np

from utils import softmax

REDUCE_STEPS = 1000
class EpsilonGreedy():

    def __init__(self, epsilon: float, reduce_on_steps = False) -> None:
        
        self.epsilon = epsilon
        self.steps = 0
        self.reduce_on_steps = reduce_on_steps

    def get_action(self, action_estimates):
        
        # Get random probability
        prob = random.random()
        self.steps += 1

        if self.reduce_on_steps and self.steps == REDUCE_STEPS:
            self.steps = 0

            if self.epsilon > 0:
                self.epsilon = round(self.epsilon -0.1, 1)
                print("Epsilon:",self.epsilon)
        
        # Explore
        if prob < self.epsilon:
            return np.random.randint(0, len(action_estimates))
        
        # Exploit
        else:
            return np.argmax(action_estimates)
    
    def get_action_probs(self, action_estimates):
        """
        Returns the probability of each action. Taking into account the epsilon-greedy policy
        """
        # Get random probability
        prob = random.random()

        # Explore
        if prob < self.epsilon:
            return np.repeat(1 / len(action_estimates), len(action_estimates))
        
        # Exploit
        else:
            return softmax(action_estimates) 
    
    def get_greedy_probs(self, action_estimates):
        """
        Returns the probality of the greedy policy (1 for the action that maximizes the expected reward, 0 for the rest)
        """

        # Initialize
        greedy_probs = np.zeros(len(action_estimates))

        # Get greedy action
        greedy_action = self.get_greedy_action(action_estimates)

        # Set probability of the greedy action to one
        greedy_probs[greedy_action] = 1.0

        return greedy_probs
    
    def get_greedy_action(self, action_estimates):
        """
        Returns the greedy action. This is used to compute: max_a Q(S',a)
        """

        return np.argmax(action_estimates)
    
class EpsilonGreedyWeights():

    def __init__(self, epsilon: float, reduce: int) -> None:
        
        self.epsilon = epsilon
        self.reduce = reduce
        self.steps = 1
    

    def get_action(self, features, weights):
        
        # Get random probability
        prob = random.random()

        if self.reduce is not None:
            self.steps += 1
            if self.steps == self.reduce:
                self.steps = 1

                if self.epsilon > 0.1:
                    self.epsilon = round(self.epsilon -0.1, 1)

                print("Epsilon: ",self.epsilon)

        # Explore
        if prob < self.epsilon:
            return np.random.randint(0, weights.shape[1])
        
        # Exploit
        else:

            # Shape 1xf * fx3 = 1x3
            return np.argmax(np.dot(features, weights))
        
        