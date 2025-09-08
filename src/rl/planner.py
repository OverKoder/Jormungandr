from random import sample
from queue import PriorityQueue

from .agent import Jormungandr
from .environments import SnakeGame

class DynaQ():

    def __init__(self, agent: Jormungandr, environment: SnakeGame, planning_steps: int, learning_algorithm: str) -> None:
        
        # Agent
        self.agent = agent

		# Environment to plan
        self.environment = environment
          
        # Model of the environment
        self.model = []

        # Planning steps
        self.planning_steps = planning_steps

        # Type of update to do on Q function
        self.learning_algorithm = learning_algorithm

        return

    def reset_model(self):

        self.model = []
        return
    
    def add_step(self, features):
        """
        Adds to the model a step
        Model(S,A) <- R, S'
        """
        self.model.append(features)

        return
    
    def plan(self):
        """
        Plans ahead.
        """

        # Check that we don't sample more than what we have
        k = self.planning_steps if self.planning_steps <= len(self.model) else len(self.model)
        samples = sample(self.model, k = k)

        if self.learning_algorithm == 'sarsa':
            for state, action, reward, next_state, next_action in samples:
                self.agent.update_Sarsa(state, action, reward, next_state, next_action, terminal=False)

        elif self.learning_algorithm == 'qlearning':
            for state, action, reward, next_state in samples:
                self.agent.update_QLearning(state, action, reward, next_state, terminal=False)

        elif self.learning_algorithm == 'nStepSarsa':
            for state, action, gain in samples:
                self.agent.update_n_step_sarsa(state, action, gain)

        elif self.learning_algorithm == 'nStepOffPolicy':
            for state, action, sampling_ratio, gain in samples:
                self.agent.update_n_step_offpolicy(state, action, sampling_ratio, gain)

        return

class DynaQWithPriority():

    def __init__(self, agent, environment, planning_steps: int, learning_algorithm: str, threshold: int = 5) -> None:
        
        # Agent
        self.agent = agent

        # Environment to plan
        self.environment = environment
          
        # Model of the environment
        self.model = []

        # Planning steps
        self.planning_steps = planning_steps

        # PriorityQueue
        self.prioqueue = PriorityQueue()

        # Type of update to do on Q function
        self.learning_algorithm = learning_algorithm

        # Threshold for priority sweep
        self.threshold = threshold

        # Transitions between states
        self.transitions = {}
        return
    
    def reset_model(self):
        """
        Resets the model of the planner
        """
        self.transitions = {}
        self.model = []
        return
    
    def add_step(self, features):
        """
        Adds to the model a step
        Model(S,A) <- R, S'
        """

        if self.learning_algorithm == 'sarsa':
            state, action, reward, next_state, next_action = features
            priority = -1 * abs(reward + self.agent.Q[*next_state.values()][next_action] - self.agent.Q[*state.values()][action])
            try:
                self.transitions[*next_state.values()].append([state, action, reward, next_action])
            except:
                self.transitions[*next_state.values()] = [[state, action, reward, next_action]]

        elif self.learning_algorithm == 'qlearning':
            state, action, reward, next_state = features
            priority = -1 * abs(reward + self.agent.Q[*next_state.values()][self.agent.action_selector.get_greedy_action(self.agent.get_Q(next_state))] - self.agent.Q[*state.values()][action])
            try:
                self.transitions[*next_state.values()].append([state, action, reward])
            except:
                self.transitions[*next_state.values()] = [[state, action, reward]]
        
        # Put it in priority queue
        if abs(priority) > self.threshold:
            self.prioqueue.put((priority, features))

        return
    
    def plan(self):

        steps = 0
        while not self.prioqueue.empty() and steps != self.planning_steps:
            _, features = self.prioqueue.get()

            if self.learning_algorithm == 'sarsa':
                state, action, reward, next_state, next_action = features
                self.agent.update_Sarsa(state, action, reward, next_state, next_action, terminal=False)

                # For all states that lead to S
                try:
                    transitions = self.transitions[*state.values()]

                    # Here we take the previous states that lead to the current state
                    for prev_state, prev_action, curr_reward, curr_state, curr_action in transitions:
                        priority = -1 * abs(curr_reward + self.agent.Q[*curr_state.values()][curr_action] - self.agent.Q[*prev_state.values()][prev_action])

                        if abs(priority) > self.threshold:
                            self.prioqueue.put((priority, features))
                except:
                    pass
            
            elif self.learning_algorithm == 'qlearning':
                state, action, reward, next_state = features
                self.agent.update_QLearning(state, action, reward, next_state, terminal=False)

                # For all states that lead to S
                try:
                    transitions = self.transitions[*state.values()]

                    # Here we take the previous states that lead to the current state
                    for prev_state, prev_action, curr_reward, curr_state in transitions:
                        priority = -1 * abs(curr_reward + self.agent.Q[*curr_state.values()][self.agent.action_selector.get_greedy_action(self.agent.get_Q(curr_state))] - self.agent.Q[*prev_state.values()][prev_action])

                        if abs(priority) > self.threshold:
                            self.prioqueue.put((priority, features))
                except:
                    pass

            steps += 1
        
        return