from .agent import Jormungandr
from .environments import SnakeGame
from .planner import DynaQ, DynaQWithPriority
import numpy as np
import time

from random import sample

class SARSA():
	"""
	Trains the agent using on-policy SARSA to estimate Q
	"""

	def __init__(self, agent: Jormungandr, environment: SnakeGame, planner: DynaQ, save_path: str, test: bool):

		# Agent
		self.agent = agent

		# Environment to solve
		self.environment = environment

		# Path to save Q
		self.save_path = save_path

		# Whether to train or test
		self.test = test

		# Planner
		self.planner = planner

		return

	def run_episode(self):

		# Reset environment
		self.environment.reset()

		# Reset planner
		if self.planner is not None:
			self.planner.reset_model()

		# Get first action
		action = self.agent.select_action(self.environment.state)

		terminal = False
		self.reward_sum = 0

		while not terminal:

			# Act on the enviroment, observe R and S'
			terminal, reward, next_state = self.environment.step(action)
			self.reward_sum += reward

			# Choose next action
			next_action = self.agent.select_action(next_state)

			# Update Q
			if not self.test:
				self.agent.update_Sarsa(self.environment.state, action, reward, next_state, next_action, terminal)

			# Add to planner
			if self.planner is not None:
				self.planner.add_step([self.environment.state, action, reward, next_state, next_action])

			# Update S and A
			self.environment.state = next_state
			action = next_action

		if not self.test:
			np.save(self.save_path, self.agent.Q)

		# Plan ahead
		if self.planner is not None:
			self.planner.plan()

		return

class QLearning():
	"""
	Implements off-policy Q-Learning to estimate Q
	"""

	def __init__(self, agent: Jormungandr, environment: SnakeGame, planner: DynaQ, save_path: str, test: bool):

		# Agent
		self.agent = agent

		# Environment to solve
		self.environment = environment

		# Path to save Q
		self.save_path = save_path

		# Whether to test or train
		self.test = test

		# Planner
		self.planner = planner
		return

	def run_episode(self):

		# Initialize the state
		self.environment.reset()

		# Reset planner
		if self.planner is not None:
			self.planner.reset_model()

		terminal = False
		self.reward_sum = 0
		while not terminal:
			
			# Get action
			action = self.agent.select_action(self.environment.state)

			# Act on the enviroment, observe R and S'
			terminal, reward, next_state = self.environment.step(action)
			self.reward_sum += reward
			if not self.test:
				# Update Q
				self.agent.update_QLearning(self.environment.state, action, reward, next_state, terminal)

			# Add to planner
			if self.planner is not None:
				self.planner.add_step([self.environment.state, action, reward, next_state])

			# Update S 
			self.environment.state = next_state
		
		if not self.test:
			np.save(self.save_path, self.agent.Q)

		# Plan ahead
		if self.planner is not None:
			self.planner.plan()

		return

class n_step_SARSA():
	"""
	Implements on-policy n-step SARSA to estimate Q
	"""
	def __init__(self, n_steps: int, agent: Jormungandr, environment, planner: DynaQ, save_path: str, test: bool) -> None:
		
		# Number of steps
		self.n_steps = n_steps

		# Agent
		self.agent = agent

		# Environment to solve
		self.environment = environment

		# Path to save V
		self.save_path = save_path

		# Whether to test or train
		self.test = test

		# Planner (DynaQWithPriority not currently implemented for nStep)
		self.planner = planner if isinstance(planner, DynaQ) else None

	def run_episode(self):
		
		T = 2**31
		tau, t = 0, 0

		state_list, reward_list, action_list = [None] * (self.n_steps), [0] * (self.n_steps + 1), [0] * (self.n_steps)

		# Initialize state
		self.environment.reset()

		# Reset planner
		if self.planner is not None:
			self.planner.reset_model()

		# Select first action
		action = self.agent.select_action(self.environment.state)
		
		# Get starting state and action
		state_list.insert(0, self.environment.state)
		action_list.insert(0, action)
		self.reward_sum = 0

		while tau != T - 1:
			
			if t < T:

				# Take action and observe reward and next state
				terminal, reward, next_state = self.environment.step(action)
				self.reward_sum += reward

				# Store reward and next state as t + 1
				reward_list[(t + 1) % (self.n_steps + 1)] = reward
				state_list[(t + 1) % (self.n_steps + 1)] = next_state

				# If we reached the goal
				if terminal:
					T = t + 1

				else:

					# Update state
					self.environment.state = next_state

					# Select next action and store it
					next_action = self.agent.select_action(self.environment.state)
					action_list[(t + 1) % (self.n_steps + 1)] = next_action

					# Update action
					action = next_action

			# tau is the time whose state's estimate is being updated
			tau = t - self.n_steps + 1

			if tau >= 0:
				
				state_tau = state_list[tau % (self.n_steps + 1)]
				action_tau = action_list[tau % (self.n_steps + 1)]

				# Expected gain
				gain = 0
				for i in range(tau + 1, min(tau + self.n_steps, T) + 1):
					gain += reward_list[i % (self.n_steps + 1)]

				# If we didn't reach the end of the episode, replace the remaining
				# missing rewards with the estimate in Q
				if tau + self.n_steps < T:

					tmp_state = state_list[(tau + self.n_steps) % (self.n_steps + 1)]
					tmp_action = action_list[(tau + self.n_steps) % (self.n_steps + 1)]
					gain = gain + self.agent.get_Q(tmp_state)[tmp_action]

				self.agent.update_n_step_sarsa(state_tau, action_tau, gain)		

				# Add to planner
				if self.planner is not None:
					self.planner.add_step([state_tau, action_tau, gain])

			t += 1
		
		if not self.test:
			np.save(self.save_path, self.agent.Q)

		# Plan ahead
		if self.planner is not None:
			self.planner.plan()
		return

class n_step_OffPolicy():
	"""
	Implements off-policy n-step SARSA to estimate Q
	"""
	def __init__(self, n_steps: int, agent: Jormungandr, environment, planner: DynaQ, save_path: str, test: bool) -> None:
		
		# Number of steps
		self.n_steps = n_steps

		# Agent
		self.agent = agent

		# Environment to solve
		self.environment = environment

		# Path to save V
		self.save_path = save_path

		# Whether to test or train
		self.test = test

		# Planner (DynaQWithPriority not currently implemented for nStep)
		self.planner = planner if isinstance(planner, DynaQ) else None

	def run_episode(self):
		
		T = 2**31
		tau, t = 0, 0

		state_list, reward_list, action_list = [None] * (self.n_steps), [0] * (self.n_steps + 1), [0] * (self.n_steps)

		# Initialize state
		self.environment.reset()

		# Reset planner
		if self.planner is not None:
			self.planner.reset_model()

		# Select first action
		action = self.agent.select_action(self.environment.state)
		
		# Get starting state and action
		state_list.insert(0, self.environment.state)
		action_list.insert(0, action)
		self.reward_sum = 0

		while tau != T - 1:
			
			if t < T:

				# Take action and observe reward and next state
				terminal, reward, next_state = self.environment.step(action)
				self.reward_sum += reward

				# Store reward and next state as t + 1
				reward_list[(t + 1) % (self.n_steps + 1)] = reward
				state_list[(t + 1) % (self.n_steps + 1)] = next_state

				# If we reached the goal
				if terminal:
					T = t + 1

				else:
					# Update state
					self.environment.state = next_state

					# Select next action and store it
					next_action = self.agent.select_action(self.environment.state)
					action_list[(t + 1) % (self.n_steps + 1)] = next_action

					# Update action
					action = next_action

			# tau is the time whose state's estimate is being updated
			tau = t - self.n_steps + 1

			if tau >= 0:
				
				state_tau = state_list[tau % (self.n_steps + 1)]
				action_tau = action_list[tau % (self.n_steps + 1)]

				# Compute sampling ratio
				sampling_ratio = 1
				for i in range(tau + 1, min(tau + self.n_steps, T - 1) + 1):
					
					tmp_state = state_list[i % (self.n_steps + 1)]
					tmp_action = action_list[i % (self.n_steps + 1)]

					# Pi(Ai, Si) / b(Ai, Si)
					sampling_ratio *= self.agent.action_selector.get_greedy_probs(self.agent.get_Q(tmp_state)) [tmp_action] / self.agent.action_selector.get_action_probs(self.agent.get_Q(tmp_state)) [tmp_action] 

				# Expected gain
				gain = 0
				for i in range(tau + 1, min(tau + self.n_steps, T) + 1):
					gain += reward_list[i % (self.n_steps + 1)]

				# If we didn't reach the end of the episode, replace the remaining
				# missing rewards with the estimate in Q
				if tau + self.n_steps < T:

					tmp_state = state_list[(tau + self.n_steps) % (self.n_steps + 1)]
					tmp_action = action_list[(tau + self.n_steps) % (self.n_steps + 1)]
					gain = gain + self.agent.get_Q(tmp_state)[tmp_action]

				self.agent.update_n_step_offpolicy(state_tau, action_tau, sampling_ratio, gain)		

				# Add to planner
				if self.planner is not None:
					self.planner.add_step([state_tau, action_tau, sampling_ratio, gain])

			t += 1
		
		if not self.test:
			np.save(self.save_path, self.agent.Q)
			
		# Plan ahead
		if self.planner is not None:
			self.planner.plan()
		return