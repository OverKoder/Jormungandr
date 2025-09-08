from collections import namedtuple, deque
import random
import numpy as np

from .environments import SnakeGame

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

# DQN uses a replay memory, defined like this
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Returns a batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

  def __init__(self, hidden_layers: list = [100,100,100]):
        super().__init__()

        # 11 is the input size
        prev_features = 11
        layers = []
        for hidden in hidden_layers:
            layers.append(nn.Linear(in_features = prev_features, out_features = hidden))
            layers.append(nn.ReLU())
            prev_features = hidden

        # Hard coded 3 because we only have 3 actions, no need to parametrize
        layers.append(nn.Linear(in_features = prev_features, out_features = 3))

        # DQN
        self.layers = nn.Sequential(*layers)

        return

  def forward(self, state):

        # Initialize gradient on the state
        state = torch.from_numpy(state).float().requires_grad_(True)

        return self.layers(state)   

class DQNSolver():

    def __init__(self, environment, batch_size: int,  hidden_layers: list = [100,100,100], tau = 0.001):
        
        # Policy and Target networks
        self.policy_net = DQN(hidden_layers)
        self.target_net = DQN(hidden_layers)

        # Copy policy target's weights into target net
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Replay memory
        self.replay_memory = ReplayMemory(batch_size)

        # Optimizer
        self.optimizer = optim.AdamW(params = self.policy_net.parameters(), lr = 1e-2)

        # Environment
        self.environment = environment

        # Others
        self.batch_size = batch_size
        self.tau = tau
        return
  
    def optimize_model(self):
        if len(self.replay_memory) < self.batch_size:
            return
        transitions = self.replay_memory.sample(self.batch_size)

        #This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.tautarget_net(non_final_next_states).max(1).values

        # Compute the expected Q values (no gamma)
        expected_state_action_values = next_state_values+ reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping (avoid huge gradients)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return

    def run_episode(self):
        
        # Initialize
        terminal = False

        self.environment.reset()
        self.reward_sum = 0

        while not terminal:

            # Get action
            action = self.policy_net(np.array(list(self.environment.state.values())))

            # Get reward and next state
            terminal, reward, next_state = self.environment.step(action)
            self.reward_sum += reward

            if terminal:
                # Save into replay memory (but next state is none)
                self.replay_memory.push((self.environment.state.values(), action, reward, None))
            else:
                # Save into replay memory
                self.replay_memory.push((self.environment.state.values(), action, reward, next_state.values()))
            
            self.environment.state = next_state
            
            # Optimize policy net
            self.optimize_model()

            # Update target net
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
            self.target_net.load_state_dict(target_net_state_dict)
        return