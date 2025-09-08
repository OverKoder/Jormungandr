import numpy as np  

from .environments import SnakeGame

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCriticNN():
    def __init__(self, actor_hidden_layers: list, critic_hidden_layers: list) -> None:
        super().__init__()

        # 11 is the input size
        prev_features = 11
        actor_layers = []
        for hidden in actor_hidden_layers:
            actor_layers.append(nn.Linear(in_features = prev_features, out_features = hidden))
            actor_layers.append(nn.ReLU())
            prev_features = hidden

        # Hard coded 3 because we only have 3 actions, no need to parametrize
        actor_layers.append(nn.Linear(in_features = prev_features, out_features = 3))
        actor_layers.append(nn.Softmax())

        # Actor NN
        self.actor = nn.Sequential(*actor_layers)

        prev_features = 11
        critic_layers = []
        for hidden in critic_hidden_layers:
            critic_layers.append(nn.Linear(in_features = prev_features, out_features = hidden))
            critic_layers.append(nn.ReLU())
            prev_features = hidden

        critic_layers.append(nn.Linear(in_features = prev_features, out_features = 1))

        # Critic NN
        self.critic = nn.Sequential(*critic_layers)

        return

    def forward(self, state):

        # Initialize gradient on the state
        state = torch.from_numpy(state).float().requires_grad_(True)

        # Value function
        value = self.critic(state)

        # Action probabilities
        action_probs = self.actor(state)
        
        return value, action_probs
    
class ActorCritic():
    def __init__(self, environment: SnakeGame, actor_hidden_layers: list = [100,100,100], critic_hidden_layers: list = [100,100,100]) -> None:

        self.environment = environment

        # Neural Network and optimizer
        self.actor_critic = ActorCriticNN(actor_hidden_layers = actor_hidden_layers, critic_hidden_layers = critic_hidden_layers)
        self.ac_optimizer = optim.AdamW(self.actor_critic.parameters(), lr=1e-3)

        # Reward sum for plotting
        self.reward_sum = 0
        return
    
    def run_episode(self):
        log_probs = []
        values = []
        rewards = []
        
        terminal = False

        self.environment.reset()
        self.reward_sum = 0

        while not terminal:

            # Get value function and action probs
            value, action_probs = self.actor_critic.forward(np.array(list(self.environment.state.values())))
            value = value.detach().numpy()[0]
            probs = action_probs.detach().numpy() 

            # Select action (not greedy to maintain exploration)
            action = np.random.choice(3, p=np.squeeze(probs))

            # Get log probabilities for loss function
            log_prob = torch.log(action_probs.squeeze(0)[action])
            entropy = -np.sum(np.mean(probs) * np.log(probs))

            # Get reward and next state
            terminal, reward, next_state = self.environment.step(action)

            rewards.append(reward)
            self.reward_sum += reward

            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy

            # Next state
            self.environment.state = next_state

            if terminal:
                Qval, _ = self.actor_critic.forward(np.array(list(next_state.values())))
                Qval = Qval.detach().numpy()[0]

        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + Qval
            Qvals[t] = Qval

        #update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)
        
        # Advantage and loss
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        # Backward and optimize
        self.ac_optimizer.zero_grad()
        ac_loss.backward()
        self.ac_optimizer.step()

        return