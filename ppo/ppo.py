import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probabilities = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), np.array(self.probabilities), np.array(
            self.values), np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, probabilities, values, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probabilities.append(probabilities)
        self.values.append(values)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probabilities = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dimensions, alpha, fc1_dimensions=256, fc2_dimensions=256,
                 cpname="ppo/actor_network.h5"):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = cpname
        self.actor = nn.Sequential(
            nn.Linear(input_dimensions, fc1_dimensions),
            nn.ReLU(),
            nn.Linear(fc1_dimensions, fc2_dimensions),
            nn.ReLU(),
            nn.Linear(fc2_dimensions, n_actions),
            nn.Softmax(dim=1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dimensions, alpha, fc1_dimensions=256, fc2_dimensions=256,
                 cpname="ppo/critic_network.h5"):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = cpname
        self.critic = nn.Sequential(
            nn.Linear(input_dimensions, fc1_dimensions),
            nn.ReLU(),
            nn.Linear(fc1_dimensions, fc2_dimensions),
            nn.ReLU(),
            nn.Linear(fc2_dimensions, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class PPOAgent:
    def __init__(self, n_actions, input_dimensions, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, N=2048,
                 n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.actor = ActorNetwork(n_actions, input_dimensions, alpha)
        self.critic = CriticNetwork(input_dimensions, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probabilities, values, reward, done):
        self.memory.store_memory(state['board'], action, probabilities, values, reward, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation['board']], dtype=T.float).to(self.actor.device)
        distribution = self.actor(state)
        value = self.critic(state)
        action = distribution.sample()
        probabilities = T.squeeze(distribution.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        return action, probabilities, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_array, action_array, old_probabilities_array, values_array, reward_array, dones_array, batches = self.memory.generate_batches()
            values = values_array
            advantage = np.zeros(len(reward_array), dtype=np.float32)
            for time_step in range(len(reward_array) - 1):
                discount = 1
                advantage_time_step = 0
                for k in range(time_step, len(reward_array) - 1):
                    advantage_time_step += discount * (
                            reward_array[k] + self.gamma * values[k + 1] * (1 - int(dones_array[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[time_step] = advantage_time_step
            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_array[batch], dtype=T.float).to(self.actor.device)
                old_probabilities = T.tensor(old_probabilities_array[batch]).to(self.actor.device)
                actions = T.tensor(action_array[batch]).to(self.actor.device)
                distribution = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)
                new_probabilities = distribution.log_prob(actions)
                probabilities_ratio = new_probabilities.exp() / old_probabilities.exp()
                weighted_probabilities = advantage[batch] * probabilities_ratio
                weighted_clipped_probabilities = T.clamp(probabilities_ratio, 1 - self.policy_clip,
                                                         1 + self.policy_clip) * advantage[batch]
                actor_loss = T.min(weighted_probabilities, weighted_clipped_probabilities).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.sum().backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            self.memory.clear_memory()
