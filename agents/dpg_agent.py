import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_out = nn.Linear(hidden_dim, action_dim)
        self.action_limit = action_limit

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.action_out(x)) * self.action_limit
        return action

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_value(x)
        return q_value

class ReplayBuffer:
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.device = device

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.buffer[idx] for idx in batch])

        state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
        action = torch.tensor(np.array(action), dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(self.device)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DPGAgent:
    def __init__(self, env, device='cpu', gamma=0.99, tau=0.005,
                 actor_lr=1e-4, critic_lr=1e-3, buffer_capacity=1000000, batch_size=256):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.action_limit = env.action_space.high[0]

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, self.action_limit).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim, self.action_limit).to(self.device)
        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, device=self.device)

        # Noise for exploration
        self.noise_std = 0.1

    def select_action(self, state, noise=True):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().detach().numpy()[0]
        if noise:
            action += np.random.normal(0, self.noise_std, size=action.shape)
        return np.clip(action, -self.action_limit, self.action_limit)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        # Critic update
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_value = reward + (1 - done) * self.gamma * target_q

        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft updates
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, total_steps):
        state = self.env.reset()
        total_reward = 0
        step = 0

        while step < total_steps:
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            if done:
                state = self.env.reset()
                total_reward = 0

            self.update()
            step += 1

        print(f"Training completed for {total_steps} steps.")

    def save_model(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
