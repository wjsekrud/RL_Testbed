import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class SoftQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_value(x)
        return q_value

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256, action_space=None):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.action_space = action_space

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)

        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_space.high[0]

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

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

class SACAgent:
    def __init__(self, env, device='cpu', gamma=0.99, tau=0.005, alpha=0.2,
                 policy_lr=3e-4, q_lr=3e-4, buffer_capacity=1000000, batch_size=256):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.batch_size = batch_size

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_space = env.action_space

        # Networks
        self.policy_net = PolicyNetwork(obs_dim, action_dim, action_space=action_space).to(self.device)
        self.q_net1 = SoftQNetwork(obs_dim, action_dim).to(self.device)
        self.q_net2 = SoftQNetwork(obs_dim, action_dim).to(self.device)
        self.target_q_net1 = SoftQNetwork(obs_dim, action_dim).to(self.device)
        self.target_q_net2 = SoftQNetwork(obs_dim, action_dim).to(self.device)

        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, device=self.device)

    def select_action(self, state, evaluate=False):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        if evaluate:
            with torch.no_grad():
                mean, _ = self.policy_net.forward(state)
                action = torch.tanh(mean) * self.env.action_space.high[0]
                action = action.cpu().numpy()[0]
        else:
            action, _ = self.policy_net.sample(state)
            action = action.cpu().detach().numpy()[0]
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_action, next_log_prob = self.policy_net.sample(next_state)
            target_q1 = self.target_q_net1(next_state, next_action)
            target_q2 = self.target_q_net2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = reward + (1 - done) * self.gamma * target_q

        # Update Q-functions
        current_q1 = self.q_net1(state, action)
        current_q2 = self.q_net2(state, action)
        q1_loss = F.mse_loss(current_q1, target_value)
        q2_loss = F.mse_loss(current_q2, target_value)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update Policy network
        new_action, log_prob = self.policy_net.sample(state)
        q1_new = self.q_net1(state, new_action)
        q2_new = self.q_net2(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_prob - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft Updates
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, total_steps):
        state, _ = self.env.reset()
        total_reward = 0
        step = 0

        while step < total_steps:
            action = self.select_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            total_reward += reward

            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            if done or truncated:
                state, _ = self.env.reset()
                total_reward = 0

            self.update()
            step += 1

        print(f"Training completed for {total_steps} steps.")

    def save_model(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'q_net1': self.q_net1.state_dict(),
            'q_net2': self.q_net2.state_dict(),
            'target_q_net1': self.target_q_net1.state_dict(),
            'target_q_net2': self.target_q_net2.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.q_net1.load_state_dict(checkpoint['q_net1'])
        self.q_net2.load_state_dict(checkpoint['q_net2'])
        self.target_q_net1.load_state_dict(checkpoint['target_q_net1'])
        self.target_q_net2.load_state_dict(checkpoint['target_q_net2'])
