import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class A3CNetwork(nn.Module):
    def __init__(self, input_dim, action_space):
        super(A3CNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.policy = nn.Linear(128, action_space)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value

class A3CAgent:
    def __init__(self, env, global_model, optimizer, gamma=0.99):
        self.env = env
        self.global_model = global_model
        self.local_model = A3CNetwork(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optimizer
        self.gamma = gamma

    def sync_with_global(self):
        self.local_model.load_state_dict(self.global_model.state_dict())

    def train(self, max_steps):
        self.sync_with_global()
        state = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
        done = False
        while not done:
            logits, value = self.local_model(state)
            action_prob = F.softmax(logits, dim=-1)
            action = torch.multinomial(action_prob, 1).item()
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            _, next_value = self.local_model(next_state)

            # Compute advantage and loss
            advantage = reward + self.gamma * next_value * (1 - int(done)) - value
            value_loss = advantage.pow(2)
            policy_loss = -torch.log(action_prob[0, action]) * advantage.detach()

            # Total loss
            loss = (value_loss + policy_loss).mean()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
                global_param._grad = local_param.grad
            self.optimizer.step()

            state = next_state
            if done or total_reward >= max_steps:
                break
        return total_reward
