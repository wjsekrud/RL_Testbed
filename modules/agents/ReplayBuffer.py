import torch
import numpy as np

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
