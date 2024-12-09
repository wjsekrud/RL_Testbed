import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import ReplayBuffer
from torch.distributions import Normal

class SoftQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, is2d, hidden_dim=256):
        super(SoftQNetwork, self).__init__()
        print(f"Q, input dim= {input_dim}, action dim = {action_dim}" )

        # for 2d environments 16 - 1 
        self.conv1=nn.Conv2d(input_dim,16,kernel_size=3,stride=3) # [N, 1, 96, 96] -> [N, 16, 32, 32]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=1, stride=1)  # [N, 16, 32, 32] -> [N, 32, 16, 16]
        self.in_features=16*8*8
    
        if is2d:
            print("2d environment")
            self.fc1 = nn.Linear(self.in_features, hidden_dim)
        else:
            self.fc1 = nn.Linear(input_dim + action_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        #print("QForward")

        try:
            x = torch.cat((state, action), dim=-1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            q_value = self.q_value(x)
        except:
            x = F.relu(self.conv1(state))
            x = F.relu(self.conv2(x))
            x = x.view((-1, self.in_features))
            x = self.fc1(x)
            x = self.fc2(x)
            q_value = self.q_value(x)

        return q_value

class SoftVNetwork(nn.Module):
    def __init__(self, input_dim : int, is2d, hidden_dim=256):
        super(SoftVNetwork, self).__init__()

        print("V, input dim= ", input_dim)

        # for 2d environments 16 - 1 
        self.conv1= nn.Conv2d(input_dim,16,kernel_size=3,stride=3) # [N, 1, 96, 96] -> [N, 16, 32, 32]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=1, stride=1)  # [N, 16, 32, 32] -> [N, 32, 16, 16]
        self.in_features=16*8*8
    
        if is2d:
            print("2d environment")
            self.fc1 = nn.Linear(self.in_features, hidden_dim)
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        #print("VForward")
        try:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            value = self.out(x)
        except:
            x = F.relu(self.conv1(state))
            x = F.relu(self.conv2(x))
            x = x.view((-1, self.in_features))
            x = self.fc1(state)
            x = self.fc2(x)
            value = self.out(x)

        return value

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256, action_space=None, is2d=False):
        super(PolicyNetwork, self).__init__() 

        # for 2d environments 16 - 1 
        self.conv1=nn.Conv2d(input_dim,16,kernel_size=3,stride=3) # [N, 1, 96, 96] -> [N, 16, 32, 32]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=1, stride=1)  # [N, 16, 32, 32] -> [N, 32, 16, 16]
        self.in_features=16*8*8
    
        if is2d:
            print("2d environment")
            self.fc1 = nn.Linear(self.in_features, hidden_dim)
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.action_space = action_space

        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        #print("A forward.")
        try:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
        except:
            x = F.relu(self.conv1(state))
            x = F.relu(self.conv2(x))
            x = x.view((-1, self.in_features))
            x = self.fc1(x)
            x = self.fc2(x)

        mu = self.mean(x).tanh()

        log_std = self.log_std(x).tanh()
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        std = torch.exp(log_std)

        dist = Normal(mu,std)
        z = dist.rsample()

        action = z.tanh()
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, mu, log_prob

class SACAgent:
    def __init__(self, app, env, device='cpu', gamma=0.99, tau=0.005,
                 policy_lr=3e-4, q_lr=3e-4, warmup_steps = 5000, buffer_capacity=1000000, batch_size=256):
        self.app = app
        self.env = env
        self.is2d = (self.env.unwrapped.spec.id == 'CarRacing-v2')  
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.warmup_steps = warmup_steps

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_space = env.action_space

        self.target_entropy = -np.prod((action_dim,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=q_lr)

        self.batch_size = batch_size
        self.policy_update_freq = 2
        

        # Networks
        self.actor = PolicyNetwork(obs_dim, action_dim, action_space=action_space, is2d=self.is2d).to(self.device) #Actor

        self.q_net1 = SoftQNetwork(obs_dim, action_dim, is2d=self.is2d).to(self.device) #Q function
        self.q_net2 = SoftQNetwork(obs_dim, action_dim, is2d=self.is2d).to(self.device)

        self.vf = SoftVNetwork(obs_dim, is2d=self.is2d).to(self.device) #V function
        self.vf_target = SoftVNetwork(obs_dim, is2d=self.is2d).to(self.device)
        #self.vf_target.load_state_dict(self.vf.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_lr)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=q_lr)
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, device=self.device)

        self.total_steps = 0

    def select_action(self, state, evaluate=False):
        if evaluate:
            with torch.no_grad():
                x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                action, mu, log_prob = self.actor.forward(x)
                return mu.detach().cpu().numpy()[0]

        else:
            x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            action, mu, log_prob = self.actor(x)
            action = action.detach().cpu().numpy()[0]
            #mu.mu.detach().cpu().numpy()[0]
            #log_prob = log_prob.detach().cpu().numpy()[0]

            return action

            #action, _ = self.policy_net.sample(state)
            #action = action.cpu().detach().numpy()[0]

        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        
        state, action, reward, next_state, done =  map(lambda x: x.to(self.device), self.replay_buffer.sample(self.batch_size))
        new_action, mu, log_prob = self.actor(state)

        alpha_loss = (-self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()

        mask = 1 - done
        # q function loss
        #Qprint("Qloss")
        q_1_pred = self.q_net1(state, action)
        q_2_pred = self.q_net2(state, action)
        v_target = self.vf_target(next_state)
        q_target = reward + self.gamma * v_target * mask
        qf_1_loss = F.mse_loss(q_1_pred, q_target.detach())
        qf_2_loss = F.mse_loss(q_2_pred, q_target.detach())

        # v function loss
        #print("Vloss")
        v_pred = self.vf(state)
        q_pred = torch.min(self.q_net1(state, new_action), self.q_net2(state, new_action))
        v_target = q_pred - alpha * log_prob
        vf_loss = F.mse_loss(v_pred, v_target.detach())

        if self.total_steps % self.policy_update_freq == 0:
            # actor loss
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()

            # Train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # v target update
            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)

        # Train Q-functions
        self.q1_optimizer.zero_grad()
        qf_1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        qf_2_loss.backward()
        self.q2_optimizer.step()

        #qf_loss = qf_1_loss + qf_2_loss

        # train V function
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

    def _target_soft_update(self):
        tau = self.tau
        #print("V soft update")
        for t_param, l_param in zip(self.vf_target.parameters(), self.vf.parameters()):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def train(self, max_steps):
        state, _ = self.env.reset()
        total_reward = 0
        while self.total_steps < max_steps:
            action = self.select_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            total_reward += reward
            done = done or truncated

            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            if done:
                state, _ = self.env.reset()
                total_reward = 0

            if self.total_steps > self.warmup_steps:
                self.update()

            self.total_steps += 1

            if self.total_steps % 5000 == 0:
                self.inspection(self.total_steps)

    def inspection(self, iteration):
        print(f"tryinstpection...{self.env.unwrapped.spec.id}")
        self.save_model(self.app.PATH + f'\\agents\\checkpoints\\{self.env.unwrapped.spec.id}_sac_model.pth')
        reward = self.app.test_agent(True)
        self.app.update_plot(iteration, reward)

    def save_model(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'q_net1': self.q_net1.state_dict(),
            'q_net2': self.q_net2.state_dict(),
            'vf': self.vf.state_dict(),
            'vf_target': self.vf_target.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.q_net1.load_state_dict(checkpoint['q_net1'])
        self.q_net2.load_state_dict(checkpoint['q_net2'])
        self.vf.load_state_dict(checkpoint['vf'])
        self.vf_target.load_state_dict(checkpoint['vf_target'])
