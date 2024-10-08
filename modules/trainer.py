import gym
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import argparse

from agents.a3c_agent import A3CNetwork, A3CAgent
from agents.sac_agent import SACAgent
from agents.dpg_agent import DPGAgent

def train_a3c_agent(global_model, optimizer, env_name, max_steps):
    env = gym.make(env_name)
    agent = A3CAgent(env, global_model, optimizer)
    agent.train(max_steps=max_steps)

def train_sac_agent(env_name, total_steps):
    env = gym.make(env_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = SACAgent(env, device=device)
    agent.train(total_steps=total_steps)
    agent.save_model('./checkpoints/sac_model.pth')

def train_dpg_agent(env_name, total_steps):
    env = gym.make(env_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DPGAgent(env, device=device)
    agent.train(total_steps=total_steps)
    agent.save_model('./checkpoints/pg_model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Trainer')
    parser.add_argument('--env_name', type=str, default='Pendulum-v1', help='Gym environment name')
    parser.add_argument('--algorithm', type=str, default='sac', choices=['a3c', 'sac', 'dpg'], help='Algorithm to use')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode for A3C')
    parser.add_argument('--total_steps', type=int, default=100000, help='Total training steps for SAC and DPG')

    args = parser.parse_args()

    if args.algorithm == 'a3c':
        env = gym.make(args.env_name)
        global_model = A3CNetwork(env.observation_space.shape[0], env.action_space.n).to('cuda')
        global_model.share_memory()
        optimizer = optim.Adam(global_model.parameters(), lr=1e-4)

        processes = []
        num_processes = mp.cpu_count()
        for rank in range(num_processes):
            p = mp.Process(target=train_a3c_agent, args=(global_model, optimizer, args.env_name, args.max_steps))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        torch.save(global_model.state_dict(), 'a3c_model.pth')

    elif args.algorithm == 'sac':
        train_sac_agent(args.env_name, args.total_steps)

    elif args.algorithm == 'dpg':
        train_dpg_agent(args.env_name, args.total_steps)
