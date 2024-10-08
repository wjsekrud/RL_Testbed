import gym
import torch
import argparse
import torch.nn.functional as F

from agents.a3c_agent import A3CNetwork
from agents.sac_agent import PolicyNetwork

def test_a3c_agent(global_model, env_name):
    env = gym.make(env_name)
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to('cuda')
    done = False
    total_reward = 0
    while not done:
        logits, _ = global_model(state)
        action_prob = F.softmax(logits, dim=-1)
        action = torch.argmax(action_prob, dim=1).item()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to('cuda')
        env.render()
    print(f"Total Reward: {total_reward}")

def test_sac_agent(policy_net, env_name):
    env = gym.make(env_name)
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = policy_net.select_action(state, evaluate=True)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        env.render()
    print(f"Total Reward: {total_reward}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Tester')
    parser.add_argument('--env_name', type=str, default='Pendulum-v1', help='Gym environment name')
    parser.add_argument('--algorithm', type=str, default='sac', choices=['a3c', 'sac'], help='Algorithm to use')

    args = parser.parse_args()

    if args.algorithm == 'a3c':
        env = gym.make(args.env_name)
        global_model = A3CNetwork(env.observation_space.shape[0], env.action_space.n).to('cuda')
        global_model.load_state_dict(torch.load('a3c_model.pth'))
        test_a3c_agent(global_model, args.env_name)

    elif args.algorithm == 'sac':
        env = gym.make(args.env_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0],
                                   action_space=env.action_space).to(device)
        checkpoint = torch.load('sac_model.pth')
        policy_net.load_state_dict(checkpoint['policy_net'])
        test_sac_agent(policy_net, args.env_name)
