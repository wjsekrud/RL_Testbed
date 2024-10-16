import gymnasium as gym
import torch
import argparse
import os

from agents.a3c_agent import A3CNetwork
from agents.sac_agent import PolicyNetwork
from agents.dpg_agent import ActorNetwork

def record_a3c_agent(global_model, env_name, video_dir, video_length=500):
    env = gym.make(env_name, render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, video_folder=video_dir, name_prefix='a3c')
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < video_length:
        logits, _ = global_model(state)
        action_prob = torch.softmax(logits, dim=-1)
        action = torch.argmax(action_prob, dim=1).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        steps += 1
    env.close()
    print(f"Total Reward: {total_reward}")
    print(f"Video saved to {video_dir}")

def record_sac_agent(policy_net, env_name, video_dir, video_length=500):
    env = gym.make(env_name, render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, video_folder=video_dir, name_prefix='sac')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action, mu, _ = policy_net(state_tensor)
        action = mu.detach().cpu().numpy()[0]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state
    env.close()
    print(f"Total Reward: {total_reward}")
    print(f"Video saved to {video_dir}")

def record_dpg_agent(actor_net, env_name, video_dir, video_length=500):
    env = gym.make(env_name, render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, video_folder=video_dir, name_prefix='dpg')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < video_length:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = actor_net(state_tensor).cpu().numpy()[0]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state
        steps += 1
    env.close()
    print(f"Total Reward: {total_reward}")
    print(f"Video saved to {video_dir}")

def record_agent(algorithm, env_name, video_dir, video_length):
    if algorithm == 'a3c':
        env = gym.make(env_name)
        global_model = A3CNetwork(env.observation_space.shape[0], env.action_space.n)
        global_model.load_state_dict(torch.load(os.getcwd() + f'\\agents\checkpoints\\{env_name}_a3c_model.pth'))
        record_a3c_agent(global_model, env_name, video_dir, video_length)

    elif algorithm == 'sac':
        env = gym.make(env_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0],
                                   action_space=env.action_space).to(device)
        checkpoint = torch.load(os.getcwd() + f'\\agents\checkpoints\\{env_name}_sac_model.pth')
        policy_net.load_state_dict(checkpoint['actor'])
        record_sac_agent(policy_net, env_name, video_dir, video_length)

    elif algorithm == 'dpg':
        env = gym.make(env_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        actor_net = ActorNetwork(env.observation_space.shape[0], env.action_space.shape[0],
                                 action_limit=env.action_space.high[0]).to(device)
        checkpoint = torch.load(os.getcwd() + f'\\agents\checkpoints\\{env_name}_dpg_model.pth')
        actor_net.load_state_dict(checkpoint['actor'])
        record_dpg_agent(actor_net, env_name, video_dir, video_length)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent Video Recorder')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Gymnasium environment name')
    parser.add_argument('--algorithm', type=str, default='a3c', choices=['a3c', 'sac', 'dpg'], help='Algorithm to use')
    parser.add_argument('--video_dir', type=str, default='videos', help='Directory to save videos')
    parser.add_argument('--video_length', type=int, default=500, help='Maximum steps to record')
    args = parser.parse_args()

    os.makedirs(args.video_dir, exist_ok=True)
    record_agent(args.algorithm, args.env_name, args.video_dir, args.video_length)
