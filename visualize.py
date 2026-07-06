import gymnasium as gym
import torch
from agent import PolicyNetwork

env = gym.make("Ant-v5", render_mode="human")
policy = PolicyNetwork()
policy.load_state_dict(torch.load("policy_weights.pth"))

observation, info = env.reset()

for _ in range(5000):
    state = torch.FloatTensor(observation)
    mean, std = policy(state)
    action = mean.detach().numpy()
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()

