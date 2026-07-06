from agent import PolicyNetwork
import matplotlib.pyplot as plt
import gymnasium
import torch

env = gymnasium.make("Ant-v5")

policy = PolicyNetwork()

optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)

states = []
actions = []
log_probs = []
rewards = []
dones = []

observation, info = env.reset()

for step in range(2048):
    state = torch.FloatTensor(observation)
    mean, std  = policy(state)
    dist = torch.distributions.Normal(mean, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum()

    action = action.detach().numpy()
    observation, reward, terminated, truncated, info = env.step(action)
    
    log_probs.append(log_prob)
    rewards.append(reward)
    states.append(state)
    dones.append(terminated or truncated)
    actions.append(action)

    if terminated or truncated:
        observation, info = env.reset