from agent import PolicyNetwork, ValueNetwork
import matplotlib.pyplot as plt
import gymnasium
import torch
import numpy

env = gymnasium.make("Ant-v5")

policy = PolicyNetwork()

critic = ValueNetwork()

optimizer_policy = torch.optim.Adam(policy.parameters(), lr=0.00003)

optimizer_critic = torch.optim.Adam(critic.parameters(), lr=0.00003)

observation, info = env.reset()

all_rewards = []


for iteration in range(500):

    states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values= []
    
    for step in range(2048):
        state = torch.FloatTensor(observation)
        mean, std  = policy(state)

        predict = critic(state)
        values.append(predict.item())
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        action = action.detach().numpy()
        observation, reward, terminated, truncated, info = env.step(action)
        
        log_probs.append(log_prob.detach())
        rewards.append(reward)
        states.append(state)
        dones.append(terminated or truncated)
        actions.append(action)

        if terminated or truncated:
            observation, info = env.reset()


    states = torch.stack(states)
    actions = torch.FloatTensor(numpy.array(actions))
    log_probs = torch.stack(log_probs)
    rewards = torch.FloatTensor(rewards)
    total_reward = rewards.mean().item() 
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    dones = torch.FloatTensor(dones)
    
    advantages = []
    gae = 0
    gamma = 0.99
    lam = 0.95

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    advantages = torch.FloatTensor(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(10):
        mean, std = policy(states)
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=1)
        
        ratio = torch.exp(new_log_probs - log_probs)
        clip_ratio = torch.clamp(ratio, 0.8, 1.2)
        loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
        
        optimizer_policy.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer_policy.step()

        values_pred = critic(states).squeeze()
        critic_loss = ((values_pred - rewards) ** 2).mean()
        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

    print(f"Iteration {iteration}, Mean Reward: {total_reward:.2f}")

    all_rewards.append(total_reward)

    torch.save(policy.state_dict(), "policy_weights.pth")

plt.plot(all_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Cuadruped Learning Curve")
plt.savefig("learning_curve.png")
plt.show()