import gymnasium as gym

env = gym.make("Ant-v5", render_mode =  "human")
obserbation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()