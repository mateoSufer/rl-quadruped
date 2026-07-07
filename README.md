# Quadruped RL Agent

## Description
This project trains a quadruped agent to learn how to walk forward 
from scratch. At the start it moves randomly, and over time it learns 
to coordinate its eight joints to move forward efficiently.

## Algorithm
I implemented PPO (Proximal Policy Optimization) with an actor-critic 
architecture. The policy network outputs mean and standard deviation 
for each of the 8 joints, sampling actions from a Gaussian distribution. 
The value network estimates expected rewards to calculate the advantage. 
Training uses gradient clipping and policy clipping to keep updates 
stable, learning from batches of 2048 steps.

## Technologies
- Python
- PyTorch — actor and critic networks, training
- MuJoCo — quadruped physics simulation
- Gymnasium — environment interface
- Matplotlib — learning curve visualization

## Results
The agent goes from -3.2 to over 1.0 mean reward in 500 iterations 
with a clear learning curve. A key issue I found was that calculating 
the advantage without the value network produced near-zero deltas, 
making learning almost impossible. Adding the critic fixed this and 
the agent started improving immediately.
