import torch.nn as nn
import torch

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(105, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, 8)
        self.std = nn.Linear(256, 8)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.std(x))

        return mean, std