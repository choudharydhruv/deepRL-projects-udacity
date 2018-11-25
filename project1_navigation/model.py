import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        """Deep Q-network with 2 FC layers
        """
        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DuelingDQNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        """Deep Q-network with 2 FC layers and Dueling architecture head
        """
        super(DuelingDQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.fc1_v = nn.Linear(128, 64)
        self.fc2_v = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        value = F.relu(self.fc1_v(x))
        value = self.fc2_v(value)

        x = F.relu(self.fc2(x))
        advantage = self.fc3(x)

        q = value.expand_as(advantage) +
                (advantage - advantage.mean(1, keepdim=True).expand_as(advantage))
        return q
