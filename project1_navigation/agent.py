import numpy as np
import random
from collections import namedtuple, deque

from model import DQNetwork, DuelingDQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():

    def __init__(self, state_size, action_size, seed,
        replay_buffer_len=int(1e5), batch_size=64,
        gamma=0.99, #discount
        tau=1e-3, #soft target update
        lrate=5e-4,
        target_update_freq=4,
        double_dqn=False,
        dueling=False):
        """
        RL Agent that solves the Udacity Banana environment
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.replay_buffer_len = replay_buffer_len
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn

        #Dueling architecture
        if dueling == True:
            self.dqn_local = DuelingDQNetwork(state_size, action_size, seed).to(device)
            self.dqn_target = DuelingDQNetwork(state_size, action_size, seed).to(device)
        else:
            self.dqn_local = DQNetwork(state_size, action_size, seed).to(device)
            self.dqn_target = DQNetwork(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.dqn_local.parameters(), lr=lrate)

        #Experience buffer that randomly samples experiences from a moving window buffer
        self.replay_buffer = ExperienceReplayBuffer(action_size,
                                    replay_buffer_len, batch_size, seed)

        #Time step needed to update the target network
        self.t_epoch = 0

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done) #add experience
        self.t_epoch = (self.t_epoch + 1) % self.target_update_freq
        if self.t_epoch == 0:
            if len(self.replay_buffer) > int(self.batch_size):
                experiences = self.replay_buffer.sample()
                self.learn(experiences)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.dqn_local.eval()
        with torch.no_grad():
            action_values = self.dqn_local(state)
        self.dqn_local.train()

        # Epsilon-greedy
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        if self.double_dqn == True:
            #q_next_action = self.dqn_local(next_states).detach().max(1)[1].unsqueeze(1)
            q_next_action = torch.argmax(self.dqn_local(next_states), dim=-1).unsqueeze(1)
            q_next_state = self.dqn_target(next_states).detach().gather(1, q_next_action)
        else:
            #Get next state q value by taking max
            q_next_state = self.dqn_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_target = rewards + (self.gamma * q_next_state * (1 - dones))
        q_expected = self.dqn_local(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.dqn_local, self.dqn_target)

    def soft_update(self, local, target):
        #Update target network as a weighted average of local and target network
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

class ExperienceReplayBuffer():

    def __init__(self, action_size, buffer_len, batch_size, seed):

        self.action_size = action_size
        self.buffer = deque(maxlen=buffer_len)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action",
                                                    "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self):
        experiences = random.sample(self.buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)
