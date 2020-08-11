import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class Policy(nn.Module):
    """Policy Network of the agent."""

    def __init__(self, in_dim, out_dim, hid_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

    def dist(self, x):
        probs = self.forward(x)
        return Categorical(probs=probs)


# Environment
env = gym.make('CartPole-v1')
obs_dim = env.observation_space.shape
act_dim = env.action_space.n
print('Obs dim:', obs_dim)
print('Action dim:', act_dim)

# Hyperparameters
num_episodes = 10
max_t = 1000
lr = 6.25e-4
hidden_dim = 128

# Policy
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
policy = Policy(obs_dim[0], act_dim, hidden_dim).to(device)
optimizer = torch.optim.Adam(lr=lr)


def select_action(obs: np.array):
    """Selects an action by sampling from the policy."""
    with torch.no_grad():
        obs = torch.from_numpy(obs).unsqueeze(0).float().to(device)
        act = policy.dist(obs).sample().cpu().item()

    return act


for episode in range(num_episodes):

    obs = env.reset()
    for t in range(max_t):
        action = select_action(obs)
        env.render()
        next_obs, reward, done, _ = env.step(action)

        obs = next_obs

        if done:
            break

env.close()
