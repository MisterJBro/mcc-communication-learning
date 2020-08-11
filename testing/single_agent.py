import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


env = gym.make('gym_mcc_treasure_hunt:MCCTreasureHunt-v0',
               red_guides=0, blue_collector=0)

# Hyperparameters
max_iters = 20000  # Also max batch size
num_epochs = 101
learning_rate = 2e-4
gamma = 0.95
batch_size = 250

# Environment
num_world_blocks = 5
width = 5
height = 5
obs_dim = width*height*num_world_blocks
act_dim = env.action_space.nvec[0]

# Use gpu if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PolicyNet(nn.Module):
    """ Policy network for single agent """

    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


# Create network and optimizer
policy = PolicyNet(obs_dim, act_dim).to(device)
optimizer = optim.RMSprop(policy.parameters(), lr=learning_rate)


def preprocessing(obs: np.array) -> torch.Tensor:
    """
    Preprocesses an observation, so that a network can easily work with it.
    Shape of result tensor: (HEIGHT, WIDTH, BLOCK_CHANNEL)
    """

    state = np.zeros((obs.size, num_world_blocks), dtype=np.uint8)
    state[np.arange(obs.size), obs.reshape(-1)] = 1
    state = state.reshape(obs.shape + (num_world_blocks,))
    state = torch.from_numpy(state)
    return state.float()


def get_policy(obs):
    """ Get the categorical policy distribution """
    obs = obs.to(device)
    probs = policy(obs)
    return Categorical(probs)


def get_action(obs):
    """ Get the action from the policy """
    return get_policy(obs).sample().item()


def compute_loss(obs, act, weights):
    """ Compute the gradient loss for the policy """
    obs = obs.to(device)
    act = act.to(device)
    weights = weights.to(device)
    logp = get_policy(obs).log_prob(act)
    return -(logp*weights).mean()


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


def train_one_epoch(t):
    """ Play batch size games to train to get one policy gradient update """
    # Lists for saving information
    batch_obs = []
    batch_acts = []
    batch_weights = []
    batch_rets = []
    batch_lens = []

    # Get first obs and save rewards
    obs = preprocessing(env.reset()[0])
    ep_rews = []

    # Play until done or max time steps reached
    for iter in range(max_iters):
        # Save obs
        batch_obs.append(obs)

        # Act in the environment
        act = get_action(obs.unsqueeze(0))
        obs, rew, done, _ = env.step([act])
        obs = preprocessing(obs[0])

        # Save action, reward
        batch_acts.append(act)

        if t == 100:
            env.render()

        if done:
            # reset episode-specific variables
            #obs, done, ep_rews = preprocessing(env.reset()[0]), False, []

            # end experience loop if we have enough of it
            # if len(batch_obs) > batch_size:
            ep_rews.append(10.0)
            break
        else:
            ep_rews.append(rew[0]-0.00001)
    # If episode is over, record info about episode
    ep_ret, ep_len = sum(ep_rews), len(ep_rews)
    batch_rets.append(ep_ret)
    batch_lens.append(ep_len)

    # The weight for each logprob(a|s) is R(tau)
    batch_weights += list(reward_to_go(ep_rews))

    # Get trainings data
    obs = torch.stack(batch_obs).to(device)
    act = torch.as_tensor(batch_acts, dtype=torch.int32).to(device)
    weights = torch.as_tensor(
        batch_weights, dtype=torch.float32).to(device)

    # Create dataset out of trainings data
    #train_data = TensorDataset(obs_set, act_set, weights_set)
    # train_loader = DataLoader(
    #    dataset=train_data, batch_size=batch_size, shuffle=True)

    # Update policy with gradient descent
    # for obs, act, weights in train_loader:
    for i in range(5):
        optimizer.zero_grad()
        batch_loss = compute_loss(obs, act, weights)
        batch_loss.backward()
        optimizer.step()
    return batch_loss, batch_rets, batch_lens


# Training loop
rewards = []
for i in range(num_epochs):
    batch_loss, batch_rets, batch_lens = train_one_epoch(i)
    print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %d' %
          (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
    rewards.append(np.mean(batch_rets))

plt.plot(rewards)
plt.show()

"""
plt.plot(allRewards)
plt.show()

obs = preprocessing(env.reset()[0])
for iter in range(max_iters):
    action_dist = policy(obs.to(device)).detach().cpu().numpy()
    action = [np.random.choice(
        num_actions, p=action_dist.reshape(-1))]

    obs, reward, done, _ = env.step(action)
    obs = preprocessing(obs[0])

    env.render()
    if done:
        break
"""
env.close()
