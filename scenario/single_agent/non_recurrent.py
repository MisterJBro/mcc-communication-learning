import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import random
import gym
import numpy as np
from scenario.utils.buffer import Buffer
import seaborn as sns
import pathlib

PROJECT_PATH = pathlib.Path(
    __file__).parent.absolute().as_posix()


class Policy(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Policy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, out_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        probs = self.model(x)
        return Categorical(probs=probs)


class ValueFunction(nn.Module):
    def __init__(self, in_dim):
        super(ValueFunction, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x).reshape(-1)


class Agents:
    def __init__(self, env, seed=0, device='cuda:0', lr_policy=2e-3, lr_value=2e-3, gamma=0.99, max_steps=500,
                 hidden_size=128, batch_size=64, iters_policy=40, iters_value=40, lam=0.97, clip_ratio=0.2,
                 target_kl=0.05, num_layers=1, grad_clip=1.0, entropy_factor=0.0):
        # RNG seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Environment
        self.num_world_blocks = 5
        self.width = 5
        self.height = 5
        self.env = env
        self.obs_dim = (self.num_world_blocks, self.height, self.width)
        self.act_dim = env.action_space.nvec[0]
        print('Observation shape:', self.obs_dim)
        print('Action number:', self.act_dim)

        # Network
        in_dim = self.obs_dim[0]*self.obs_dim[1]*self.obs_dim[2]
        self.device = torch.device(device)
        self.policy = Policy(
            in_dim, self.act_dim).to(self.device)
        self.value = ValueFunction(
            in_dim).to(self.device)

        self.optimizer_policy = optim.Adam(
            self.policy.parameters(), lr=lr_policy)
        self.optimizer_value = optim.Adam(
            self.value.parameters(), lr=lr_value)
        self.batch_size = batch_size
        self.iters_policy = iters_policy
        self.iters_value = iters_value
        self.criterion = nn.MSELoss()
        self.grad_clip = grad_clip

        # RL
        self.max_rew = 0
        self.gamma = gamma
        self.lam = lam
        self.max_steps = max_steps
        self.buffer = Buffer(self.max_steps*self.batch_size,
                             self.obs_dim, self.gamma, self.lam)
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.entropy_factor = entropy_factor

    def preprocess(self, obs):
        state = np.zeros((obs.size, self.num_world_blocks), dtype=np.uint8)
        state[np.arange(obs.size), obs.reshape(-1)] = 1
        state = state.reshape(obs.shape + (self.num_world_blocks,))
        return np.moveaxis(state, -1, 0)

    def sample_batch(self):
        self.buffer.clear()
        rews = []

        while True:
            obs = self.preprocess(self.env.reset()[0])
            episode_rew = 0

            for step in range(self.max_steps):
                act = self.get_action(obs)
                next_obs, rew, done, _ = self.env.step([act])
                next_obs = self.preprocess(next_obs[0])
                rew = rew[0]

                self.buffer.store(obs, act, rew)
                obs = next_obs
                episode_rew += rew

                if done:
                    break
            rews.append(episode_rew)
            self.reward_and_advantage()
            if self.buffer.ptr >= self.batch_size*self.max_steps:
                break

        return rews

    def reward_and_advantage(self):
        obs = torch.as_tensor(
            self.buffer.obs_buf[self.buffer.last_ptr:self.buffer.ptr], dtype=torch.float32).reshape(self.buffer.ptr-self.buffer.last_ptr, -1).to(self.device)
        with torch.no_grad():
            values = self.value(obs).cpu().numpy()
        self.buffer.expected_returns()
        self.buffer.advantage_estimation(values, 0.0)
        self.buffer.next_episode()

    def get_action(self, obs):
        obs = torch.as_tensor(
            obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            dist = self.policy(obs.reshape(1, -1))

        return dist.sample().item()

    def compute_policy_gradient(self, obs, act, adv, old_logp):
        dist = self.policy(obs)
        logp = dist.log_prob(act)

        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*adv
        loss = -torch.min(ratio*adv, clipped).mean() - \
            self.entropy_factor*dist.entropy().mean()
        kl_approx = (old_logp - logp).mean().item()
        return loss, kl_approx

    def update_policy(self, obs, act, adv):
        full_loss = 0
        with torch.no_grad():
            old_logp = self.policy(obs).log_prob(act).to(self.device)
        for i in range(self.iters_policy):
            self.optimizer_policy.zero_grad()
            loss, kl = self.compute_policy_gradient(obs, act, adv, old_logp)
            full_loss += loss.item()
            loss.backward()
            if kl > self.target_kl:
                return full_loss
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.grad_clip)
            self.optimizer_policy.step()
        return full_loss

    def update_value(self, obs, ret):
        full_loss = 0
        for i in range(self.iters_value):
            self.optimizer_value.zero_grad()
            input = self.value(obs)
            loss = self.criterion(input, ret)
            full_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.value.parameters(), self.grad_clip)
            self.optimizer_value.step()
        return full_loss

    def update(self):
        obs = torch.as_tensor(
            self.buffer.obs_buf[:self.buffer.ptr], dtype=torch.float32, device=self.device)
        obs = obs.reshape(self.batch_size*self.max_steps, -1)
        act = torch.as_tensor(
            self.buffer.act_buf[:self.buffer.ptr], dtype=torch.int32, device=self.device)
        ret = torch.as_tensor(
            self.buffer.ret_buf[:self.buffer.ptr], dtype=torch.float32, device=self.device)
        self.buffer.standardize_adv()
        adv = torch.as_tensor(
            self.buffer.adv_buf[:self.buffer.ptr], dtype=torch.float32, device=self.device)

        pol_loss = self.update_policy(obs, act, adv)
        val_loss = self.update_value(obs, ret)
        return pol_loss, val_loss

    def train(self, epochs, prev_rews=[]):
        epoch_rews = []
        for epoch in range(epochs):
            rews = self.sample_batch()
            mean_rew = np.mean(rews)
            epoch_rews.append(mean_rew)
            if mean_rew > self.max_rew:
                self.max_rew = mean_rew
                self.save(epoch_rews)
            pol_loss, val_loss = self.update()

            print('Epoch: {:4}  Average Reward: {:6}  Policy Loss: {:7}  Value Loss: {:7}'.format(
                epoch, np.round(mean_rew, 3), np.round(pol_loss, 4), np.round(val_loss, 4)))

    def plot(self, arr, title='', xlabel='Epochs', ylabel='Average Reward'):
        sns.set()
        plt.plot(arr)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def save(self, rews=[], path='{}/model_non_recurrent.pt'.format(PROJECT_PATH)):
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optim_p': self.optimizer_policy.state_dict(),
            'optim_v': self.optimizer_value.state_dict(),
            'rews': rews
        }, path)

    def load(self, path='{}/model_non_recurrent.pt'.format(PROJECT_PATH)):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        self.optimizer_policy.load_state_dict(checkpoint['optim_p'])
        self.optimizer_value.load_state_dict(checkpoint['optim_v'])
        return checkpoint['rews']

    def test(self):
        obs = self.preprocess(self.env.reset()[0])
        episode_rew = 0

        while True:
            import time
            self.env.render()
            act = self.get_action(obs)
            obs, rew, done, _ = self.env.step([act])
            obs = self.preprocess(obs[0])
            episode_rew += rew[0]
            time.sleep(0.01)

            if done:
                break


if __name__ == "__main__":
    env = gym.make('gym_mcc_treasure_hunt:MCCTreasureHunt-v0',
                   red_guides=0, blue_collector=0)
    agents = Agents(env)
    agents.load()
    while True:
        input('Press enter to continue')
        agents.test()
