import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import random
import gym
import numpy as np
from .buffer import Buffer
from .networks import Policy, ValueFunction


class Agent:
    def __init__(self, env, seed=0, device='cpu', lr_policy=3e-3, lr_value=5e-3, gamma=0.99, max_steps=10_000,
                 hidden_size=32, batch_size=4096, iters_policy=80, iters_value=80, lam=0.97, clip_ratio=0.2,
                 target_kl=0.05):
        # RNG seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Environment
        self.env = env
        self.obs_dim = env.observation_space.shape
        self.act_dim = env.action_space.n

        # Network
        self.device = torch.device(device)
        self.policy = Policy(
            self.obs_dim[0], self.act_dim, hidden_size).to(self.device)
        self.value = ValueFunction(
            self.obs_dim[0], hidden_size).to(self.device)

        self.optimizer_policy = optim.Adam(
            self.policy.parameters(), lr=lr_policy)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr_value)
        self.batch_size = batch_size
        self.iters_policy = iters_policy
        self.iters_value = iters_value
        self.criterion = nn.MSELoss()

        # RL
        self.gamma = gamma
        self.lam = lam
        self.buffer = Buffer(max_steps, self.obs_dim, self.gamma, self.lam)
        self.max_steps = max_steps
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl

    def sample_batch(self):
        self.buffer.clear()
        rews = []

        while True:
            obs = self.env.reset()
            episode_rew = 0

            for step in range(self.max_steps):
                act = self.get_action(obs)
                next_obs, rew, done, _ = self.env.step(act)

                self.buffer.store(obs, act, rew)
                obs = next_obs
                episode_rew += rew

                if done:
                    break
            rews.append(episode_rew)
            self.reward_and_adantage()

            if self.buffer.ptr > self.batch_size:
                break

        return rews

    def reward_and_adantage(self):
        obs = torch.as_tensor(
            self.buffer.obs_buf[self.buffer.last_ptr:self.buffer.ptr], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            values = self.value(obs).reshape(-1).cpu().numpy()
        self.buffer.expected_returns()
        self.buffer.advantage_estimation(values, 0.0)

        self.buffer.last_ptr = self.buffer.ptr

    def get_action(self, obs):
        obs = torch.as_tensor(
            obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.policy(obs)

        return dist.sample().item()

    def compute_policy_gradient(self, obs, act, adv, old_logp):
        logp = self.policy(obs).log_prob(act).to(self.device)

        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*adv
        loss = -torch.min(ratio*adv, clipped).mean()
        kl_approx = (old_logp - logp).mean().item()
        return loss, kl_approx

    def update_policy(self, obs, act, adv):
        with torch.no_grad():
            old_logp = self.policy(obs).log_prob(act).to(self.device)
        for i in range(self.iters_policy):
            self.optimizer_policy.zero_grad()
            loss, kl = self.compute_policy_gradient(obs, act, adv, old_logp)
            if kl > self.target_kl:
                break
            loss.backward()
            self.optimizer_policy.step()

    def update_value(self, obs, ret):
        for i in range(self.iters_value):
            self.optimizer_value.zero_grad()

            input = self.value(obs).reshape(-1)
            loss = self.criterion(input, ret)
            loss.backward()
            self.optimizer_value.step()

    def update(self):
        obs = torch.as_tensor(
            self.buffer.obs_buf[0:self.buffer.ptr], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(
            self.buffer.act_buf[0:self.buffer.ptr], dtype=torch.int32, device=self.device)
        ret = torch.as_tensor(
            self.buffer.ret_buf[0:self.buffer.ptr], dtype=torch.float32).to(self.device)
        self.buffer.standardize_adv()
        adv = torch.as_tensor(
            self.buffer.adv_buf[0:self.buffer.ptr], dtype=torch.float32, device=self.device)

        self.update_policy(obs, act, adv)
        self.update_value(obs, ret)

    def train(self, epochs):
        for epoch in range(epochs):
            rews = self.sample_batch()
            self.update()

            print('Epoch: {:4}  Average Reward: {:4}'.format(
                epoch, np.mean(rews)))

    def test(self):
        obs = self.env.reset()
        episode_rew = 0

        while True:
            self.env.render()
            act = self.get_action(obs)
            obs, rew, done, _ = self.env.step(act)
            episode_rew += rew

            if done:
                break
        self.env.close()


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    agent = Agent(env)
    agent.train(100)
    while True:
        input('Press enter to continue')
        agent.test()
