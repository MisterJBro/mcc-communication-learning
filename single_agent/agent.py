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
    def __init__(self, env, seed=0, device='cuda:0', lr_policy=1e-2, lr_value=1e-2, gamma=0.99, max_steps=10_000,
                 hidden_size=64, batch_size=2048, iters_policy=80, iters_value=80, lam=0.97, clip_ratio=0.2,
                 target_kl=0.05, num_layers=2, grad_clip=1.0):
        # RNG seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Environment
        self.env = env
        self.obs_dim = (2,)  # env.observation_space.shape
        self.act_dim = env.action_space.n
        print('Observation shape:', self.obs_dim)
        print('Action number:', self.act_dim)

        # Network
        self.device = torch.device(device)
        self.policy = Policy(
            self.obs_dim[0], self.act_dim, hidden_size,  num_layers).to(self.device)
        self.value = ValueFunction(
            self.obs_dim[0], hidden_size, num_layers).to(self.device)

        self.optimizer_policy = optim.Adam(
            self.policy.parameters(), lr=lr_policy)
        self.optimizer_value = optim.Adam(
            self.value.parameters(), lr=lr_value)
        self.batch_size = batch_size
        self.iters_policy = iters_policy
        self.iters_value = iters_value
        self.criterion = nn.MSELoss()
        self.grad_clip = grad_clip

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.reset_state()

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
            # POMDP change
            obs = obs[[0, 2]]

            episode_rew = 0

            for step in range(self.max_steps):
                act = self.get_action(obs)
                next_obs, rew, done, _ = self.env.step(act)
                # POMDP change
                next_obs = next_obs[[0, 2]]

                self.buffer.store(obs, act, rew)
                obs = next_obs
                episode_rew += rew

                if done:
                    break
            rews.append(episode_rew)
            self.reward_and_advantage()
            self.reset_state()

            if self.buffer.ptr > self.batch_size:
                break

        return rews

    def reward_and_advantage(self):
        obs = torch.as_tensor(
            self.buffer.obs_buf[self.buffer.last_ptr:self.buffer.ptr], dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            values = self.value(obs).cpu().numpy()
        self.buffer.expected_returns()
        self.buffer.advantage_estimation(values, 0.0)
        self.buffer.next_eipsode()

    def get_action(self, obs):
        obs = torch.as_tensor(
            obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, self.policy_state = self.policy.with_state(
                obs, self.policy_state)

        return dist.sample().item()

    def compute_policy_gradient(self, obs, act, adv, old_logp):
        logp = self.policy(obs).log_prob(act).to(self.device)

        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*adv
        loss = -torch.min(ratio*adv, clipped).mean()
        kl_approx = (old_logp - logp).mean().item()
        return loss, kl_approx

    def update_policy(self, obs, act, adv):
        full_loss = 0
        with torch.no_grad():
            old_logp = torch.empty(
                self.buffer.ptr, dtype=torch.float32, device=self.device)
            for start, end in self.buffer.ptrs:
                old_logp[start:end] = self.policy(
                    obs[start:end].unsqueeze(0)).log_prob(act[start:end]).to(self.device)
        for i in range(self.iters_policy):
            self.optimizer_policy.zero_grad()
            kls = []
            for start, end in self.buffer.ptrs:
                loss, kl = self.compute_policy_gradient(
                    obs[start:end].unsqueeze(0), act[start:end], adv[start:end], old_logp[start:end])
                kls.append(kl)
                full_loss += loss.item()
                loss.backward()
            if np.mean(kls) > self.target_kl:
                return full_loss
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.grad_clip)
            self.optimizer_policy.step()
        return full_loss

    def update_value(self, obs, ret):
        full_loss = 0
        for i in range(self.iters_value):
            self.optimizer_value.zero_grad()
            for start, end in self.buffer.ptrs:
                input = self.value(obs[start:end].unsqueeze(0))
                loss = self.criterion(input, ret[start:end])
                full_loss += loss.item()
                loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.value.parameters(), self.grad_clip)
            self.optimizer_value.step()
        return full_loss

    def update(self):
        obs = torch.as_tensor(
            self.buffer.obs_buf[:self.buffer.ptr], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(
            self.buffer.act_buf[:self.buffer.ptr], dtype=torch.int32, device=self.device)
        ret = torch.as_tensor(
            self.buffer.ret_buf[:self.buffer.ptr], dtype=torch.float32, device=self.device)
        #padded_obs, lens = self.buffer.get_padded_obs()
        # packed_seqs = torch.nn.utils.rnn.pack_padded_sequence(
        #    padded_obs, lengths=lens, batch_first=True, enforce_sorted=False).to(self.device)
        self.buffer.standardize_adv()
        adv = torch.as_tensor(
            self.buffer.adv_buf[:self.buffer.ptr], dtype=torch.float32, device=self.device)

        pol_loss = self.update_policy(obs, act, adv)
        val_loss = self.update_value(obs, ret)
        return pol_loss, val_loss

    def train(self, epochs):
        for epoch in range(epochs):
            rews = self.sample_batch()
            pol_loss, val_loss = self.update()

            print('Epoch: {:4}  Average Reward: {:6}  Policy Loss: {:8}  Value Loss: {:08}'.format(
                epoch, np.round(np.mean(rews), 3), np.round(pol_loss, 4), np.round(val_loss, 4)))

    def test(self):
        obs = self.env.reset()
        obs = obs[[0, 2]]
        episode_rew = 0

        while True:
            self.env.render()
            act = self.get_action(obs)
            obs, rew, done, _ = self.env.step(act)
            obs = obs[[0, 2]]
            episode_rew += rew

            if done:
                break
        self.reset_state()
        self.env.close()

    def reset_state(self):
        self.policy_state = (
            torch.zeros(self.num_layers, 1, self.hidden_size,
                        device=self.device),
            torch.zeros(self.num_layers, 1, self.hidden_size,
                        device=self.device),
        )


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = Agent(env)
    agent.train(70)
    while True:
        input('Press enter to continue')
        agent.test()
