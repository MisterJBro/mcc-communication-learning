import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import random
import gym
import numpy as np
from scenario.single_agent.buffer import Buffer
from scenario.single_agent.networks import Policy
import seaborn as sns
import pathlib
from scenario.utils.envs import Envs

PROJECT_PATH = pathlib.Path(
    __file__).parent.absolute().as_posix()


class Agents:
    def __init__(self, seed=0, device='cuda:0', lr_collector=1e-3, lr_guide=1e-3, gamma=0.99, max_steps=500,
                 fc_hidden=64, rnn_hidden=128, batch_size=128, iters=60, lam=0.97, clip_ratio=0.2, target_kl=0.05,
                 num_layers=1, grad_clip=1.0, symbol_num=5, tau=1.0):
        # RNG seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Environment
        self.num_world_blocks = 5
        self.envs = Envs(batch_size, red_guides=0, blue_collector=0)
        self.obs_dim = (self.num_world_blocks,) + \
            self.envs.observation_space.shape
        self.act_dim = self.envs.action_space.nvec[0]
        self.agents_num = self.envs.agents_num
        print('Observation shape:', self.obs_dim)
        print('Action number:', self.act_dim)
        print('Agent number:', self.agents_num)

        # Networks
        in_dim = self.obs_dim[0]*self.obs_dim[1]*self.obs_dim[2]
        self.device = torch.device(device)
        self.collector = Policy(
            in_dim, self.act_dim, symbol_num, fc_hidden=fc_hidden, rnn_hidden=rnn_hidden, num_layers=num_layers, tau=tau).to(self.device)
        self.optimizer_c = optim.Adam(
            self.collector.parameters(), lr=lr_collector)

        self.batch_size = batch_size
        self.iters = iters
        self.criterion = nn.MSELoss()
        self.grad_clip = grad_clip

        self.symbol_num = symbol_num
        self.num_layers = num_layers
        self.rnn_hidden = rnn_hidden
        self.reset_states()

        # RL
        self.max_rew = 0
        self.gamma = gamma
        self.lam = lam
        self.max_steps = max_steps
        self.buffer_c = Buffer(self.batch_size, self.max_steps,
                               self.obs_dim, self.gamma, self.lam)
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl

    def single_preprocess(self, obs):
        state = np.zeros((obs.size, self.num_world_blocks), dtype=np.uint8)
        state[np.arange(obs.size), obs.reshape(-1)] = 1
        state = state.reshape(obs.shape + (self.num_world_blocks,))
        return np.moveaxis(state, -1, 0)

    def preprocess(self, obs_list):
        obs_c = []
        for x in range(self.batch_size):
            obs_c.append(self.single_preprocess(obs_list[x][0]))
        return np.array(obs_c)

    def sample_batch(self):
        self.buffer_c.clear()
        episode_rew = np.zeros(self.batch_size)

        obs_c = self.preprocess(self.envs.reset())

        for step in range(self.max_steps):
            acts = self.get_actions(obs_c)
            next_obs, rews, _, _ = self.envs.step(acts)
            rews = np.array(rews)
            next_obs_c = self.preprocess(next_obs)

            episode_rew += rews[:, 0]
            self.buffer_c.store(obs_c, acts[:, 0], rews[:, 0])

            obs_c = next_obs_c
        self.reward_and_advantage()
        self.reset_states()

        return episode_rew

    def reward_and_advantage(self):
        obs = torch.as_tensor(self.buffer_c.obs_buf, dtype=torch.float32).reshape(
            self.batch_size, self.max_steps, -1).to(self.device)
        with torch.no_grad():
            values = self.collector.value_only(obs).reshape(
                self.batch_size, self.max_steps,).cpu().numpy()
        self.buffer_c.expected_returns()
        self.buffer_c.advantage_estimation(
            values, np.zeros((self.batch_size, 1)))

    def get_actions(self, obs_c):
        obs_c = torch.as_tensor(
            obs_c, dtype=torch.float32).reshape(self.batch_size, 1, -1).to(self.device)
        with torch.no_grad():
            act_dist_c, message_c, self.state_c = self.collector.next_action(
                obs_c, 1, self.state_c)
            act_c = act_dist_c.sample().cpu().numpy()

        return np.expand_dims(act_c, axis=1)

    def compute_policy_gradient(self, net, dist, act, adv, old_logp):
        logp = dist.log_prob(act)

        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*adv
        loss = -torch.min(ratio*adv, clipped).mean()
        kl_approx = (old_logp - logp).mean().item()
        return loss, kl_approx

    def update_net(self, net, opt, obs, act, adv, ret):
        full_loss = 0
        with torch.no_grad():
            old_logp = net.action_only(obs).log_prob(act).to(self.device)
        for i in range(self.iters):
            opt.zero_grad()
            dist, msgs, vals = net(obs, 0)

            loss, kl = self.compute_policy_gradient(
                net, dist, act, adv, old_logp)
            full_loss += loss.item()
            if kl > self.target_kl:
                return full_loss
            loss.backward(retain_graph=True)

            loss = self.criterion(vals.reshape(-1), ret)
            full_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), self.grad_clip)
            opt.step()
        return full_loss

    def update(self):
        losses = []
        obs = torch.as_tensor(
            self.buffer_c.obs_buf, dtype=torch.float32, device=self.device)
        obs = obs.reshape(self.batch_size, self.max_steps, -1)
        act = torch.as_tensor(
            self.buffer_c.act_buf, dtype=torch.int32, device=self.device).reshape(-1)
        ret = torch.as_tensor(
            self.buffer_c.ret_buf, dtype=torch.float32, device=self.device).reshape(-1)
        self.buffer_c.standardize_adv()
        adv = torch.as_tensor(
            self.buffer_c.adv_buf, dtype=torch.float32, device=self.device).reshape(-1)

        losses.append(self.update_net(self.collector,
                                      self.optimizer_c, obs, act, adv, ret))
        return losses

    def train(self, epochs, prev_rews=[]):
        epoch_rews = []

        for epoch in range(epochs):
            rews = self.sample_batch()
            mean_rew = rews.mean()
            epoch_rews.append(mean_rew)
            if mean_rew > self.max_rew:
                self.max_rew = mean_rew
                self.save(epoch_rews)
            losses = self.update()

            print('Epoch: {:4}  Average Reward: {:6}'.format(
                epoch, np.round(mean_rew, 3)))

    def plot(self, arr, title='', xlabel='Epochs', ylabel='Average Reward'):
        sns.set()
        plt.plot(arr)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def save(self, rews=[], path='{}/model.pt'.format(PROJECT_PATH)):
        torch.save({
            'collector': self.collector.state_dict(),
            'optim_c': self.optimizer_c.state_dict(),
            'rews': rews
        }, path)

    def load(self, path='{}/model.pt'.format(PROJECT_PATH)):
        checkpoint = torch.load(path)
        self.collector.load_state_dict(checkpoint['collector'])
        self.optimizer_c.load_state_dict(checkpoint['optim_c'])
        return checkpoint['rews']

    def test(self):
        obs = self.preprocess(self.envs.reset())
        episode_rew = 0

        for step in range(self.max_steps):
            self.envs.envs[0].render()
            act = self.get_actions(obs)
            obs, rew, _, _ = self.envs.step(act)
            rew = np.array(rew)
            obs = self.preprocess(obs)

            episode_rew += rew[0][0]
        print('Result reward: ', episode_rew)
        self.reset_states()

    def reset_states(self):
        self.state_c = (
            torch.zeros(self.num_layers, self.batch_size, self.rnn_hidden,
                        device=self.device),
            torch.zeros(self.num_layers, self.batch_size, self.rnn_hidden,
                        device=self.device),
        )


if __name__ == "__main__":
    agents = Agents()
    agents.max_rew = 100
    # agents.load()
    agents.train(200)

    while True:
        input('Press enter to continue')
        agents.test()
