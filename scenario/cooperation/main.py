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
from scenario.cooperation.networks import Policy
import seaborn as sns
import pathlib
from scenario.utils.envs import Envs

PROJECT_PATH = pathlib.Path(
    __file__).parent.absolute().as_posix()


class Agents:
    def __init__(self, envs, seed=0, device='cuda:0', lr_collector=2e-3, lr_guide=2e-3, gamma=0.99, max_steps=500,
                 fc_hidden=64, rnn_hidden=128, batch_size=64, iters_policy=40, iters_value=40, lam=0.97, clip_ratio=0.2,
                 target_kl=0.03, num_layers=1, grad_clip=1.0, entropy_factor=0.0, message_num=5, tau=1.0):
        # RNG seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Environment
        self.num_world_blocks = 5
        self.width = 5
        self.height = 5
        self.envs = envs
        self.obs_dim = (self.num_world_blocks,) + envs.observation_space.shape
        self.act_dim = envs.action_space.nvec[0]
        self.agents_num = envs.agents_num
        print('Observation shape:', self.obs_dim)
        print('Action number:', self.act_dim)
        print('Agent number:', self.agents_num)

        # Networks
        in_dim = self.obs_dim[0]*self.obs_dim[1]*self.obs_dim[2]
        self.device = torch.device(device)
        self.collector = Policy(
            in_dim, self.act_dim, message_num, fc_hidden=fc_hidden, rnn_hidden=rnn_hidden, num_layers=num_layers, tau=tau).to(self.device)
        self.guide = Policy(
            in_dim, self.act_dim, message_num, fc_hidden=fc_hidden, rnn_hidden=rnn_hidden, num_layers=num_layers, tau=tau).to(self.device)

        self.optimizer_c = optim.Adam(
            self.collector.parameters(), lr=lr_collector)
        self.optimizer_g = optim.Adam(
            self.guide.parameters(), lr=lr_guide)
        self.batch_size = batch_size
        self.iters_policy = iters_policy
        self.iters_value = iters_value
        self.criterion = nn.MSELoss()
        self.grad_clip = grad_clip

        self.num_layers = num_layers
        self.rnn_hidden = rnn_hidden
        self.reset_states()

        # RL
        self.max_rew = 0
        self.gamma = gamma
        self.lam = lam
        self.max_steps = max_steps
        self.buffer_c = Buffer(self.max_steps*self.batch_size,
                               self.obs_dim, self.gamma, self.lam)
        self.buffer_g = Buffer(self.max_steps*self.batch_size,
                               self.obs_dim, self.gamma, self.lam)
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.entropy_factor = entropy_factor

    def single_preprocess(self, obs):
        state = np.zeros((obs.size, self.num_world_blocks), dtype=np.uint8)
        state[np.arange(obs.size), obs.reshape(-1)] = 1
        state = state.reshape(obs.shape + (self.num_world_blocks,))
        return np.moveaxis(state, -1, 0)

    def preprocess(self, obs_list):
        print(obs_list)
        return [self.single_preprocess(obs) for obs in obs_list]

    def sample_batch(self):
        self.buffer_c.clear()
        self.buffer_g.clear()
        total_rews = []

        while True:
            obs_c, obs_g = self.preprocess(self.envs.reset())
            episode_rew = 0

            for step in range(self.max_steps):
                acts = self.get_actions(obs_c, obs_g)
                next_obs, rews, done, _ = self.envs.step(acts)
                next_obs_c, next_obs_g = self.preprocess(next_obs)

                self.buffer_c.store(obs_c, acts[0], rews[0])
                self.buffer_g.store(obs_g, acts[1], rews[1])

                obs = next_obs
                episode_rew += rews[0] + rews[1]

                if done:
                    break
            total_rews.append(episode_rew)
            self.reward_and_advantage()
            self.reset_states()
            if self.buffer_c.ptr >= self.batch_size*self.max_steps:
                break

        return total_rews

    def reward_and_advantage(self):
        for buffer, net in [(self.buffer_c, self.collector), (self.buffer_g, self.guide)]:
            obs = torch.as_tensor(
                buffer.obs_buf[buffer.last_ptr:buffer.ptr], dtype=torch.float32).reshape(1, buffer.ptr-buffer.last_ptr, -1).to(self.device)
            with torch.no_grad():
                values = net.value_only(obs).cpu().numpy()
            buffer.expected_returns()
            buffer.advantage_estimation(values, 0.0)
            buffer.next_episode()

    def get_actions(self, obs_c, obs_g):
        obs_c = torch.as_tensor(
            obs_c, dtype=torch.float32).reshape(1, 1, -1).to(self.device)
        obs_g = torch.as_tensor(
            obs_g, dtype=torch.float32).reshape(1, 1, -1).to(self.device)
        with torch.no_grad():
            act_dist_c, message_c, self.state_c = self.collector.next_action(
                obs_c, 1, self.state_c)
            act_dist_g, message_g, self.state_g = self.guide.next_action(
                obs_g, 1, self.state_g)

            act_c = act_dist_c.sample().cpu().item()
            act_g = act_dist_g.sample().cpu().item()

        return [act_c, act_g]

    def compute_policy_gradient(self, net, dist, act, adv, old_logp):
        logp = dist.log_prob(act)

        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*adv
        loss = -torch.min(ratio*adv, clipped).mean() - \
            self.entropy_factor*dist.entropy().mean()
        kl_approx = (old_logp - logp).mean().item()
        return loss, kl_approx

    def update_policy(self, net, opt, obs, act, adv, ret):
        full_loss = 0
        with torch.no_grad():
            old_logp = net.action_only(obs).log_prob(act).to(self.device)
        for i in range(self.iters_policy):
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
        pol_losses = []
        val_losses = []
        for index, (buffer, net, opt) in enumerate([(self.buffer_c, self.collector, self.optimizer_c),
                                                    (self.buffer_g, self.guide, self.optimizer_g)]):
            obs = torch.as_tensor(
                buffer.obs_buf[:buffer.ptr], dtype=torch.float32, device=self.device)
            obs = obs.reshape(self.batch_size, self.max_steps, -1)
            act = torch.as_tensor(
                buffer.act_buf[:buffer.ptr], dtype=torch.int32, device=self.device)
            ret = torch.as_tensor(
                buffer.ret_buf[:buffer.ptr], dtype=torch.float32, device=self.device)
            buffer.standardize_adv()
            adv = torch.as_tensor(
                buffer.adv_buf[:buffer.ptr], dtype=torch.float32, device=self.device)

            pol_losses.append(self.update_policy(net, opt, obs, act, adv, ret))
            val_losses.append(0)  # self.update_value(obs, ret))
        return pol_losses, val_losses

    def train(self, epochs, prev_rews=[]):
        epoch_rews = []

        for epoch in range(epochs):
            import time

            start = time.time()
            rews = self.sample_batch()
            mean_rew = np.array(rews).mean()
            epoch_rews.append(mean_rew)
            if mean_rew > self.max_rew:
                self.max_rew = mean_rew
                self.save(epoch_rews)
            print(time.time()-start)
            pol_losses, val_losses = self.update()

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
            'guide': self.guide.state_dict(),
            'optim_c': self.optimizer_c.state_dict(),
            'optim_g': self.optimizer_g.state_dict(),
            'rews': rews
        }, path)

    def load(self, path='{}/model.pt'.format(PROJECT_PATH)):
        checkpoint = torch.load(path)
        self.collector.state_dict(checkpoint['collector'])
        self.guide.state_dict(checkpoint['guide'])
        self.optimizer_c.load_state_dict(checkpoint['optim_c'])
        self.optimizer_g.load_state_dict(checkpoint['optim_g'])
        return checkpoint['rews']

    def test(self):
        obs = self.preprocess(self.envs.reset())
        episode_rew = 0

        while True:
            self.envs.render()
            act = self.get_actions(obs)
            obs, rew, done, _ = self.envs.step(act)
            obs = self.preprocess(obs)

            episode_rew += rew[0] + rew[1]

            if done:
                break
        self.reset_state()

    def reset_states(self):
        self.state_c = (
            torch.zeros(self.num_layers, 1, self.rnn_hidden,
                        device=self.device),
            torch.zeros(self.num_layers, 1, self.rnn_hidden,
                        device=self.device),
        )
        self.state_g = (
            torch.zeros(self.num_layers, 1, self.rnn_hidden,
                        device=self.device),
            torch.zeros(self.num_layers, 1, self.rnn_hidden,
                        device=self.device),
        )


if __name__ == "__main__":
    batch_size = 64
    envs = Envs(batch_size)
    agents = Agents(envs, batch_size=batch_size)
    agents.train(10)

    while True:
        input('Press enter to continue')
        agents.test()
