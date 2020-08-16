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
from scenario.utils.networks import Policy, ValueFunction
import seaborn as sns
import pathlib

PROJECT_PATH = pathlib.Path(
    __file__).parent.absolute().as_posix()


class Agents:
    def __init__(self, env, seed=0, device='cuda:0', lr_policy=2e-5, lr_value=2e-5, gamma=0.99, max_steps=500,
                 hidden_size=128, batch_size=256, iters_policy=40, iters_value=40, lam=0.97, clip_ratio=0.2,
                 target_kl=0.03, num_layers=1, grad_clip=1.0, entropy_factor=0.0):
        # RNG seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Environment
        self.num_world_blocks = 5
        self.width = 5
        self.height = 5
        self.env = env
        self.obs_dim = (self.num_world_blocks,) + env.observation_space.shape
        self.act_dim = env.action_space.nvec[0]
        self.agents_num = env.agents_num
        print('Observation shape:', self.obs_dim)
        print('Action number:', self.act_dim)
        print('Agent number:', self.agents_num)

        # Network
        in_dim = self.obs_dim[0]*self.obs_dim[1]*self.obs_dim[2]
        self.device = torch.device(device)
        self.policy = Policy(
            in_dim, self.act_dim, rnn_hidden=hidden_size,  num_layers=num_layers).to(self.device)
        self.value = ValueFunction(
            in_dim, rnn_hidden=hidden_size,  num_layers=num_layers).to(self.device)

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
        self.max_rew = 0
        self.gamma = gamma
        self.lam = lam
        self.max_steps = max_steps
        self.buffer_r = Buffer(self.max_steps*self.batch_size,
                               self.obs_dim, self.gamma, self.lam)
        self.buffer_b = Buffer(self.max_steps*self.batch_size,
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
        processed = [self.single_preprocess(obs) for obs in obs_list]
        tmp = np.copy(processed[1][4])
        processed[1][4] = np.copy(processed[1][3])
        processed[1][3] = tmp
        return np.stack(processed)

    def sample_batch(self):
        self.buffer_r.clear()
        self.buffer_b.clear()
        rews = []

        while True:
            obs = self.preprocess(self.env.reset())
            episode_rew = [0, 0]

            for step in range(self.max_steps):
                act = self.get_actions(obs)
                next_obs, rew, done, _ = self.env.step(act)
                next_obs = self.preprocess(next_obs)

                self.buffer_r.store(obs[0], act[0], rew[0])
                self.buffer_b.store(obs[1], act[1], rew[1])
                obs = next_obs
                episode_rew[0] += rew[0]
                episode_rew[1] += rew[1]

                if done:
                    break
            rews.append(episode_rew)
            self.reward_and_advantage()
            self.reset_state()
            if self.buffer_r.ptr >= self.batch_size*self.max_steps:
                break

        return rews

    def reward_and_advantage(self):
        for buffer in [self.buffer_r, self.buffer_b]:
            obs = torch.as_tensor(
                buffer.obs_buf[buffer.last_ptr:buffer.ptr], dtype=torch.float32).reshape(1, buffer.ptr-buffer.last_ptr, -1).to(self.device)
            with torch.no_grad():
                values = self.value(obs).cpu().numpy()
            buffer.expected_returns()
            buffer.advantage_estimation(values, 0.0)
            buffer.next_episode()

    def get_actions(self, obs):
        obs = torch.as_tensor(
            obs, dtype=torch.float32).reshape(self.agents_num, 1, -1).to(self.device)
        with torch.no_grad():
            dist, self.policy_state = self.policy.with_state(
                obs, self.policy_state)

        return dist.sample().cpu().numpy()

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
        pol_losses = []
        val_losses = []
        for index, buffer in enumerate([self.buffer_r, self.buffer_b]):
            obs = torch.as_tensor(
                buffer.obs_buf[:buffer.ptr], dtype=torch.float32, device=self.device)
            obs = obs.reshape(self.batch_size, self.max_steps,
                              self.num_world_blocks, self.height, self.width)
            act = torch.as_tensor(
                buffer.act_buf[:buffer.ptr], dtype=torch.int32, device=self.device)
            ret = torch.as_tensor(
                buffer.ret_buf[:buffer.ptr], dtype=torch.float32, device=self.device)
            buffer.standardize_adv()
            adv = torch.as_tensor(
                buffer.adv_buf[:buffer.ptr], dtype=torch.float32, device=self.device)

            pol_losses.append(self.update_policy(obs, act, adv))
            val_losses.append(self.update_value(obs, ret))
        return pol_losses, val_losses

    def train(self, epochs, prev_rews=[]):
        epoch_rews = []

        for epoch in range(epochs):
            rews = self.sample_batch()
            mean_rew = np.array(rews).mean(0)
            epoch_rews.append(mean_rew)
            if mean_rew.mean() > self.max_rew:
                self.max_rew = mean_rew.mean()
                self.save(epoch_rews)
            pol_losses, val_losses = self.update()

            print('Epoch: {:4}  Red Rew: {:6}  Blue Rew: {:6}  Average Reward: {:6}'.format(
                epoch, np.round(mean_rew[0], 3), np.round(mean_rew[1], 3), np.round(mean_rew.mean(), 3)))

    def plot(self, arr, title='', xlabel='Epochs', ylabel='Average Reward'):
        sns.set()
        plt.plot(arr)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def save(self, rews=[], path='{}/model.pt'.format(PROJECT_PATH)):
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optim_p': self.optimizer_policy.state_dict(),
            'optim_v': self.optimizer_value.state_dict(),
            'rews': rews
        }, path)

    def load(self, path='{}/model.pt'.format(PROJECT_PATH)):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        self.optimizer_policy.load_state_dict(checkpoint['optim_p'])
        self.optimizer_value.load_state_dict(checkpoint['optim_v'])
        return checkpoint['rews']

    def test(self):
        obs = self.preprocess(self.env.reset())
        episode_rew = [0, 0]

        while True:
            self.env.render()
            act = self.get_actions(obs)
            obs, rew, done, _ = self.env.step(act)
            obs = self.preprocess(obs)

            episode_rew[0] += rew[0]
            episode_rew[1] += rew[1]

            if done:
                break
        self.reset_state()
        self.env.close()

    def reset_state(self):
        self.policy_state = (
            torch.zeros(self.num_layers, self.agents_num, self.hidden_size,
                        device=self.device),
            torch.zeros(self.num_layers, self.agents_num, self.hidden_size,
                        device=self.device),
        )


if __name__ == "__main__":
    env = gym.make('gym_mcc_treasure_hunt:MCCTreasureHunt-v0',
                   red_guides=0, blue_collector=1)
    agents = Agents(env)
    print(np.array(agents.load()))
    #agents.max_rew = 3.73
    # agents.train(100)

    while True:
        input('Press enter to continue')
        agents.test()
