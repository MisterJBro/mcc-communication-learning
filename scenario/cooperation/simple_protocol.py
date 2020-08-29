import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import random
import gym
import numpy as np
from scenario.cooperation.buffers import Buffers
from scenario.cooperation.networks import Policy, Listener, Speaker
import seaborn as sns
import pathlib
from scenario.utils.envs import Envs

PROJECT_PATH = pathlib.Path(
    __file__).parent.absolute().as_posix()


class Agents:
    def __init__(self, seed=0, device='cuda:0', lr_collector=1e-3, lr_guide=1e-3, gamma=0.99, max_steps=500,
                 fc_hidden=64, rnn_hidden=128, batch_size=512, iters=20, lam=0.97, clip_ratio=0.2, target_kl=0.02,
                 num_layers=1, grad_clip=1.0, symbol_num=5, tau=1.0):
        # RNG seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Environment
        self.num_world_blocks = 5
        self.envs = Envs(batch_size, red_guides=1,
                         blue_collector=0, show_messages=True)
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
        self.collector = Listener(
            in_dim, self.act_dim, symbol_num, fc_hidden=fc_hidden, rnn_hidden=rnn_hidden, num_layers=num_layers, tau=tau).to(self.device)
        self.guide = Speaker(
            in_dim, self.act_dim, symbol_num, fc_hidden=fc_hidden, rnn_hidden=rnn_hidden, num_layers=num_layers, tau=tau).to(self.device)

        self.optimizer_c = optim.RMSprop(
            self.collector.parameters(), lr=lr_collector)
        self.optimizer_g = optim.RMSprop(
            self.guide.parameters(), lr=lr_guide)
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
        self.buffers = Buffers(self.batch_size, self.max_steps,
                               self.obs_dim, self.gamma, self.lam, self.symbol_num)
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl

    def single_preprocess(self, obs):
        state = np.zeros((obs.size, self.num_world_blocks), dtype=np.uint8)
        state[np.arange(obs.size), obs.reshape(-1)] = 1
        state = state.reshape(obs.shape + (self.num_world_blocks,))
        return np.moveaxis(state, -1, 0)

    def preprocess(self, obs_list):
        obs_c, obs_g = [], []
        for x in range(self.batch_size):
            obs_c.append(self.single_preprocess(obs_list[x][0]))
            obs_g.append(self.single_preprocess(obs_list[x][1]))
        return np.array([obs_c, obs_g])

    def sample_batch(self):
        """ Samples a batch of trajectories """

        self.buffers.clear()
        batch_rew = np.zeros(self.batch_size)
        obs = self.preprocess(self.envs.reset())
        msg = torch.zeros((self.batch_size, self.symbol_num)).to(self.device)

        for step in range(self.max_steps):
            acts, next_msg = self.get_actions(obs, msg)
            next_obs, rews, _, _ = self.envs.step(acts)
            next_obs = self.preprocess(next_obs)

            team_rew = np.sum(rews, axis=1)
            self.buffers.store(obs, acts, team_rew, msg)
            batch_rew += team_rew

            obs = next_obs
            msg = next_msg
        self.reward_and_advantage()
        self.reset_states()

        return np.mean(batch_rew)

    def reward_and_advantage(self):
        """ Calculates the rewards and General Advantage Estimation """

        obs_c = torch.as_tensor(self.buffers.buffer_c.obs_buf, dtype=torch.float32).reshape(
            self.batch_size, self.max_steps, -1).to(self.device)
        obs_g = torch.as_tensor(self.buffers.buffer_g.obs_buf, dtype=torch.float32).reshape(
            self.batch_size, self.max_steps, -1).to(self.device)
        msg = self.buffers.backprop_msg

        with torch.no_grad():
            val_c = self.collector.value_only(obs_c, msg).reshape(
                self.batch_size, self.max_steps).cpu().numpy()
            val_g = self.guide.value_only(obs_g).reshape(
                self.batch_size, self.max_steps).cpu().numpy()

        self.buffers.expected_returns()
        self.buffers.advantage_estimation([val_c, val_g])
        self.buffers.standardize_adv()

    def get_actions(self, obs, msg):
        """ Gets action according the agents networks """

        obs = torch.as_tensor(obs, dtype=torch.float32).reshape(
            self.agents_num, self.batch_size, 1, -1).to(self.device)
        msg = torch.as_tensor(msg, dtype=torch.float32).reshape(
            self.batch_size, 1, -1).to(self.device)

        act_dist_c, self.state_c = self.collector.next_action(
            obs[0], msg, self.state_c)
        act_dist_g, next_msg, self.state_g = self.guide.next_action(
            obs[1], self.state_g)

        act_c = act_dist_c.sample().cpu().numpy()
        act_g = act_dist_g.sample().cpu().numpy()

        return np.stack([act_c, act_g]).T, next_msg

    def compute_policy_gradient(self, net, dist, act, adv, old_logp):
        logp = dist.log_prob(act)

        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*adv
        loss = -torch.min(ratio*adv, clipped).mean()
        kl_approx = (old_logp - logp).mean().item()
        return loss, kl_approx

    def update_net(self, net, opt, obs, act, adv, ret, msg=None, other_net=None, other_obs=None):
        full_loss = 0
        with torch.no_grad():
            if msg is not None:
                old_logp = net.action_only(
                    obs, msg).log_prob(act).to(self.device)
            else:
                old_logp = net.action_only(
                    obs).log_prob(act).to(self.device)

        for i in range(self.iters):
            self.optimizer_c.zero_grad()
            self.optimizer_g.zero_grad()

            if msg is not None:
                dist, vals = net(obs, msg)
            else:
                dist, _, vals = net(obs)

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

            self.optimizer_c.step()
            self.optimizer_g.step()
            if other_net is not None:
                _, msg, _ = other_net(other_obs[:, :-1])
                msg = msg.reshape(self.batch_size, self.max_steps-1, -1)
                msg = torch.cat(
                    (torch.zeros(self.batch_size, 1, self.symbol_num).to(self.device), msg), 1)

        return full_loss

    def update(self):
        obs_c, act_c, ret_c, adv_c, msg, obs_g, act_g, ret_g, adv_g = self.buffers.get_tensors(
            self.device)

        self.update_net(
            self.collector, self.optimizer_c, obs_c, act_c, adv_c, ret_c, msg=msg, other_net=self.guide, other_obs=obs_g)
        self.update_net(
            self.guide, self.optimizer_g, obs_g, act_g, adv_g, ret_g)

        return []

    def train(self, epochs):
        """ Trains the agent for given epochs """
        epoch_rews = []

        for epoch in range(epochs):
            rew = self.sample_batch()
            epoch_rews.append(rew)

            if rew > self.max_rew:
                self.max_rew = rew
                self.save()
            self.update()

            print('Epoch: {:4}  Average Reward: {:5}'.format(
                epoch, np.round(rew, 3)))

    def plot(self, arr, title='', xlabel='Epochs', ylabel='Average Reward'):
        sns.set()
        plt.plot(arr)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def save(self, path='{}/simple_protocol.pt'.format(PROJECT_PATH)):
        """ Saves the networks and optimizers to later continue training """
        torch.save({
            'collector': self.collector.state_dict(),
            'guide': self.guide.state_dict(),
            'optim_c': self.optimizer_c.state_dict(),
            'optim_g': self.optimizer_g.state_dict(),
        }, path)

    def load(self, path='{}/simple_protocol.pt'.format(PROJECT_PATH)):
        """ Loads a training checkpoint """
        checkpoint = torch.load(path)
        self.collector.load_state_dict(checkpoint['collector'])
        self.guide.load_state_dict(checkpoint['guide'])
        self.optimizer_c.load_state_dict(checkpoint['optim_c'])
        self.optimizer_g.load_state_dict(checkpoint['optim_g'])

    def test(self):
        obs = self.preprocess(self.envs.reset())
        msg = torch.zeros((self.batch_size, self.symbol_num)).to(self.device)
        episode_rew = 0
        self.last_message = ''

        for step in range(self.max_steps):
            import time
            # time.sleep(0.01)
            #msg[0] = torch.tensor([0., 0., 0., 1., 0.]).to(self.device)
            msg_show = msg[0].detach().cpu().numpy()
            self.last_message = self.interpret(msg_show)

            self.envs.envs[0].render(msg=self.interpret(msg_show))
            # print(msg[0].detach().cpu().numpy())
            acts, msg = self.get_actions(obs, msg)
            obs, rews, _, _ = self.envs.step(acts)
            obs = self.preprocess(obs)

            episode_rew += rews[0][0] + rews[0][1]
        print('Result reward: ', episode_rew)
        self.reset_states()

    def interpret(self, msg):
        if msg[0]:
            return self.last_message
        if msg[1]:
            return 'Treasure in Tunnel 1'
        if msg[2]:
            return 'Nothing in Tunnel 1'
        if msg[3]:
            return 'Nothing in Tunnel 1'
        if msg[4]:
            return 'Not ready!'

    def reset_states(self):
        """ Reset cell and hidden rnn states """

        self.state_c = (
            torch.zeros(self.num_layers, self.batch_size, self.rnn_hidden,
                        device=self.device),
            torch.zeros(self.num_layers, self.batch_size, self.rnn_hidden,
                        device=self.device),
        )
        self.state_g = (
            torch.zeros(self.num_layers, self.batch_size, self.rnn_hidden,
                        device=self.device),
            torch.zeros(self.num_layers, self.batch_size, self.rnn_hidden,
                        device=self.device),
        )


if __name__ == "__main__":
    agents = Agents()
    agents.load()

    while True:
        input('Press enter to continue')
        agents.test()
