import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
import random
import gym
import numpy as np
from scenario.mcc.buffers import Buffers
from scenario.mcc.networks import Policy, Listener, Speaker, Normal, ActionValue
from scenario.mcc.preprocess import preprocess, preprocess_state
import seaborn as sns
import pathlib
from scenario.utils.envs import Envs

PROJECT_PATH = pathlib.Path(
    __file__).parent.absolute().as_posix()


class Agents:
    def __init__(self, seed=0, device='cuda:0', lr_collector=6e-4, lr_guide=6e-4, lr_enemy=6e-4, lr_critic=8e-4, gamma=0.99,
                 fc_hidden=64, rnn_hidden=128, batch_size=64, lam=0.97, clip_ratio=0.2, iters=40, max_steps=500, critic_iters=80,
                 num_layers=1, grad_clip=1.0, symbol_num=5, tau=1.0):
        # RNG seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Environment
        self.num_world_blocks = 5
        self.envs = Envs(batch_size, red_guides=1,
                         blue_collector=1, competition=True)
        self.obs_dim = (self.num_world_blocks,) + \
            self.envs.observation_space.shape
        self.act_dim = self.envs.action_space.nvec[0]
        self.agents_num = self.envs.agents_num
        self.state_dim = (self.num_world_blocks,) + self.envs.state_dim
        print('Observation shape:', self.obs_dim)
        print('Action number:', self.act_dim)
        print('Agent number:', self.agents_num)

        # Networks
        in_dim = self.obs_dim[0]*self.obs_dim[1]*self.obs_dim[2]+2
        self.device = torch.device(device)
        self.collector = Listener(
            in_dim, self.act_dim, symbol_num, fc_hidden=fc_hidden, rnn_hidden=rnn_hidden, num_layers=num_layers, tau=tau).to(self.device)
        self.guide = Speaker(
            in_dim, self.act_dim, symbol_num, fc_hidden=fc_hidden, rnn_hidden=rnn_hidden, num_layers=num_layers, tau=tau).to(self.device)
        self.enemy = Normal(
            in_dim, self.act_dim, symbol_num, fc_hidden=fc_hidden, rnn_hidden=rnn_hidden, num_layers=num_layers).to(self.device)
        self.central_critic = ActionValue(
            self.state_dim[0]*self.state_dim[1]*self.state_dim[2], 2).to(self.device)

        self.optimizer_c = optim.Adam(
            self.collector.parameters(), lr=lr_collector)
        self.optimizer_g = optim.Adam(
            self.guide.parameters(), lr=lr_guide)
        self.optimizer_e = optim.Adam(
            self.enemy.parameters(), lr=lr_enemy)
        self.optimizer_cc = optim.Adam(
            self.central_critic.parameters(), lr=lr_critic)

        milestones = [2000, 4000, 5000]
        self.scheduler_c = MultiStepLR(
            self.optimizer_c, milestones=milestones, gamma=0.5)
        self.scheduler_g = MultiStepLR(
            self.optimizer_g, milestones=milestones, gamma=0.5)
        self.scheduler_e = MultiStepLR(
            self.optimizer_e, milestones=milestones, gamma=0.5)
        self.batch_size = batch_size
        self.val_criterion = nn.MSELoss()
        self.act_val_criterion = nn.MSELoss()
        self.pred_criterion = nn.CrossEntropyLoss()

        self.iters = iters
        self.critic_iters = critic_iters
        self.red_iters = self.iters
        self.blue_iters = self.iters

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
                               (in_dim,), self.act_dim, self.gamma, self.lam, self.symbol_num, self.state_dim)
        self.clip_ratio = clip_ratio

    def sample_batch(self):
        """ Samples a batch of trajectories """
        self.buffers.clear()
        batch_rew = np.zeros((3, self.batch_size))
        obs = preprocess(self.envs.reset())
        msg = torch.zeros((self.batch_size, self.symbol_num)).to(self.device)

        for step in range(self.max_steps):
            #import time
            # time.sleep(0.5)
            # self.envs.envs[0].render()
            acts, dsts, next_msg = self.get_actions(obs, msg)
            next_obs, rews, _, states = self.envs.step(acts)
            next_obs = preprocess(next_obs)

            states = np.array(preprocess_state(states))

            self.buffers.store(
                obs, acts, dsts, rews[:, 0], rews[:, 1], rews[:, 2], msg, states)
            batch_rew[0] += rews[:, 0]
            batch_rew[1] += rews[:, 1]
            batch_rew[2] += rews[:, 2]

            obs = next_obs
            msg = next_msg
        self.reward_and_advantage()
        self.reset_states()

        return np.mean(batch_rew, 1)

    def reward_and_advantage(self):
        """ Calculates the rewards and General Advantage Estimation """
        obs_c = torch.as_tensor(self.buffers.buffer_c.obs_buf, dtype=torch.float32).reshape(
            self.batch_size, self.max_steps, -1).to(self.device)
        obs_g = torch.as_tensor(self.buffers.buffer_g.obs_buf, dtype=torch.float32).reshape(
            self.batch_size, self.max_steps, -1).to(self.device)
        obs_e = torch.as_tensor(self.buffers.buffer_e.obs_buf, dtype=torch.float32).reshape(
            self.batch_size, self.max_steps, -1).to(self.device)
        msg = self.buffers.backprop_msg

        with torch.no_grad():
            val_c = self.collector.value_only(obs_c, msg).reshape(
                self.batch_size, self.max_steps).cpu().numpy()
            val_g = self.guide.value_only(obs_g).reshape(
                self.batch_size, self.max_steps).cpu().numpy()
            val_e = self.enemy.value_only(obs_e).reshape(
                self.batch_size, self.max_steps).cpu().numpy()

        self.buffers.expected_returns()
        self.buffers.advantage_estimation([val_c, val_g, val_e])
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
        act_dist_e, self.state_e = self.enemy.next_action(
            obs[2], self.state_e)

        act_c = act_dist_c.sample().cpu().numpy()
        act_g = act_dist_g.sample().cpu().numpy()
        # act_dist_e.sample().cpu().numpy()
        act_e = np.ones(self.batch_size)*4

        dst_c = act_dist_c.probs.cpu().detach().numpy()
        dst_g = act_dist_g.probs.cpu().detach().numpy()
        dst_e = act_dist_e.probs.cpu().detach().numpy()

        return np.stack([act_c, act_g, act_e]).T, np.stack([dst_c, dst_g, dst_e]), next_msg

    def compute_policy_gradient(self, net, dist, act, adv, old_logp):
        """ Computes the policy gradient with PPO """
        logp = dist.log_prob(act)

        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*adv
        loss = -(torch.min(ratio*adv, clipped)).mean()
        kl_approx = (old_logp - logp).mean().item()
        return loss, kl_approx

    def update_net(self, net, opt, obs, act, adv, ret, iters, msg=None, other_net=None, other_obs=None, other_opt=None, other_act=None, other_adv=None, other_ret=None, enemy=False):
        """ Updates the net """
        policy_loss = 0
        value_loss = 0
        other_done = False
        with torch.no_grad():
            if msg is not None:
                old_logp = net.action_only(
                    obs, msg).log_prob(act).to(self.device)
            else:
                old_logp = net.action_only(
                    obs).log_prob(act).to(self.device)

            if other_net is not None:
                other_old_logp = other_net.action_only(
                    other_obs).log_prob(other_act).to(self.device)

        for i in range(iters):
            opt.zero_grad()
            if other_opt is not None:
                other_opt.zero_grad()

            if enemy:
                dist, vals = net(obs)
            elif msg is not None:
                dist, vals = net(obs, msg)
            else:
                dist, _, vals = net(obs)

            loss, kl = self.compute_policy_gradient(
                net, dist, act, adv, old_logp)
            policy_loss += loss.item()
            if kl > 0.03:
                return policy_loss, value_loss
            loss.backward(retain_graph=True)

            loss = self.val_criterion(vals.reshape(-1), ret)
            value_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), self.grad_clip)

            if other_net is not None and other_obs is not None:
                other_dist, _,  other_vals = other_net(other_obs)
                other_loss, other_kl = self.compute_policy_gradient(
                    other_net, other_dist, other_act, other_adv, other_old_logp)

                if other_kl > 0.01 and not other_done:
                    other_done = True
                else:
                    other_loss.backward(retain_graph=True)

                    other_loss = self.val_criterion(
                        other_vals.reshape(-1), other_ret)
                    other_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        other_net.parameters(), self.grad_clip)

            opt.step()
            if other_opt is not None:
                other_opt.step()
            if other_net is not None:
                _, msg, _ = other_net(other_obs[:, :-1])

                if other_done:
                    msg = msg.detach()

                msg = msg.reshape(self.batch_size, self.max_steps-1, -1)
                msg = torch.cat(
                    (torch.zeros(self.batch_size, 1, self.symbol_num).to(self.device), msg), 1)

        return policy_loss, value_loss

    def update_critic(self, states, act_c, act_e, rew_c, ret_c):
        """ Updates the central critic. """
        total_loss = 0
        acts = torch.stack([act_c, act_e], dim=1)
        frozen_vals = self.central_critic(
            states, acts).reshape(self.batch_size, -1).detach()
        targets = rew_c.reshape(self.batch_size, -1).clone()
        targets[:, :-1] += self.gamma*frozen_vals[:, 1:]

        for iter in range(self.critic_iters):
            self.optimizer_cc.zero_grad()

            if iter % 5 == 0:
                frozen_vals = self.central_critic(
                    states, acts).reshape(self.batch_size, -1).detach()
                targets = rew_c.reshape(self.batch_size, -1).clone()
                targets[:, :-1] += self.gamma*frozen_vals[:, 1:]

            vals = self.central_critic(states, acts)

            loss_c = self.act_val_criterion(
                vals.reshape(-1), ret_c.reshape(-1))
            # print(states.shape)
            # print(states[0].cpu().numpy().reshape(
            #    5, 16, 22)[1], acts[0], vals[0], ret_c[0])
            total_loss += loss_c.item()
            loss_c.backward()

            self.optimizer_cc.step()

        return total_loss

    def calculate_advantage(self, states, act_c, act_e, dst_c, dst_e):
        """ Calculate the advantage using the central critic. """
        samples = self.batch_size*self.max_steps

        with torch.no_grad():
            all_acts = torch.arange(self.act_dim, dtype=torch.int32).repeat(
                samples).reshape(-1).to(self.device)
            repeated_act_e = act_e.repeat(
                self.act_dim).reshape(self.act_dim, -1).T.reshape(-1)
            repeated_act_c = act_c.repeat(
                self.act_dim).reshape(self.act_dim, -1).T.reshape(-1)

            all_acts_c = torch.stack(
                [all_acts, repeated_act_e], dim=1)
            all_acts_e = torch.stack(
                [repeated_act_c, all_acts], dim=1)

            repeated_states = states.repeat(
                1, self.act_dim).reshape(samples*5, -1)

            all_vals_c = self.central_critic(
                repeated_states, all_acts_c).reshape(-1, self.act_dim)
            all_vals_e = -self.central_critic(
                repeated_states, all_acts_e).reshape(-1, self.act_dim)

            vals_c = all_vals_c.gather(
                1, act_c.long().reshape(-1, 1)).reshape(-1)
            vals_e = all_vals_e.gather(
                1, act_e.long().reshape(-1, 1)).reshape(-1)

            adv_c = vals_c  # - (dst_c*all_vals_c).sum(1)
            adv_e = vals_e  # - (dst_e*all_vals_e).sum(1)

            #adv_c = adv_c.reshape(self.batch_size, self.max_steps)
            #adv_e = adv_e.reshape(self.batch_size, self.max_steps)

            # adv_c = (adv_c-adv_c.mean(1).reshape(-1, 1)) / \
            #    adv_c.std(1).reshape(-1, 1)
            # adv_e = (adv_e-adv_e.mean(1).reshape(-1, 1)) / \
            #    adv_e.std(1).reshape(-1, 1)

        return adv_c.reshape(-1), adv_e.reshape(-1)

    def update(self):
        """ Updates all nets """
        obs_c, act_c, rew_c, ret_c, dst_c, obs_g, act_g, ret_g, adv_g, dst_g, obs_e, act_e, ret_e, dst_e, msg, states = self.buffers.get_tensors()

        obs_c, act_c, rew_c, ret_c, adv_c, dst_c = self.buffers.get_collector_tensors()

        act_c, act_e = act_c.to(self.device), act_e.to(self.device)
        rew_c = rew_c.to(self.device)
        states = states.to(self.device)
        dst_c, dst_e = dst_c.to(self.device), dst_e.to(self.device)

        ret_c = ret_c.to(self.device)

        cc_loss = self.update_critic(states, act_c, act_e, rew_c, ret_c)
        _, adv_e = self.calculate_advantage(
            states, act_c, act_e, dst_c, dst_e)
        del states
        del rew_c
        del dst_c
        del dst_e

        adv_c = adv_c.to(self.device)
        obs_c, ret_c = obs_c.to(self.device), ret_c.to(self.device)
        obs_g, act_g, ret_g, adv_g = obs_g.to(self.device), act_g.to(
            self.device), ret_g.to(self.device), adv_g.to(self.device)

        msg_ent = Categorical(
            probs=msg.reshape(-1, self.symbol_num).detach().cpu().mean(0)).entropy().item()

        # Training Collector/Msg/Guide - Collector - Guide
        # _, _ = self.update_net(
        #    self.collector, self.optimizer_c, obs_c, act_c, adv_c, ret_c, self.red_iters, msg=msg, other_net=self.guide, other_obs=obs_g, other_opt=self.optimizer_g, other_act=act_g, other_adv=adv_g, other_ret=ret_g)
        p_loss_c, v_loss_c = self.update_net(
            self.collector, self.optimizer_c, obs_c, act_c, adv_c, ret_c, 0, msg=msg.detach())
        p_loss_g, v_loss_g = self.update_net(
            self.guide, self.optimizer_g, obs_g, act_g, adv_g, ret_g, 0)

        obs_e, ret_e = obs_e.to(self.device), ret_e.to(self.device)
        # p_loss_e, v_loss_e = self.update_net(
        #     self.enemy, self.optimizer_e, obs_e, act_e, adv_e, ret_e, self.blue_iters, enemy=True)

        self.scheduler_c.step()
        self.scheduler_g.step()
        self.scheduler_e.step()

        p_loss_e, v_loss_e = 0, 0

        return p_loss_c, v_loss_c, p_loss_g, v_loss_g, p_loss_e, v_loss_e, msg_ent, cc_loss

    def train(self, epochs):
        """ Trains the agent for given epochs """
        epoch_rews = []

        for epoch in range(epochs):
            rew = self.sample_batch()
            epoch_rews.append(rew)

            self.save()
            p_loss_c, v_loss_c, p_loss_g, v_loss_g, p_loss_e, v_loss_e, msg_ent, cc_loss = self.update()

            print('Epoch: {:4}  Collector Rew: {:4}  Enemy Rew: {:4}  Guide Rew: {:4}  Msg Ent {:4}  CC Loss: {:4}'.format(
                epoch, np.round(rew[0], 3),  np.round(rew[2], 3), np.round(rew[1], 1), np.round(msg_ent, 3), np.round(cc_loss, 3)))
        print(epoch_rews)

    def save(self, path='{}/model.pt'.format(PROJECT_PATH)):
        """ Saves the networks and optimizers to later continue training """
        torch.save({
            'collector': self.collector.state_dict(),
            'guide': self.guide.state_dict(),
            'enemy': self.enemy.state_dict(),
            'critic': self.central_critic.state_dict(),
            'optim_c': self.optimizer_c.state_dict(),
            'optim_g': self.optimizer_g.state_dict(),
            'optim_e': self.optimizer_e.state_dict(),
            'optim_cc': self.optimizer_cc.state_dict(),
        }, path)

    def load(self, path='{}/model.pt'.format(PROJECT_PATH)):
        """ Loads a training checkpoint """
        checkpoint = torch.load(path)
        self.collector.load_state_dict(checkpoint['collector'])
        self.guide.load_state_dict(checkpoint['guide'])
        self.enemy.load_state_dict(checkpoint['enemy'])
        self.central_critic.load_state_dict(checkpoint['critic'])
        self.optimizer_c.load_state_dict(checkpoint['optim_c'])
        self.optimizer_g.load_state_dict(checkpoint['optim_g'])
        self.optimizer_e.load_state_dict(checkpoint['optim_e'])
        self.optimizer_cc.load_state_dict(checkpoint['optim_cc'])

    def test(self):
        """ Tests the agent """
        obs = preprocess(self.envs.reset_with_score())
        msg = torch.zeros((self.batch_size, self.symbol_num)).to(self.device)
        episode_rew = 0
        msg_sum = np.zeros(self.symbol_num)

        for step in range(self.max_steps):
            # msg[0] = torch.tensor([0., 0., 0., 1., 0.]).to(self.device)

            self.envs.envs[0].render()
            print(msg[0].detach().cpu().numpy())
            msg_sum += msg[0].detach().cpu().numpy()
            acts, msg = self.get_actions(obs, msg)
            obs, rews, _, _ = self.envs.step_with_score(acts)
            obs = preprocess(obs)

            episode_rew += rews[0][0]
        print('Result reward: ', episode_rew)
        print(msg_sum)
        self.reset_states()

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
        self.state_e = (
            torch.zeros(self.num_layers, self.batch_size, self.rnn_hidden,
                        device=self.device),
            torch.zeros(self.num_layers, self.batch_size, self.rnn_hidden,
                        device=self.device),
        )


if __name__ == "__main__":
    agents = Agents()
    agents.load()
    agents.train(1000)

    import code
    # code.interact(local=locals())
    while True:
        input('Press enter to continue')
        agents.test()
