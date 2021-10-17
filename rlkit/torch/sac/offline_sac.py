from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from rlkit.core.trainer import Trainer
from rlkit.core.eval_util import create_stats_ordered_dict


class SoftActorCritic(Trainer):
    """
    version that:
        - uses reparameterization trick
        - has two Q functions and a V function
    TODO: Recently in rlkit there is a version which only uses two Q functions
    as well as an implementation of entropy tuning but I have not implemented
    those
    """
    def __init__(
            self,
            policy,
            qf1,
            qf2,
            vf,

            reward_scale=1.0,
            discount=0.99,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            soft_target_tau=1e-2,

            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,

            use_grad_clip=True,

            optimizer_class=optim.Adam,
            beta_1=0.9,
    ):
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.vf = vf
        self.reward_scale = reward_scale
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight

        self.target_vf = vf.copy()
        self.eval_statistics = None

        self.use_grad_clip = use_grad_clip

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
            betas=(beta_1, 0.999)
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
            betas=(beta_1, 0.999)
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
            betas=(beta_1, 0.999)
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
            betas=(beta_1, 0.999)
        )


    def train_step(self, batch):
        rewards = self.reward_scale * batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        v_pred = self.vf(obs)
         
        '''
        mean_expected_rew = self.target_vf(obs).detach().cpu()
        mean_expected_rew =  np.mean(np.array(mean_expected_rew).reshape(-1))
        print(mean_expected_rew)
        '''

        # Make sure policy accounts for squashing functions like tanh correctly!
        # policy_outputs = self.policy(obs, return_log_prob=True)
        action_num = 10
        obs = obs.unsqueeze(1).repeat(1, action_num, 1).view(-1, obs.shape[1])
        policy_outputs = self.policy(obs, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        """
        QF Loss
        """
        target_v_values = self.target_vf(next_obs)
        q_target = rewards + (1. - terminals) * self.discount * target_v_values
        qf1_loss = 0.5 * torch.mean((q1_pred - q_target.detach())**2)
        qf2_loss = 0.5 * torch.mean((q2_pred - q_target.detach())**2)

        """
        VF Loss
        """
        q1_new_acts = self.qf1(obs, new_actions)
        q2_new_acts = self.qf2(obs, new_actions)
        q_new_actions = torch.min(q1_new_acts, q2_new_acts)

        q_new_actions = q_new_actions.view(batch['actions'].shape[0], action_num, 1)
        q_new_actions, ind = torch.min(q_new_actions, dim=1)
        log_pi = log_pi.view(batch['actions'].shape[0], action_num)
        log_pi = torch.gather(log_pi, 1, ind)
        policy_log_std = policy_log_std.view(batch['actions'].shape[0], action_num, batch['actions'].shape[1])
        ind = ind.unsqueeze(2).repeat(1, 1, batch['actions'].shape[1])
        policy_log_std = torch.gather(policy_log_std, 1, ind)
        policy_mean = policy_mean.view(batch['actions'].shape[0], action_num, batch['actions'].shape[1])
        policy_mean = torch.gather(policy_mean, 1, ind)

        v_target = q_new_actions - log_pi
        vf_loss = 0.5 * torch.mean((v_pred - v_target.detach())**2)

        # print("v_tar: {}, log_pi:{}".format(torch.mean(v_target).detach().cpu(), torch.mean(log_pi).detach().cpu()))

        """
        Policy Loss
        """
        policy_loss = torch.mean(log_pi - q_new_actions)
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        policy_reg_loss = mean_reg_loss + std_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        clip_gradient(self.qf1_optimizer)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        clip_gradient(self.qf2_optimizer)
        self.qf2_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        clip_gradient(self.vf_optimizer)
        self.vf_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        clip_gradient(self.policy_optimizer)
        self.policy_optimizer.step()

        self._update_target_network()

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics['Reward Scale'] = self.reward_scale
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))


    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.vf,
            self.target_vf,
        ]


    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)


    def get_snapshot(self):
        return dict(
            qf1=self.qf1,
            qf2=self.qf2,
            policy=self.policy,
            vf=self.vf,
            target_vf=self.target_vf,
        )
    

    def get_eval_statistics(self):
        return self.eval_statistics
    

    def end_epoch(self):
        self.eval_statistics = None

def clip_gradient(optimizer, grad_clip=0.5):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)