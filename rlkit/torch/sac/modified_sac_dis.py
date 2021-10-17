from collections import OrderedDict
import os
import traceback
import gc

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.core.trainer import Trainer
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core import logger


class SoftActorCritic(Trainer):
    """
    version that:
        - uses reparameterization trick
        - has two Q functions and a V function
    """

    def __init__(
            self,
            policy,
            qfs,
            vf,
            lambda_td,
            target_qfs=None,

            reward_scale=1.0,
            discount=0.99,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            lambda_lr=1e-3,
            soft_target_tau=1e-2,

            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,

            use_grad_clip=True,
            use_huber_loss=False,

            optimizer_class=optim.Adam,
            beta_1=0.9,
            q_lambda=2.0,
            log_lambda=None,
            q_lambda_min=0.01,
            q_lambda_max=10.0,
            ep_clip=0.2,
            use_seq_loss=False,
            target_thresh=40.0,

            q_update_times=1,
            bc_reg_weight=0.0
    ):
        self.policy = policy
        self.qfs = qfs if isinstance(qfs, list) else [qfs]
        self.vf = vf
        self.student_policy = None
        self.student_qfs = None
        self.lambda_td = lambda_td
        self.alpha_pi = 1.0
        self.alpha_q = 0.5
        self.gamma = 0.99
        self.beta_1 = beta_1

        self.reward_scale = reward_scale
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight

        if target_qfs is None:
            self.target_qfs = []
            for qf in self.qfs:
                self.target_qfs.append(qf.copy())
        else:
            self.target_qfs = target_qfs
        self.eval_statistics = None
        self.init_qfs = []
        for qf in self.qfs:
            self.init_qfs.append(qf.copy())
        self.init_policy = self.policy.copy()

        self.use_grad_clip = use_grad_clip
        self.use_huber_loss = use_huber_loss
        assert not self.use_huber_loss

        self.optimizer_class = optimizer_class
        self.policy_lr = policy_lr
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
            betas=(beta_1, 0.999)
        )
        self.student_policy_optimizer = None

        self.qf_lr = qf_lr
        self.qfs_optimizer = []
        for qf in self.qfs:
            self.qfs_optimizer.append(optimizer_class(
                qf.parameters(),
                lr=qf_lr,
                betas=(beta_1, 0.999)
            ))
        self.student_qfs_optimizer = None

        if log_lambda is None:
            self.log_lambda = ptu.zeros(1, requires_grad=True)
        else:
            self.log_lambda = log_lambda
        self.lambda_optimizer = optimizer_class(
            [self.log_lambda],
            lr=lambda_lr,
            betas=(beta_1, 0.999)
        )
        self.lambda_min = q_lambda_min
        self.lambda_max = q_lambda_max
        self.ep_clip = ep_clip
        self.use_seq_loss = use_seq_loss

        self.target_thresh = target_thresh
        print("TARGET THRESH = ", target_thresh)

        self.q_update_times = q_update_times
        self.bc_reg_weight = bc_reg_weight

        self.init_student()

    def q_min(self, obs, actions):
        qfs_new_actions = [qf(obs, actions) for qf in self.qfs]
        qfs_new_actions = torch.cat(qfs_new_actions, dim=-1)
        qfs_new_actions = torch.unsqueeze(torch.min(qfs_new_actions, dim=-1).values, dim=-1)
        return qfs_new_actions

    def student_q_min(self, obs, actions):
        qfs_new_actions = [sqf(obs, actions) for sqf in self.student_qfs]
        qfs_new_actions = torch.cat(qfs_new_actions, dim=-1)
        qfs_new_actions = torch.unsqueeze(torch.min(qfs_new_actions, dim=-1).values, dim=-1)
        return qfs_new_actions

    def q_mean(self, obs, actions):
        qfs_new_actions = [qf(obs, actions) for qf in self.qfs]
        qfs_new_actions = torch.cat(qfs_new_actions, dim=-1)
        qfs_new_actions = torch.unsqueeze(torch.mean(qfs_new_actions, dim=-1), dim=-1)
        return qfs_new_actions

    def student_q_mean(self, obs, actions):
        qfs_new_actions = [sqf(obs, actions) for sqf in self.student_qfs]
        qfs_new_actions = torch.cat(qfs_new_actions, dim=-1)
        qfs_new_actions = torch.unsqueeze(torch.mean(qfs_new_actions, dim=-1), dim=-1)
        return qfs_new_actions

    def init_student(self):
        self.student_qfs = []
        assert self.qfs != self.student_qfs, "Use deepcopy!!!!!"
        for qf in self.init_qfs:
            self.student_qfs.append(qf.copy())
        self.student_qfs_optimizer = []
        for sqf in self.student_qfs:
            self.student_qfs_optimizer.append(self.optimizer_class(
                sqf.parameters(),
                lr=self.qf_lr,
                betas=(self.beta_1, 0.999)
            ))
        self.student_policy = self.init_policy.copy()
        self.student_policy_optimizer = self.optimizer_class(
            self.student_policy.parameters(),
            lr=self.policy_lr,
            betas=(self.beta_1, 0.999)
        )
        for net in [self.student_policy] + self.student_qfs:
            net.to(ptu.device)

    def can_replace(self, buffer, trajs=None):
        return buffer.has_similar_traj(policy_func=self.student_policy, trajs=trajs)

    def replace(self, use_seq_training=True, **kwargs):
        if use_seq_training:
            self.train_student(**kwargs)
        rp = kwargs.get("replace_flag", True)
        if not rp:
            return
        del self.qfs
        del self.policy
        del self.qfs_optimizer
        del self.policy_optimizer
        gc.collect()
        self.qfs = self.student_qfs
        self.policy = self.student_policy
        self.qfs_optimizer = []
        for qf in self.student_qfs:
            self.qfs_optimizer.append(self.optimizer_class(
                qf.parameters(),
                lr=self.qf_lr,
                betas=(self.beta_1, 0.999)
            ))
        self.policy_optimizer = self.optimizer_class(
            self.policy.parameters(),
            lr=self.policy_lr,
            betas=(self.beta_1, 0.999)
        )
        self.init_student()

    def train_student(self, **kwargs):
        try:
            buffer = kwargs["buffer"]
        except KeyError as ke:
            traceback.print_exc()
            print("--- Must provide buffer data!")
            raise KeyError
        sp_buffer = kwargs.get("sp_buffer", None)
        num_seq_training = kwargs.get("num_seq_training", 5)
        batch_size = kwargs.get("batch_size", 256)
        if sp_buffer is not None:
            batch_size *= 4
        for t in range(num_seq_training):
            anneal = float(t) / num_seq_training
            batch = buffer.random_batch(batch_size // 4)
            batch = np_to_pytorch_batch(batch)
            batch['actions'] = self.policy.get_mean(batch['observations'])
            if sp_buffer is not None:
                sp_batch = sp_buffer.random_batch(batch_size - batch_size // 4)
                sp_batch = np_to_pytorch_batch(sp_batch)
                for key in batch.keys():
                    batch[key] = torch.cat([batch[key], sp_batch[key]])
            self.train_student_batch(batch, anneal)

    def train_student_batch(self, batch, anneal):
        obs = batch['observations']
        actions = batch['actions']

        if self.student_policy is None:
            self.init_student()
        alpha_pi = self.alpha_pi * (1 - anneal)
        alpha_q = self.alpha_q * (1 - anneal)

        '''
        KL divergence between two Gaussian distributions N1(m1, s1) and N2(m2, s2) is
        KL(N1, N2) = log(s2/s1) - 0.5 + (s1^2 + (m1-m2)^2) / (2 * s2^2)
        '''
        # loss_kl = self.student_policy.log_std - self.policy.log_std - 0.5 \
        #     + (self.student_policy.std ** 2 + (self.student_policy.get_mean(obs)
        #                                        - self.policy.get_mean(obs)
        #                                        ) ** 2
        #        ) / (2 * self.policy.std ** 2)
        loss_kl = (self.student_policy.get_mean(obs) - actions) ** 2

        loss_kl = torch.sum(loss_kl, dim=1, keepdim=True)

        loss_qs = torch.mean(torch.stack(
            [(q(obs, actions) - sq(obs, actions)) ** 2 for q, sq in zip(self.qfs, self.student_qfs)], dim=0
        ), dim=0)

        '''
        PG loss and TD loss require online sampling, and probably 
        go outside of the support of offline dataset. We only use
        sequence loss in distillation.
        '''
        # student_next_actions = self.student_policy(next_obs, return_log_prob=True)[0]
        # adv = rewards \
        #     + self.gamma * self.student_q_min(next_obs, student_next_actions).detach() \
        #     - self.student_q_min(obs, actions)
        # log_prob = self.policy.get_log_prob(obs, actions)
        # student_log_prob = self.student_policy.get_log_prob(obs, actions)
        # with torch.no_grad():
        #     rho = torch.exp(student_log_prob - log_prob)
        #     rho.clamp_(1 - self.ep_clip, 1 + self.ep_clip)
        #     rho_adv = rho * adv
        #
        # loss_pg = -student_log_prob * rho_adv
        # loss_td = adv ** 2 * rho
        #
        # assert loss_kl.shape == loss_pg.shape, "policy loss shapes mismatch!"
        # assert loss_qs.shape == loss_td.shape, "q loss shapes mismatch!"
        # if self.use_seq_loss:
        #     student_policy_loss = torch.mean(alpha_pi * loss_kl)
        #     student_q_loss = torch.mean(alpha_q * loss_qs)
        # else:
        #     student_policy_loss = torch.mean(alpha_pi * loss_kl + loss_pg)
        #     student_q_loss = torch.mean(alpha_q * loss_qs + self.lambda_td * loss_td)

        student_policy_loss = torch.mean(alpha_pi * loss_kl)
        student_q_loss = torch.mean(alpha_q * loss_qs)

        for i, opt in enumerate(self.student_qfs_optimizer):
            opt.zero_grad()
        student_q_loss.backward(retain_graph=True)
        for i, opt in enumerate(self.student_qfs_optimizer):
            clip_gradient(opt)
            opt.step()

        self.student_policy_optimizer.zero_grad()
        student_policy_loss.backward(retain_graph=True)
        clip_gradient(self.student_policy_optimizer)
        self.student_policy_optimizer.step()

    def train_step(self, batch, **kwargs):
        training_mode = kwargs.get("training_mode", None)
        anneal = kwargs.get("anneal", -1)

        if training_mode is None or training_mode == "rl":
            rewards = self.reward_scale * batch['rewards']
            terminals = batch['terminals']
            obs = batch['observations']
            actions = batch['actions']
            next_obs = batch['next_observations']

            # torch.autograd.set_detect_anomaly(True)

            """
            QF Loss
            """
            if self.q_update_times == -1:
                q_update_times = 1 + kwargs.get("extra_upd", 0)
            else:
                q_update_times = self.q_update_times if kwargs.get("extra_upd", False) else 1
            for _ in range(q_update_times):
                with torch.no_grad():
                    policy_outputs_ = self.policy(next_obs, return_log_prob=True)
                    next_actions = policy_outputs_[0].detach()
                    target_q_values = [qf(next_obs, next_actions) for qf in self.target_qfs]
                    target_q_values = torch.cat(target_q_values, dim=-1)
                    min_q_values = torch.min(target_q_values, dim=-1).values
                    target_v_values = torch.unsqueeze(min_q_values, dim=-1)
                    q_target = rewards + (1. - terminals) * self.discount * target_v_values

                with torch.no_grad():
                    _policy_outputs = self.policy(obs, return_log_prob=True)
                    _actions = _policy_outputs[0].detach()
                    _target_q_values = [qf(obs, _actions) for qf in self.target_qfs]
                    _target_q_values = torch.cat(_target_q_values, dim=-1)
                    _min_q_values = torch.min(_target_q_values, dim=-1).values
                    _q_target = torch.unsqueeze(_min_q_values, dim=-1)

                qfs_pred = [qf(obs, actions) for qf in self.qfs]
                _qfs_pred = [qf(obs, _actions) for qf in self.qfs]

                if self.use_huber_loss:
                    q_target_loss = [torch.mean(huber_loss(q_pred - q_target.detach())) for q_pred in qfs_pred]
                    q_extra_loss = [
                        self.log_lambda.exp() * torch.mean(
                            2 * huber_loss(_q_pred - _q_target.detach()) - self.target_thresh
                        )
                        for _q_pred in _qfs_pred
                    ]
                else:
                    q_target_loss = [0.5 * torch.mean((q_pred - q_target.detach()) ** 2) for q_pred in qfs_pred]
                    q_extra_loss = [
                        self.log_lambda.exp() * torch.mean(
                            (_q_pred - _q_target.detach()) ** 2 - self.target_thresh
                        )
                        for _q_pred in _qfs_pred
                    ]

                q_target_loss_mean = 0.0
                q_extra_loss_mean = 0.0
                qfs_loss = []
                lambda_loss = ptu.zeros(1, requires_grad=True)
                for q_loss, _q_loss in zip(q_target_loss, q_extra_loss):
                    q_target_loss_mean += np.mean(ptu.get_numpy(q_loss))
                    q_extra_loss_mean += np.mean(ptu.get_numpy(_q_loss))
                    qfs_loss.append(q_loss + _q_loss)
                    lambda_loss = lambda_loss + _q_loss

                q_target_loss_mean /= len(qfs_loss)
                q_extra_loss_mean /= len(qfs_loss)
                q_loss_mean = q_target_loss_mean + q_extra_loss_mean
                lambda_loss = - lambda_loss / len(qfs_loss)

                self.lambda_optimizer.zero_grad()
                lambda_loss.backward(retain_graph=True)
                # clip_gradient(self.lambda_optimizer)
                self.lambda_optimizer.step()
                self.log_lambda.data.clamp_(min=np.log(self.lambda_min), max=np.log(self.lambda_max))

                for i, opt in enumerate(self.qfs_optimizer):
                    opt.zero_grad()
                    qfs_loss[i].backward()
                    clip_gradient(opt)
                    opt.step()

            """
            Policy Loss
            """

            # Make sure policy accounts for squashing functions like tanh correctly!

            log_prob = self.policy.get_log_prob(obs, actions)
            mle_policy_loss = -1.0 * log_prob.mean()

            action_num = 10
            obs_ = obs.unsqueeze(1).repeat(1, action_num, 1).view(-1, obs.shape[1])

            policy_outputs = self.policy(obs_, return_log_prob=True)
            new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
            qfs_new_actions = self.q_min(obs_, new_actions)
            # qfs_new_actions = [qf(obs_, new_actions) for qf in self.qfs]
            # qfs_new_actions = torch.cat(qfs_new_actions, dim=-1)
            # qfs_new_actions = torch.unsqueeze(torch.min(qfs_new_actions, dim=-1).values, dim=-1)

            policy_grad_loss = torch.mean(-qfs_new_actions)
            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
            bc_reg_loss = self.bc_reg_weight * mle_policy_loss
            policy_reg_loss = mean_reg_loss + std_reg_loss + bc_reg_loss
            policy_loss = policy_grad_loss + policy_reg_loss

            """
            Update networks
            """

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            clip_gradient(self.policy_optimizer)
            self.policy_optimizer.step()

            self._update_target_network()

            """
            Student loss
            """

            if anneal > -1:
                if self.student_policy is None:
                    self.init_student()
                alpha_pi = self.alpha_pi * (1 - anneal)
                alpha_q = self.alpha_q * (1 - anneal)
                '''
                KL divergence between two Gaussian distributions N1(m1, s1) and N2(m2, s2) is
                KL(N1, N2) = log(s2/s1) - 0.5 + (s1^2 + (m1-m2)^2) / (2 * s2^2)
                '''
                loss_kl = self.student_policy.log_std - self.policy.log_std - 0.5 \
                          + (self.student_policy.std ** 2 + (self.student_policy.get_mean(obs)
                                                             - self.policy.get_mean(obs)
                                                             ) ** 2
                             ) / (2 * self.policy.std ** 2)
                loss_kl = torch.sum(loss_kl, dim=1, keepdim=True)

                loss_qs = torch.mean(torch.stack(
                    [(q(obs, actions) - sq(obs, actions)) ** 2 for q, sq in zip(self.qfs, self.student_qfs)], dim=0
                ), dim=0)

                student_next_actions = self.student_policy(next_obs, return_log_prob=True)[0]
                adv = rewards \
                      + self.gamma * self.student_q_min(next_obs, student_next_actions).detach() \
                      - self.student_q_min(obs, actions)
                log_prob = self.policy.get_log_prob(obs, actions)
                student_log_prob = self.student_policy.get_log_prob(obs, actions)
                with torch.no_grad():
                    rho = torch.exp(student_log_prob - log_prob)
                    rho.clamp_(1 - self.ep_clip, 1 + self.ep_clip)
                    rho_adv = rho * adv

                loss_pg = -student_log_prob * rho_adv
                loss_td = adv ** 2 * rho

                assert loss_kl.shape == loss_pg.shape, "policy loss shapes mismatch!"
                assert loss_qs.shape == loss_td.shape, "q loss shapes mismatch!"
                if self.use_seq_loss:
                    student_policy_loss = torch.mean(alpha_pi * loss_kl)
                    student_q_loss = torch.mean(alpha_q * loss_qs)
                else:
                    student_policy_loss = torch.mean(alpha_pi * loss_kl + loss_pg)
                    student_q_loss = torch.mean(alpha_q * loss_qs + self.lambda_td * loss_td)

                for i, opt in enumerate(self.student_qfs_optimizer):
                    opt.zero_grad()
                student_q_loss.backward(retain_graph=True)
                for i, opt in enumerate(self.student_qfs_optimizer):
                    clip_gradient(opt)
                    opt.step()

                self.student_policy_optimizer.zero_grad()
                student_policy_loss.backward(retain_graph=True)
                clip_gradient(self.student_policy_optimizer)
                self.student_policy_optimizer.step()

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
                self.eval_statistics['QFs Loss'] = q_loss_mean
                self.eval_statistics['QFs Target Loss'] = q_target_loss_mean
                self.eval_statistics['QFs Extra Loss'] = q_extra_loss_mean
                self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
                self.eval_statistics['Lambda'] = ptu.get_numpy(self.log_lambda.exp())
                self.eval_statistics['Lambda Loss'] = np.mean(ptu.get_numpy(lambda_loss))
                if anneal > -1:
                    self.eval_statistics['Student KL Loss'] = ptu.get_numpy(torch.mean(alpha_pi * loss_kl))
                    self.eval_statistics['Student PG Loss'] = ptu.get_numpy(torch.mean(loss_pg))
                    self.eval_statistics['Student QF Loss'] = ptu.get_numpy(torch.mean(alpha_q * loss_qs))
                    self.eval_statistics['Student TD Loss'] = ptu.get_numpy(torch.mean(self.lambda_td * loss_td))
                    self.eval_statistics['Student TMP Adv'] = ptu.get_numpy(torch.mean(adv))
                    self.eval_statistics['Student TMP Rho'] = ptu.get_numpy(torch.mean(rho))
                    self.eval_statistics['Student Log Prob'] = ptu.get_numpy(torch.mean(student_log_prob))
                else:
                    self.eval_statistics['Student KL Loss'] = self.eval_statistics['Student PG Loss'] \
                        = self.eval_statistics['Student QF Loss'] = self.eval_statistics['Student TD Loss'] \
                        = self.eval_statistics['Student TMP Adv'] = self.eval_statistics['Student TMP Rho'] \
                        = self.eval_statistics['Student Log Prob'] = 0
                for i, q_pred in enumerate(qfs_pred):
                    self.eval_statistics.update(create_stats_ordered_dict(
                        'Q%d Predictions' % (i + 1),
                        ptu.get_numpy(q_pred),
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

        elif training_mode == "bc" or training_mode == "pretrain":
            obs = batch["observations"]
            acts = batch["actions"]

            log_prob = self.policy.get_log_prob(obs, acts)
            policy_loss = -1.0 * log_prob.mean()

            policy_outputs = self.policy(obs, return_log_prob=True)
            new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

            # squared_diff = (new_actions - acts) ** 2
            # policy_loss = torch.sum(squared_diff, dim=-1).mean()

            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
            policy_reg_loss = mean_reg_loss + std_reg_loss
            policy_loss = policy_loss + policy_reg_loss

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            '''
            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
                self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                    policy_loss
                ))
            '''

        else:
            raise ValueError("Invalid training mode")

    @property
    def networks(self):
        return [self.policy] + [self.student_policy] + self.qfs + self.target_qfs + self.student_qfs

    def _update_target_network(self):
        for qf, target_qf in zip(self.qfs, self.target_qfs):
            ptu.soft_update_from_to(qf, target_qf, self.soft_target_tau)

    def get_snapshot(self):
        return dict(
            qfs=self.qfs,
            target_qfs=self.target_qfs,
            policy=self.policy,
            log_lambda=self.log_lambda
        )

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        print("END_EPOCH")
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


def huber_loss(x, delta=1):
    return torch.where(
        torch.abs(x) < delta,
        0.5 * (x ** 2),
        delta * (torch.abs(x) - 0.5 * delta)
    )
