import numpy as np
import torch
from torch import nn as nn

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.distributions import ReparamTanhMultivariateNormal
from rlkit.torch.distributions import ReparamMultivariateNormalDiag
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy


    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)

    def get_actions(self, observations):
        return self.stochastic_policy.get_actions(observations,
                                                  deterministic=True)
    
    def train(self, mode):
        pass
    

    def set_num_steps_total(self, num):
        pass
    

    def to(self, device):
        self.stochastic_policy.to(device)


class DiscretePolicy(Mlp, ExplorationPolicy):
    """
    Usage:
    policy = DiscretePolicy(...)
    action, log_prob = policy(obs, return_log_prob=True)
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            output_activation=nn.LogSoftmax(1),
            **kwargs
        )

    def get_action(self, obs_np, deterministic=False):
        action = self.get_actions(obs_np[None], deterministic=deterministic)
        return action, {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np)[0]

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=False,
    ):
        log_probs, pre_act = super().forward(obs, return_preactivations=True)

        if deterministic:
            log_prob, idx = torch.max(log_probs, 1)
            return (idx, None)
        else:
            # Using Gumbel-Max trick to sample from the multinomials
            u = torch.rand(pre_act.size(), requires_grad=False)
            gumbel = -torch.log(-torch.log(u))
            _, idx = torch.max(gumbel + pre_act, 1)

            idx = torch.unsqueeze(idx, 1)
            log_prob = torch.gather(log_probs, 1, idx)
            return (idx, log_prob)
    
    def get_log_pis(self, obs):
        return super().forward(obs)


class MlpPolicy(Mlp, ExplorationPolicy):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        init_w=1e-3,
        **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
    
    def get_action(self, obs_np, deterministic=False):
        '''
        deterministic=False makes no diff, just doing this for
        consistency in interface for now
        '''
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        actions = actions[None]
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np)[0]


class ReparamTanhMultivariateGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = ReparamTanhMultivariateGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
    This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            assert LOG_SIG_MIN <= np.log(std) <= LOG_SIG_MAX
            std = std * np.ones((1, action_dim))
            self.log_std = ptu.from_numpy(np.log(std), requires_grad=False)

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def get_mean(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        return mean

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=False,
            return_tanh_normal=False
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
            if self.dropout is not None:
                h = self.dropouts[i](h)
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
            if return_log_prob:
                action, pre_tanh_value = tanh_normal.sample(
                    return_pretanh_value=True
                )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
            else:
                action = tanh_normal.sample()

        if return_tanh_normal:
            return (
                action, mean, log_std, log_prob, expected_log_prob, std,
                mean_action_log_prob, pre_tanh_value, tanh_normal,
            )
        return (
            action, mean, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob, pre_tanh_value,
        )

    def get_log_prob(self, obs, acts, return_normal_params=False, pre_tanh_value=None, squash=True):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
        log_prob = tanh_normal.log_prob(acts, pre_tanh_value, squash=squash)

        if return_normal_params:
            return log_prob, mean, log_std
        return log_prob


class ReparamMultivariateGaussianPolicy(Mlp, ExplorationPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            assert LOG_SIG_MIN <= np.log(std) <= LOG_SIG_MAX
            std = std*np.ones((1,action_dim))
            self.log_std = ptu.from_numpy(np.log(std), requires_grad=False)

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=False,
            return_normal=False
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        mean = torch.clamp(mean, min=-1.0, max=1.0)

        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, np.log(0.2))
            std = torch.exp(log_std)

        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        if deterministic:
            action = mean
        else:
            normal = ReparamMultivariateNormalDiag(mean, log_std)
            action = normal.sample()
            if return_log_prob:
                log_prob = normal.log_prob(action)

        if return_normal:
            return (
                action, mean, log_std, log_prob, expected_log_prob, std,
                mean_action_log_prob, normal
            )
        return (
            action, mean, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob
        )

    def get_log_prob(self, obs, acts, return_normal_params=False):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std
        
        normal = ReparamMultivariateNormalDiag(mean, log_std)
        log_prob = normal.log_prob(acts)

        if return_normal_params:
            return log_prob, mean, log_std
        return log_prob
