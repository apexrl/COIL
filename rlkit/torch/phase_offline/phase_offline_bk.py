import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch import nn
from torch import autograd
import torch.nn.functional as F

import gtimer as gt

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.core import logger, eval_util
from rlkit.torch.torch_base_algorithm import TorchBaseAlgorithm
from rlkit.data_management.path_builder import PathBuilder

from rlkit.core import logger


class PhaseOffline(TorchBaseAlgorithm):
    '''
        Phase Offline Reinforcement Learning.
    '''
    def __init__(
        self,

        mode,
        policy_trainer,

        replay_buffer,
        random_reward = 100,
        ratio = 0.7,
        pretrain_times=10,

        clamp_magnitude=1.0,

        policy_optim_batch_size=1024,
        policy_optim_batch_size_from_expert=0,

        num_policy_update_loops_per_train_call=1,
        num_policy_updates_per_loop_iter=100,

        use_grad_pen=True,
        grad_pen_weight=10,

        **kwargs
    ):
        assert mode in ['reward', 'occupancy'], 'Invalid  algorithm!'
        if kwargs['wrap_absorbing']: raise NotImplementedError()
        super().__init__(**kwargs)

        self.mode = mode
        self.ratio = ratio
        self.random_reward = random_reward
        self.pretrain_times = pretrain_times

        self.replay_buffer = replay_buffer
        self.pair_replay_buffer = None

        self.policy_trainer = policy_trainer
        self.policy_optim_batch_size = policy_optim_batch_size
        self.policy_optim_batch_size_from_expert = policy_optim_batch_size_from_expert

        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight

        self.num_policy_update_loops_per_train_call = num_policy_update_loops_per_train_call
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter

        self.trained = False

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        for i in range(self.pretrain_times):
            self._do_training(-1, pretrain=True)
        self.trained = True

    def start_training(self, start_epoch=0):
        observation = self._start_new_rollout()

        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            for steps_this_epoch in range(self.num_env_steps_per_epoch):
                """
                action, agent_info = self._get_action_and_info(observation)
                if self.render: self.training_env.render()

                next_ob, raw_reward, terminal, env_info = (
                    self.training_env.step(action)
                )
                if self.no_terminal: terminal = False
                """
                self._n_env_steps_total += 1

                """
                reward = np.array([raw_reward])
                terminal = np.array([terminal])
                self._handle_step(
                    observation,
                    action,
                    reward,
                    next_ob,
                    np.array([False]) if self.no_terminal else terminal,
                    absorbing=np.array([0., 0.]),
                    agent_info=agent_info,
                    env_info=env_info,
                )
                if terminal[0]:
                    if self.wrap_absorbing:
                        raise NotImplementedError()
                        '''
                        If we wrap absorbing states, two additional
                        transitions must be added: (s_T, s_abs) and
                        (s_abs, s_abs). In Disc Actor Critic paper
                        they make s_abs be a vector of 0s with last
                        dim set to 1. Here we are going to add the following:
                        ([next_ob,0], random_action, [next_ob, 1]) and
                        ([next_ob,1], random_action, [next_ob, 1])
                        This way we can handle varying types of terminal states.
                        '''
                        # next_ob is the absorbing state
                        # for now just taking the previous action
                        self._handle_step(
                            next_ob,
                            action,
                            # env.action_space.sample(),
                            # the reward doesn't matter
                            reward,
                            next_ob,
                            np.array([False]),
                            absorbing=np.array([0.0, 1.0]),
                            agent_info=agent_info,
                            env_info=env_info
                        )
                        self._handle_step(
                            next_ob,
                            action,
                            # env.action_space.sample(),
                            # the reward doesn't matter
                            reward,
                            next_ob,
                            np.array([False]),
                            absorbing=np.array([1.0, 1.0]),
                            agent_info=agent_info,
                            env_info=env_info
                        )
                    self._handle_rollout_ending()
                    observation = self._start_new_rollout()
                elif len(self._current_path_builder) >= self.max_path_length:
                    self._handle_rollout_ending()
                    observation = self._start_new_rollout()
                else:
                    observation = next_ob
                """

                if self._n_env_steps_total % self.num_steps_between_train_calls == 0:
                    gt.stamp('sample')
                    self._try_to_train(epoch)
                    gt.stamp('train')

            gt.stamp('sample')
            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch()
        
    def _can_train(self):
        return self.replay_buffer.get_size() > 0

    def _try_to_train(self, epoch):
        if self._can_train():
            self.training_mode(True)
            self._do_training(epoch)
            self._n_train_steps_total += 1
            self.training_mode(False)
        pass

    def evaluate_current_policy(self):
        
        initial_pairs = self.replay_buffer.get_initial_pairs()
        initial_rews = initial_pairs['rewards']
        initial_obs = initial_pairs['observations']
        initial_next_obs = initial_pairs['next_observations']

        last_pairs = self.replay_buffer.get_last_pairs()
        last_obs = last_pairs['observations']
        gamma_pow = last_pairs['gamma_pow']
        
        mean_expected_rew = self.policy_trainer.target_vf(torch.FloatTensor(initial_obs).to(ptu.device)).detach().cpu() # - self.policy_trainer.target_vf(torch.FloatTensor(last_obs).to(ptu.device)).detach().cpu()
        mean_expected_rew =  np.array(mean_expected_rew).reshape(-1)
        
        # initial_rews = np.array(initial_rews)
    
        # mean_expected_rew = mean_expected_rew

        return mean_expected_rew # np.mean(mean_expected_rew)
    
    def get_batch(self, batch_size, keys=None):
        batch = self.pair_replay_buffer.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)

        return batch

    def _end_epoch(self):
        self.policy_trainer.end_epoch()
        super()._end_epoch()

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        return self.trained

    def evaluate(self, epoch):
        self.eval_statistics = OrderedDict()
        self.eval_statistics.update(self.policy_trainer.get_eval_statistics())
        evaluate(epoch)


    def _do_training(self, epoch, pretrain=False):
        if not pretrain:
            mean_expected_rew = self.evaluate_current_policy()
            self.pair_replay_buffer = self.replay_buffer.get_sufficient_buffer(mean_expected_rew, self.ratio, self.random_reward, self.policy_trainer.target_vf)
        else:
            self.pair_replay_buffer = self.replay_buffer.get_pretrain_buffer(self.random_reward)
        for t in range(self.num_policy_update_loops_per_train_call):
            for _ in range(self.num_policy_updates_per_loop_iter):
                self._do_policy_training(epoch, pretrain)

    def _do_policy_training(self, epoch, pretrain=False):
        policy_batch = self.get_batch(self.policy_optim_batch_size)
        
        # policy optimization step
        self.policy_trainer.train_step(policy_batch)

    def _handle_step(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminal,
        absorbing,
        agent_info,
        env_info,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            absorbing=absorbing,
            agent_infos=agent_info,
            env_infos=env_info,
        )

    def _handle_rollout_ending(self):
        """
        Implement anything that needs to happen after every rollout.
        """
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            self._exploration_paths.append(
                self._current_path_builder
            )
            self._current_path_builder = PathBuilder()
    
    
    @property
    def networks(self):
        return [self.exploration_policy] + self.policy_trainer.networks


    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(self.policy_trainer.get_snapshot())
        return snapshot


    def to(self, device):
        super().to(device)

    def evaluate(self, epoch):
        """
        Evaluate the policy, e.g. save/print progress.
        :param epoch:
        :return:
        """
        statistics = OrderedDict()
        try:
            statistics.update(self.eval_statistics)
            self.eval_statistics = None
        except:
            print('No Stats to Eval')

        logger.log("Collecting samples for evaluation")
        test_paths = self.eval_sampler.obtain_samples()

        statistics.update(eval_util.get_generic_path_information(
            test_paths, stat_prefix="Test",
        ))

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths)
        if hasattr(self.env, "log_statistics"):
            statistics.update(self.env.log_statistics(test_paths))
        if epoch % self.freq_log_visuals == 0:
            if hasattr(self.env, "log_visuals"):
                self.env.log_visuals(test_paths, epoch, logger.get_snapshot_dir())
        
        average_returns = eval_util.get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns
        for key, value in statistics.items():
            logger.record_tabular(key, value)
        
        best_statistic = statistics[self.best_key]
        if best_statistic > self.best_statistic_so_far:
            self.best_statistic_so_far = best_statistic
            if self.save_best and epoch >= self.save_best_starting_from_epoch:
                data_to_save = {
                    'epoch': epoch,
                    'statistics': statistics
                }
                data_to_save.update(self.get_epoch_snapshot(epoch))
                logger.save_extra_data(data_to_save, 'best.pkl')
                print('\n\nSAVED BEST\n\n')