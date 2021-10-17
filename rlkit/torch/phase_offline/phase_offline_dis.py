import numpy as np
from collections import OrderedDict
import torch
import gtimer as gt
import traceback
import os
import csv

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.core import eval_util
from rlkit.torch.torch_base_algorithm import TorchBaseAlgorithm
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers import PathSampler
from rlkit.core import logger


class PhaseOffline(TorchBaseAlgorithm):
    """
        Phase Offline Reinforcement Learning.
    """
    def __init__(
        self,

        mode,
        policy_trainer,

        replay_buffer,
        random_reward=100,
        ratio=0.7,
        pretrain_times=10,

        clamp_magnitude=1.0,

        policy_optim_batch_size=1024,
        policy_optim_batch_size_from_expert=0,

        num_policy_update_loops_per_train_call=100,
        num_bc_update_loops_per_train_call=100,
        num_policy_updates_per_loop_iter=1,
        bc_dist=False,
        use_seq_training=True,
        num_seq_training=5,
        num_par_training=int(3e4),

        use_grad_pen=True,
        grad_pen_weight=10,

        filter_type=None,
        filter_tau=None,

        **kwargs
    ):
        assert mode in ['reward', 'occupancy'], 'Invalid  algorithm!'
        if kwargs['wrap_absorbing']:
            raise NotImplementedError()
        super().__init__(**kwargs)

        self.mode = mode
        self.ratio = self.epsilon = ratio
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
        self.num_bc_update_loops_per_train_call = num_bc_update_loops_per_train_call
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter
        self.bc_dist = bc_dist
        self.use_seq_training = use_seq_training
        self.num_seq_training = num_seq_training
        self.num_par_training = num_par_training
        self.init_flag = False
        self.parallel_count = 0.0
        self.replaced = False
        self.bc_dist_count = 0
        self.replace_count = 0
        if self.bc_dist:
            assert self.use_seq_training, "BC distillation must use sequence training!"

        self.trained = False
        self.last_is_bc = 0

        self.total_train_steps = 0
        self.used_traj_orders = []
        self.used_traj_rews = []
        self.bc_per_iter = 0
        self.min_traj_reward = []
        self.filter_type = filter_type
        if self.filter_type == "running_mean":
            self.running_mean = 0
            self.filter_tau = filter_tau if filter_tau else 0.99

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        # for i in range(self.pretrain_times):
        #     self._do_training(-1, pretrain=True) # Actually not pretrained
        self._do_training(-1, pretrain=True)
        self.trained = True

    def start_training(self, start_epoch=0):
        observation = self._start_new_rollout()
        out_of_data = False

        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            try:
                self.replaced = False
                self._start_epoch(epoch)
                for steps_this_epoch in range(self.num_env_steps_per_epoch):
                    self._n_env_steps_total += 1
                    if self._n_env_steps_total % self.num_steps_between_train_calls == 0:
                        gt.stamp('sample')
                        self._try_to_train(epoch)
                        gt.stamp('train')
                # if 360 <= epoch <= 420 and epoch % 10 == 0:
                #     self._special_save(epoch)
            except ValueError:
                out_of_data = True
                break
            finally:
                gt.stamp('sample')
                self._try_to_eval(epoch, out_of_data=out_of_data)
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
        
        mean_expected_rew = self.policy_trainer.target_vf(torch.FloatTensor(initial_obs).to(ptu.device)).detach().cpu()
        # - self.policy_trainer.target_vf(torch.FloatTensor(last_obs).to(ptu.device)).detach().cpu()
        mean_expected_rew = np.array(mean_expected_rew).reshape(-1)
        
        # initial_rews = np.array(initial_rews)
    
        # mean_expected_rew = mean_expected_rew

        return mean_expected_rew  # np.mean(mean_expected_rew)
    
    def get_batch(self, batch_size, keys=None):
        batch = self.pair_replay_buffer.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)

        return batch

    def _end_epoch(self):
        if self.filter_type and len(self.min_traj_reward) > 0:
            if self.filter_type == "mean":
                self.replay_buffer.filter_below(np.nanmean(self.min_traj_reward))
            elif self.filter_type == "running_mean":
                nan_min = np.nanmin(self.min_traj_reward)
                if not np.isnan(nan_min):
                    self.running_mean = self.filter_tau * self.running_mean \
                                        + (1 - self.filter_tau) * nan_min
                    self.replay_buffer.filter_below(self.running_mean)
                self.min_traj_reward = []
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

    def _do_training(self, epoch, pretrain=False):
        try:
            trajs = None
            if not pretrain:
                (self.pair_replay_buffer, trajs), training_mode, traj_steps, traj_order, traj_rews, min_reward = \
                    self.replay_buffer.get_sufficient_buffer(self.epsilon, self.policy_trainer.policy,
                                                             return_trajs=True)
                self.total_train_steps += traj_steps
                self.used_traj_orders.append(traj_order)
                self.used_traj_rews.append(traj_rews)
                self.bc_per_iter += int(training_mode == "bc")
                self.min_traj_reward.append(min_reward)
            else:
                self.pair_replay_buffer = self.replay_buffer.get_pretrain_buffer(self.random_reward)
                if self.pair_replay_buffer.get_traj_num() == 0:
                    return
                training_mode = "pretrain"
                # self.pair_replay_buffer = self.replay_buffer.get_pretrain_buffer(self.random_reward)
            num_loops_per_train_call = self.num_policy_update_loops_per_train_call if training_mode == "rl" \
                else (self.num_bc_update_loops_per_train_call if training_mode == "bc" else self.pretrain_times)
            for _ in range(num_loops_per_train_call):
                if training_mode == "rl":
                    if not self.init_flag and self.parallel_count == self.num_par_training:
                        self.init_flag = True
                        self.parallel_count = 0.0
                    if self.parallel_count == self.num_par_training and not self.bc_dist:
                        print("============= REPLACE! =============")
                        logger.log("============= REPLACE! =============")
                        replace_kwargs = {}
                        if self.use_seq_training:
                            replace_kwargs["num_seq_training"] = self.num_seq_training
                            replace_kwargs["buffer"] = self.pair_replay_buffer
                        self.policy_trainer.replace(self.use_seq_training, **replace_kwargs)
                        self.replace()
                        self.parallel_count = 0.0
                    anneal = self.parallel_count / self.num_par_training if (self.init_flag
                                                                             and not self.use_seq_training
                                                                             and not self.bc_dist) else -1
                    self._do_policy_training(epoch, pretrain, training_mode="rl", extra_upd=self.last_is_bc,
                                             anneal=anneal)
                    self.parallel_count += 1
                    self.bc_dist_count = 0
                elif training_mode == 'bc' and self.bc_dist and self.init_flag:
                    sp_buffer = self.replay_buffer.get_simple_buffer()
                    replace_flag = False
                    dis_path = os.path.join(logger._snapshot_dir, "distillation-%d.csv" % self.replace_count)
                    with open(dis_path, "w") as csvfile:
                        writer = csv.DictWriter(csvfile, ["iter", "max log ratio"])
                        writer.writeheader()
                        while self.bc_dist_count <= 500:
                            self.bc_dist_count += 1
                            replace_flag, _, max_log_ratio = self.policy_trainer.can_replace(self.replay_buffer, trajs)
                            writer.writerow({"iter": self.bc_dist_count, "max log ratio": max_log_ratio})
                            if self.bc_dist_count >= 60:
                                replace_flag = False

                            if replace_flag:
                                print("============= BC REPLACE! =============")
                                logger.log("============= BC REPLACE! =============")
                            replace_kwargs = {"num_seq_training": self.num_seq_training,
                                              "buffer": self.pair_replay_buffer,
                                              "sp_buffer": sp_buffer,
                                              "replace_flag": replace_flag
                                              }
                            self.policy_trainer.replace(self.use_seq_training, **replace_kwargs)
                            if replace_flag:
                                print("BREAK at distillation iter %d" % self.bc_dist_count)
                                self.replace()
                                break
                    self.bc_dist_count = 0
                    self.replace_count += 1
                    if replace_flag:
                        break
                    else:
                        self.policy_trainer.init_student()
                        self._do_policy_training(epoch, pretrain, training_mode="bc", extra_upd=self.last_is_bc)
                else:
                    self._do_policy_training(epoch, pretrain, training_mode=training_mode, extra_upd=self.last_is_bc)
        except Exception as e:
            traceback.print_exc()
            raise e

        if not pretrain:
            self.last_is_bc = (self.last_is_bc + 1) if training_mode == "bc" else 0

    def _do_policy_training(self, epoch, pretrain=False, **kwargs):
        policy_batch = self.get_batch(self.policy_optim_batch_size)
        # policy optimization step
        self.policy_trainer.train_step(policy_batch, **kwargs)

    def replace(self):
        self.eval_policy = self.exploration_policy = self.policy_trainer.policy
        self.eval_sampler = PathSampler(
            self.env,
            self.eval_policy,
            self.num_steps_per_eval,
            self.max_path_length,
            no_terminal=False,
            render=self.render,
            render_kwargs=self.render_kwargs
        )
        self.replaced = True

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

    def get_epoch_snapshot(self, epoch, save_all=False):
        snapshot = super().get_epoch_snapshot(epoch, save_all)
        snapshot.update(self.policy_trainer.get_snapshot())
        snapshot["filter"] = self.running_mean
        snapshot["filter_tau"] = self.filter_tau
        return snapshot

    def to(self, device):
        super().to(device)

    def evaluate(self, epoch):
        """
        Evaluate the policy, e.g. save/print progress.
        :param epoch:
        :return:
        """
        self.eval_statistics = OrderedDict()
        if self.policy_trainer.get_eval_statistics() is not None:
            self.eval_statistics.update(self.policy_trainer.get_eval_statistics())
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

        statistics['Total Training Steps'] = self.total_train_steps
        statistics['Used Traj Orders'] = self.used_traj_orders
        self.used_traj_orders = []
        statistics['Used Traj Rewards'] = self.used_traj_rews
        self.used_traj_rews = []
        statistics['BC per Iteration'] = self.bc_per_iter
        self.bc_per_iter = 0
        if self.filter_type == "running_mean":
            statistics['Filter'] = self.running_mean
        statistics['Replace'] = self.replaced

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
