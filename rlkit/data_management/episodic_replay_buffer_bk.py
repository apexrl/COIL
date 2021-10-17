from collections import defaultdict
import random as python_random
from random import sample
from itertools import starmap
from functools import partial

import numpy as np
import gc

from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger

import torch

class EpisodicReplayBuffer(ReplayBuffer):
    """
    A class used to save and replay data.
    """

    def __init__(
        self,
        max_replay_buffer_size,
        max_sub_buf_size,
        observation_dim,
        action_dim,
        random_seed=1995,
        gamma = 0.99
    ):
        self._random_seed = random_seed
        self._np_rand_state = np.random.RandomState(random_seed)
        self.gamma = gamma

        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._max_sub_buf_size = max_sub_buf_size

        if isinstance(observation_dim, tuple):
            dims = [d for d in observation_dim]
            dims = [max_replay_buffer_size] + dims
            dims = tuple(dims)
            self._observations = np.zeros(dims)
            self._next_obs = np.zeros(dims)
        elif isinstance(observation_dim, dict):
            # assuming that this is a one-level dictionary
            self._observations = {}
            self._next_obs = {}

            for key, dims in observation_dim.items():
                if isinstance(dims, tuple):
                    dims = tuple([max_replay_buffer_size] + list(dims))
                else:
                    dims = (max_replay_buffer_size, dims)
                self._observations[key] = np.zeros(dims)
                self._next_obs[key] = np.zeros(dims)
        else:
            # else observation_dim is an integer
            self._observations = np.zeros((max_replay_buffer_size, observation_dim))
            self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        
        self._actions = np.zeros((max_replay_buffer_size, action_dim))

        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._data = []
        self._top = 0
        self._size = 0

        self._initial_pairs = []

        self.simple_buffer = None
        self.last_sufficient_length = 0

        self.simple_buffer = SimpleReplayBuffer(self._max_sub_buf_size, self._observation_dim, self._action_dim, self._random_seed)

    def get_size(self):
        return self._size

    def set_data(self, data, **kwargs):
        """
        Set the buffer with a list-like data.
        """
        self._data = data[:self._max_replay_buffer_size]
        self._data.sort(key=lambda traj: traj['ep_rews'])

        self._top = len(self._data) - 1
        self._size = len(self._data)
        
        self._initial_pairs = {'observations':[], 'actions':[], 'next_observations':[], 'rewards':[]}
        self._last_pairs = {'observations':[], 'actions':[], 'next_observations':[], 'rewards':[], 'gamma_pow':[]}
        for traj in self._data:
            for key in self._initial_pairs.keys():
                self._initial_pairs[key].append(traj[key][0])
                self._last_pairs[key].append(traj[key][-1])
                self._last_pairs['gamma_pow'].append(self.gamma ** len(traj['observations']))


    def add_path(self, path):
        """
        Add a path to the replay buffer.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        # path.keys() = ['actions', 'observations', 'next_observations', 'rewards', 'terminals', 'timeouts', 'ep_rews', "agent_infos", "env_infos",]
        
        if len(self._data) < self._max_replay_buffer_size:
            self._data.append(path)
            for key in self._initial_pairs.keys():
                self._initial_pairs[key].append(path[key][0])
                self._last_pairs[key].append(traj[key][-1])
                self._last_pairs['gamma_pow'].append(self.gamma ** len(traj['observations']))
        else:
            print("replace the original buffer")
            self._data[(self._top + 1) % self._max_replay_buffer_size] = path
        
        self._top += 1
        self._size = min(self._top+1, self._max_replay_buffer_size)

    def get_pretrain_buffer(self, benchmark_rew=350):
        """
        Return a replay buffer of pretrain data (random trajs).

        :param benchmark_rew: the maximum performance of a random policy
        """

        selected_trajs = [_ for _ in self._data if _['ep_discount_rews'] <= benchmark_rew]
        logger.log("Pretrained buf, Random benchmark:{}, Selected num: {}".format(benchmark_rew, len(selected_trajs)))
        
        '''
        if self.simple_buffer is not None:
            del self.simple_buffer
            gc.collect()
            self.simple_buffer = None
        self.simple_buffer = SimpleReplayBuffer(self._max_sub_buf_size, self._observation_dim, self._action_dim, self._random_seed)
        '''

        for traj in selected_trajs:
            self.simple_buffer.add_path(traj)
        
        return self.simple_buffer

    def get_sufficient_buffer(self, benchmark_rew=150, ratio=0.6, random_benchmarks=150, value_func=None):
        """
        Return a replay buffer that sufficient the reward ratio requirement.
        :param benchmark_rew: the mean performance of the current policy
        :param ratio: the reward ratio requirement
        """
        # print('benchmark_rew', benchmark_rew) 
        last_pair = self.get_last_pairs()
        last_state = torch.Tensor(last_pair['observations']).to(ptu.device)
        last_q = value_func(last_state).detach().cpu()

        last_sufficient_length = self.last_sufficient_length

        if type(benchmark_rew) == int:
            selected_trajs = [self._data[ind] for ind in range(len(self._data)) if (self._data[ind]['ep_discount_rews']+last_pair['gamma_pow'][ind]*last_q[ind]) <= benchmark_rew ]
        else:
            selected_trajs = [self._data[ind] for ind in range(len(self._data)) if ratio <= (self._data[ind]['ep_discount_rews']+last_pair['gamma_pow'][ind]*last_q[ind]) <= benchmark_rew[ind]]
        # selected_trajs = [_ for _ in self._data if _['ep_discount_rews'] <= benchmark_rew]
        
        if len(selected_trajs) <= 0:
            if benchmark_rew <= random_benchmarks:
                logger.log('Use the pretraining dataset! Current evaluation {}'.format(np.mean(benchmark_rew)))
                return self.get_pretrain_buffer(random_benchmarks)
            else:
                logger.log('Higher than the dataset! Current evaluation {}'.format(np.mean(benchmark_rew)))
                benchmark_rew = self._data[-1]['ep_rews'] * ratio
                selected_trajs = [_ for _ in self._data if ratio <= _['ep_rews']/np.mean(benchmark_rew) <= 1/ratio]
        
        
        if self.simple_buffer is not None:
            del self.simple_buffer
            gc.collect()
            self.simple_buffer = None
        self.simple_buffer = SimpleReplayBuffer(self._max_replay_buffer_size, self._observation_dim, self._action_dim, self._random_seed)

        traj_num = min(last_sufficient_length+10, len(selected_trajs))
        self.last_sufficient_length = traj_num

        raw_len = len(selected_trajs)
        selected_trajs = selected_trajs[:traj_num]
        
        logger.log("Current evaluation: {}, Sufficient num: {}, Selected num: {}, Use num: {}".format(np.mean(benchmark_rew), raw_len, traj_num, self.simple_buffer.get_traj_num()))
        for traj in selected_trajs:
            self.simple_buffer.add_path(traj)

        return self.simple_buffer

    def get_initial_pairs(self):
        """
        Return a list of initial pairs of each traj.
        """

        assert self._size > 0

        return self._initial_pairs

    def get_last_pairs(self):
        """
        Return a list of initial pairs of each traj.
        """

        assert self._size > 0

        return self._last_pairs

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        """
        Add a transition tuple.
        """
        pass

    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass
