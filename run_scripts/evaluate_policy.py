import yaml
import argparse
import joblib
import numpy as np
import os,sys,inspect
import pickle, random

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
print(sys.path)

from gym.spaces import Dict
from rlkit.envs import get_env

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.core import eval_util

from rlkit.envs.wrappers import ScaledEnv
from rlkit.samplers import PathSampler
from rlkit.torch.sac.policies import MakeDeterministic

from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy

from video import save_video


def experiment(variant):
    env_specs = variant['env_specs']
    env = get_env(env_specs)
    env.seed(env_specs['eval_env_seed'])

    print('\n\nEnv: {}'.format(env_specs['env_name']))
    print('kwargs: {}'.format(env_specs['env_kwargs']))
    print('Obs Space: {}'.format(env.observation_space))
    print('Act Space: {}\n\n'.format(env.action_space))

    obs_space = env.observation_space
    act_space = env.action_space
    
    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    
    if variant['scale_env_with_demo_stats']:
        with open('expert_demos_listing.yaml', 'r') as f:
            listings = yaml.load(f.read())
        expert_demos_path = listings[variant['expert_name']]['file_paths'][variant['expert_idx']]
        buffer_save_dict = joblib.load(expert_demos_path)
        env = ScaledEnv(
            env,
            obs_mean=buffer_save_dict['obs_mean'],
            obs_std=buffer_save_dict['obs_std'],
            acts_mean=buffer_save_dict['acts_mean'],
            acts_std=buffer_save_dict['acts_std'],
        )

    net_size = variant['policy_net_size']
    num_hidden = variant['policy_num_hidden_layers']

    policy = joblib.load(variant['policy_checkpoint'])['policy']

    if variant['offline_params']['eval_deterministic']:
        policy = MakeDeterministic(policy)
    policy.to(ptu.device)

    eval_sampler = PathSampler(
        env,
        policy,
        1,
        variant['offline_params']['max_path_length'],
        no_terminal=variant['offline_params']['no_terminal'],
        render=True,
        render_mode=variant['render_mode']
    )
    test_paths = eval_sampler.obtain_samples()
    average_returns = eval_util.get_average_returns(test_paths)
    print(average_returns)

    if variant['video_path'] and variant['render_mode'] == 'rgb_array':
        video_path = variant['video_path']
        video_path = os.path.join(video_path, variant['env_specs']['env_name'])

        print('saving videos...')
        for i, test_path in enumerate(test_paths):
            images = np.stack(test_path['image'], axis=0)
            fps = 1 // getattr(env, 'dt', 1 / 30)
            video_save_path = os.path.join(video_path, 'episode_{%d}.mp4' % i)
            save_video(images, video_save_path, fps=fps)

    return 1


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    parser.add_argument('-g', '--gpu', help='gpu id', type=int, default=0)

    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)

    # print(exp_specs)
    if exp_specs['num_gpu_per_worker'] > 0:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True)
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    # setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
