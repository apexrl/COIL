import yaml
import argparse
import joblib
import numpy as np
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
print(sys.path)

from gym.spaces import Dict
from rlkit.envs import get_env

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed, logger
from rlkit.core import eval_util
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy

from rlkit.envs.wrappers import ScaledEnv
from rlkit.samplers import PathSampler
from rlkit.torch.sac.policies import MakeDeterministic


def experiment(variant, seed):
    # with open('expert_demos_listing.yaml', 'r') as f:
    #     listings = yaml.load(f.read())ssssss
    # expert_demos_path = listings[variant['expert_name']]['file_paths'][variant['expert_idx']]
    # buffer_save_dict = joblib.load(expert_demos_path)
    # expert_replay_buffer = buffer_save_dict['train']

    # if 'minmax_env_with_demo_stats' in variant.keys():
    #     if variant['minmax_env_with_demo_stats']:
    #         print('Use minmax envs')
    #         assert 'norm_train' in buffer_save_dict.keys()
    #         expert_replay_buffer = buffer_save_dict['norm_train']

    env_specs = variant['env_specs']
    env = get_env(env_specs)
    env.seed(seed)
    env.reset()

    obs_space = env.observation_space
    act_space = env.action_space
    assert not isinstance(obs_space, Dict)
    assert len(obs_space.shape) == 1
    assert len(act_space.shape) == 1
    
    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    print('\n\nEnv: {}'.format(env_specs['env_name']))
    print('kwargs: {}'.format(env_specs['env_kwargs']))
    print('Obs Space: {}'.format(env.observation_space))
    print('Act Space: {}\n\n'.format(env.action_space))
    
    # if variant['scale_env_with_demo_stats']:
    #     env = ScaledEnv(
    #         env,
    #         obs_mean=buffer_save_dict['obs_mean'],
    #         obs_std=buffer_save_dict['obs_std'],
    #         acts_mean=buffer_save_dict['acts_mean'],
    #         acts_std=buffer_save_dict['acts_std'],
    #     )
    #
    # elif variant['minmax_env_with_demo_stats']:
    #     env = MinmaxEnv(
    #         env,
    #         obs_min=buffer_save_dict['obs_min'],
    #         obs_max=buffer_save_dict['obs_max'],
    #     )

    if variant['test_random']:
        net_size = 256
        num_hidden = 2
        policy = ReparamTanhMultivariateGaussianPolicy(
            hidden_sizes=num_hidden * [net_size],
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        if variant['eval_deterministic']:
            policy = MakeDeterministic(policy)
        policy.to(ptu.device)

        eval_sampler = PathSampler(
            env,
            policy,
            variant['num_eval_steps'],
            variant['max_path_length'],
            no_terminal=variant['no_terminal'],
            render=variant['render'],
            render_kwargs=variant['render_kwargs']
        )
        test_paths = eval_sampler.obtain_samples()
        average_returns, average_stds = eval_util.get_average_returns(test_paths, True)
        logger.log('random mean: {}'.format(average_returns))
        logger.log('random std: {}'.format(average_stds))

    policy_checkpoint = variant['policy_checkpoint']
    print('Policy Checkpoint: %s' % policy_checkpoint)
    dirs = [_ for _ in os.listdir(policy_checkpoint) if os.path.isdir(os.path.join(policy_checkpoint, _))]
    test_paths = []
    for policy_name in variant['policy_name']:
        for dir_name in dirs:
            policy_path = os.path.join(policy_checkpoint, dir_name, '%s.pkl' % policy_name)
            print("Loading from %s..." % policy_path)
            try:
                policy = joblib.load(policy_path)['exploration_policy']
            except IOError:
                print("Failed.")
                continue
            if variant['eval_deterministic']:
                policy = MakeDeterministic(policy)
            policy.to(ptu.device)

            print("Sampling...")
            eval_sampler = PathSampler(
                env,
                policy,
                variant['num_eval_steps'],
                variant['max_path_length'],
                no_terminal=variant['no_terminal'],
                render=variant['render'],
                render_kwargs=variant['render_kwargs']
            )
            test_paths += eval_sampler.obtain_samples()

    return test_paths


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    parser.add_argument('-g', '--gpu', help='gpu id', type=str, default=0)
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)

    exp_specs['env_specs']['eval_env_seed'] = exp_specs['env_specs']['training_env_seed']
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if exp_specs['num_gpu_per_worker'] > 0:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True)
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = 0
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    paths = []
    for seed in exp_specs['seed']:
        logger.log("\n\ntest on seed %d..." % seed)
        set_seed(seed)
        paths += experiment(exp_specs, seed)
        logger.log("Num paths: %d" % len(paths))
    average_returns, average_stds = eval_util.get_average_returns(paths, True)
    logger.log('test mean: {}'.format(average_returns))
    logger.log('test std: {}'.format(average_stds))
