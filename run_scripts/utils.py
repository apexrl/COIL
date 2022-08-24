import d4rl
import gym
import os
import pickle
import pandas as pd
import numpy as np


def process_d4rl_dataset(env_name, save_file, gamma=0.99):
    save_dir = os.path.join(*save_file.split('/')[:-1])
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            pass
    save_file = os.path.join(save_dir, env_name + '.pkl')
    if os.path.exists(save_file):
        print(f'Dataset file {save_file} exists')
        return
    
    # Create the environment
    env = gym.make(env_name)
    # d4rl abides by the OpenAI gym interface
    env.reset()

    # Each task is associated with a dataset
    # dataset contains observations, actions, rewards, terminals, and infos
    data_dict = env.get_dataset()
    print("data length:", len(data_dict['rewards']))
    df = pd.DataFrame({'rewards': data_dict['rewards'], 'terminals': data_dict['terminals'], 'timeouts': data_dict['timeouts']})
    indexes = df[df['terminals'] | df['timeouts']].index
    ep_len = [indexes[i] - indexes[i - 1] for i in range(1, len(indexes))]
    print("max ep length:", max(ep_len))

    num_trajs = len(indexes)
    trajectories = []
    start = 0
    for traj_ind in range(num_trajs):
        end = indexes[traj_ind] + 1
        traj_dict = {}
        for key in data_dict.keys():
            traj_dict[key] = data_dict[key][start:end-1]
        traj_dict['next_observations'] = data_dict['observations'][start + 1: end]
        gamma_array = np.logspace(0, len(traj_dict['rewards']) - 1, len(traj_dict['rewards']), base=gamma)
        traj_dict['ep_discount_rews'] = np.sum(np.array(traj_dict['rewards']) * gamma_array)
        traj_dict['ep_rews'] = np.sum(np.array(traj_dict['rewards']))
        traj_dict['agent_infos'] = [None for _ in range(len(traj_dict['next_observations']))]
        traj_dict['env_infos'] = [None for _ in range(len(traj_dict['next_observations']))]
        trajectories.append(traj_dict)
        start = end

    trajectories.sort(key=lambda traj: traj['ep_rews'])

    with open(save_file, 'wb') as f:
        pickle.dump(trajectories, f)
