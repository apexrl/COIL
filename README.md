# README

Implementation for NeurIPS 2021 paper "Curriculum Offline Imitation Learning". 

Poster:
![poster](https://github.com/apexrl/COIL/blob/master/NIPS-Poster-COIL.png)

The code is based on the [ILswiss](https://github.com/Ericonaldo/ILSwiss).

To run the code, use

`python run_experiment.py -e <your YAML file> -g <gpu id>`

An example yaml file is shown in `specs/`

Generally,  `run_experiment.py` loads the YAML file, creating multiple processes, each of which runs the script assigned in the YAML file. 

The script of COIL is `run_scripts/coil_script.py`. Dataset settings are in `demos_listing.yaml`. The core algorithm is in `rlkit/torch/phase_offline/phase_offline_coil.py` and `rlkit/torch/coil/coil.py`. New algorithms should also be put under similar directories. A trajectory replay buffer and the trajectory picking algorithm is in `rlkit/data_management/episodic_replay_buffer_coil.py`.

For training datasets, put them under the path as you determine in `demos_listing.yaml`. Specifically, D4RL datasets will be automatically downloaded and processed, then put under the determined path.

The environment list is in `rlkit/envs/envs_dict.py`. You can add customized environments by modifying this file. If the environment name in your YAML file is not in `envs_dict`, the program will invoke `gym.make` to build the environment.

---

## Update for AntMaze and spare-reward tasks

We now support [AntMaze](https://github.com/Farama-Foundation/D4RL/wiki/Tasks#antmaze) tasks, and corresponding experiment specifications are under `specs/`. For similar spare-reward tasks that only care about success or failure, one can set `offline_params: mode` in the spec file to `occupancy` instead of `reward`. Under the occupancy mode, the evaluation metric is switched from average return to success rate, which measures the ratio of successfully completing the tasks.

Our evaluation results on AntMaze tasks are listed bellow:

| Dataset | Success Rate |
|:-------:|:------------:|
|antmaze-umaze-v0| 0.61 |
|antmaze-umaze-diverse-v0| 0.58 |
|antmaze-medium-diverse-v0| 0.01 |
|antmaze-medium-play-v0| 0.00 |
|antmaze-large-diverse-v0| 0.00 |
|antmaze-large-play-v0| 0.00 |