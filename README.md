# README

The code is based on the [ILswiss](https://github.com/Ericonaldo/ILSwiss).

To run the code, use

`python run_experiment.py --srun -e <your YAML file> -g <gpu id>`.

Generally,  `run_experiment.py` loads the YAML file, creating multiple processes, each of which runs the script assigned in the YAML file. 

The script of COIL is `run_scripts/coil_script.py`. Dataset settings are in `demos_listing.yaml`. The core algorithm is in `rlkit/torch/coil/coil.py`. New algorithms should also be put under similar directories. A trajectory replay buffer and the trajectory picking algorithm is in `rlkit/data_management/episodic_replay_buffer_coil.py`.
