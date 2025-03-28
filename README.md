# Simultaneous Optimization of Discrete and Continuous Parameters Defining a Robot Controller

## Installation
We recommend using Miniconda to create your virtual env.
```
conda create -n nrm_env python=3.11
conda activate nrm_env
git clone https://github.com/hail-mary/nrm.git
cd nrm
pip install gymnasium[mujoco] stable-baselines3 pyyaml
```

## Training
```
python main.py
# specify the log directory by adding `--logdir` flag.
python main.py --logdir logs
```

## Evaluation
```
python main.py --eval [PATH_TO_CHECKPOINTS]
# optional: recording, requires pip install "gymnasium[other]"
python main.py --eval [PATH_TO_CHECKPOINTS] --record
```

## Plot results
```
python main.py --plot .\logs\history.json
```

