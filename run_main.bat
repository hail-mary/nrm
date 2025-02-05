@echo off

REM Execute the Python script with the specified arguments
python main.py --logdir nrm_ant --env Ant-v5
python main.py --logdir nrm_cheetah --env HalfCheetah-v5
python main.py --logdir nrm_hopper --env Hopper-v5
python main.py --logdir nrm_pusher --env Pusher-v5
python main.py --logdir nrm_reacher --env Reacher-v5
python main.py --logdir nrm_swimmer --env Swimmer-v5
python main.py --logdir nrm_walker --env Walker2d-v5

REM Print a message
echo Script execution completed.