import yaml
import argparse
import numpy as np
from multiprocessing import Process, Queue
import time

import torch
from nrm.model import Model
from nrm.trainer import Trainer
from nrm.logger import Logger


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def eval_only(cfg, load_from, record_video=False):
    # Load the model
    model = Model(cfg)
    model.load_policy(load_from)  # Assuming you have a method to load the trained policy

    # Evaluate the policy
    print('\n#------------------- Start Evaluation ! ---------------------#')
    total_reward = model.evaluate_policy(record_video=record_video)
    print(model.policy_kwargs["net_arch"])
    print(f"Evaluation Reward: {total_reward}")

def follower_process(result_queue, network_queue, worker_id, cfg):
    """
    Each worker process learns and sends evaluation results to the leader.
    """
    # Create variable architecture model
    model = Model(cfg)
    env = model.env

    for iteration in range(cfg['num_iterations']):
        # Update policy params
        model.learn(total_timesteps=cfg['timesteps_per_iteration'])

        # Policy evaluation
        total_reward = model.evaluate_policy(seed=worker_id)
        hidden_logs = model.hidden_logs.copy()
        model.clear_hidden_logs()

        # Send current policy params and architecture to the leader
        result_queue.put((worker_id, total_reward, model.policy_kwargs, model.policy.state_dict(),
                          hidden_logs))

        # Receive new policy params and architecture from the leader
        new_arch, new_params = network_queue.get()

        if model.policy_kwargs['net_arch'] != new_arch:  # Net architecture has been changed
            new_kwargs = model.policy_kwargs.copy()
            new_kwargs['net_arch'] = new_arch
            model.make_policy(env, new_kwargs, new_params)
        else:
            model.make_policy(env, model.policy_kwargs, new_params)

def leader_process(result_queue, network_queue, cfg):
    """
    The leader collects evaluation results and modify architecture.
    """
    num_workers = cfg['num_workers']
    num_iterations = cfg['num_iterations']
    trainer = Trainer(cfg)
    logger = Logger(cfg)

    all_rewards = []
    rewards_per_worker = [[] for _ in range(num_workers)]  # Record rewards for each worker
    neuron_counts = []  # Track neuron count changes
    iteration_list = []
    best_policy_arch = None
    best_reward = float('-inf')

    prev_updated_iteration = 0  # Track the iteration when the architecture was last modified
    smoothed_mean_reward = float('-inf')  # Initialize the smoothed mean reward
    ema_window = cfg["ema_window"]
    alpha = 2 / (ema_window + 1)  # Calculate the smoothing factor based on ema_window
    smoothed_rewards_history = []  # To store smoothed mean rewards for slope calculation

    print('\n#---------------------- Start Training ! -----------------------#')
    # Start timing the training
    start_time = time.time()
    for iteration in range(num_iterations):
        terminate = iteration == num_iterations - 1
        
        rewards = []
        params = []
        policy_arch = None
        iteration_list.append(iteration)
        results = []
        logger.step() 
        hidden_logs_list = []

        # Collect results from each worker
        for _ in range(num_workers):
            worker_id, reward, policy_kwargs, policy_weight, hidden_logs = result_queue.get()
            rewards.append(reward)
            rewards_per_worker[worker_id].append(reward)
            params.append(policy_weight)
            policy_arch = policy_kwargs["net_arch"]
            results.append((worker_id, reward, policy_kwargs, policy_weight))

            hidden_logs_list.append(list(hidden_logs.values()))
        

        avg_hidden_logs = [None] * 2
        layer1_hidden_logs = torch.stack([hidden_logs_list[i][0] for i in range(num_workers)])
        layer2_hidden_logs = torch.stack([hidden_logs_list[i][1] for i in range(num_workers)])
      
        avg_hidden_logs[0] = torch.mean(layer1_hidden_logs, dim=0)
        avg_hidden_logs[1] = torch.mean(layer2_hidden_logs, dim=0)
        # print(avg_hidden_logs[0].shape, avg_hidden_logs[1].shape) -> num_samples x num_units

        # Log statistics
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        all_rewards.append(mean_reward)

        # Update the smoothed mean reward using exponential moving average
        if iteration == 0:
            smoothed_mean_reward = mean_reward
        else:
            smoothed_mean_reward = alpha * mean_reward + (1 - alpha) * smoothed_mean_reward
        smoothed_rewards_history.append(smoothed_mean_reward)

        # Calculate the slope of the smoothed mean reward using np.polyfit
        if (iteration - prev_updated_iteration) >= ema_window:
            x = np.arange(ema_window)
            y = smoothed_rewards_history[-ema_window:]
            slope, _ = np.polyfit(x, y, 1)
        else:
            slope = None

        # Track the best policy architecture
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_policy_arch = policy_arch.copy()
            best_iteration = iteration
            should_save = True

        if should_save or terminate:
            logger.save_checkpoint(results)
            should_save = False

        # Log the iteration data
        total_neurons = sum(sum(units) for units in policy_arch.values())
        logger.log_iteration(mean_reward, std_reward, total_neurons)

        # Output: statistics
        if slope is not None:
            print(f"\nIteration {iteration}: Reward Mean = {mean_reward:.2f}, Std = {std_reward:.2f}, slope = {slope:.2f}")
        else:
            print(f"\nIteration {iteration}: Reward Mean = {mean_reward:.2f}, Std = {std_reward:.2f}, slope = N/A")

        # Output: architecture information
        print("Current Policy Architecture:")
        for layer, shape in policy_arch.items():
            print(f"  - {layer}: {shape}")

        #########################################
        rankings = trainer.yield_rankings(avg_hidden_logs)
        print(f"Current Rankings: \n {rankings}")
        
        avg_params = trainer.average_params(params)
        if iteration > 0 and slope is not None and slope < cfg['slope_threshold']:
            print(f"\n----------- Iteration {iteration}: Modifying network architecture ----------")
            arch, params, aux = trainer.preprocess(raw_arch=policy_arch, raw_params=avg_params)
            modified_arch, modified_params = trainer.modify_network(params, arch, rankings)
            modified_arch, modified_params = trainer.reconstruct(modified_arch, modified_params, aux)

            prev_updated_iteration = iteration
            trainer.clear_rankings()

        else:
            modified_arch = policy_arch
            modified_params = avg_params

        # Record neuron count
        total_neurons = sum(sum(units) for units in modified_arch.values())
        neuron_counts.append(total_neurons)

        # Send weights and new architecture to each worker
        for worker_id in range(num_workers):
            network_queue.put((modified_arch, modified_params))

        # Save the training history
        logger.save_history()

    logger.plot_learning_curve(all_rewards, rewards_per_worker, neuron_counts)

    # End timing the training
    end_time = time.time()

    # Calculate average reward
    average_reward = sum(all_rewards) / len(all_rewards)

    # Log the training summary using the Logger class
    logger.log_training_summary(start_time, end_time, best_reward, average_reward, best_policy_arch, best_iteration)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train or evaluate the policy.')
    parser.add_argument('--cfg', type=str, default='config.yaml', help='Path to the configuration file.')
    parser.add_argument('--env', type=str, help='Specify the environment name to override the config file.')
    parser.add_argument('--eval', type=str, help='Only evaluate a trained policy. Specify the directory to load the policy from.')
    parser.add_argument('--logdir', type=str, help='Specify the directory for logging.')
    parser.add_argument('--record', action='store_true', help='Record video of the best model during evaluation.')
    parser.add_argument('--plot', type=str, nargs='+', help='Plot learning curve from the specified JSON file(s).')
    args = parser.parse_args()

    # Load configuration
    if args.cfg:
        cfg = load_config(args.cfg) 
    else:
        cfg = load_config('config.yaml') # Use the --cfg argument to load the configuration file

    # Override env_name if specified
    if args.env:
        cfg['env_name'] = args.env
    
    if args.logdir:
        cfg['logdir'] = args.logdir

    # Assert if dropout rates and net_arch are consistent
    assert len(cfg['thresholds']) == len(cfg['policy_kwargs']['net_arch']['pi'])

    if args.eval:
        import pathlib
        cfg['logdir'] = pathlib.Path(args.eval).parent.parent
        # Evaluate the policy
        eval_only(cfg, load_from=args.eval, record_video=args.record)

    elif args.plot:
        # Plot learning curve from the specified JSON file
        logger = Logger(cfg, save_cfg=False)
        logger.plot_learning_curve_from_json(args.plot)

    else:
        # Training
        result_queue = Queue()
        network_queue = Queue()

        # Worker processes
        workers = []
        for worker_id in range(cfg['num_workers']):
            p = Process(target=follower_process, args=(result_queue, network_queue, worker_id, cfg))
            workers.append(p)
            p.start()

        # Leader process
        leader_process(result_queue, network_queue, cfg)

        # Wait for worker processes to finish
        for p in workers:
            p.join()

if __name__ == "__main__":
    main()
