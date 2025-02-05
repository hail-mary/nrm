import os
import yaml
import pprint
import cloudpickle
import warnings
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import json


class Logger:
    def __init__(self, cfg, save_cfg=True):     
        self.iteration = -1
        self.history = {
            "iterations": [],
            "mean_rewards": [],
            "std_rewards": [],
            "neuron_counts": []
        }
        # make log directory if not exists
        if not os.path.exists(cfg['logdir']):
            os.mkdir(cfg['logdir'])
        else:
            warnings.warn(f'{cfg["logdir"]} already exists. Logs may be overwritten.')

        # save configuration
        if save_cfg:
            # Pretty-print the configuration
            print('\n#------------------ Loaded Configuration --------------------#')
            pprint.pprint(cfg)
            config_dir = os.path.join(cfg['logdir'], 'config.yaml')
            with open(config_dir, 'w') as file:
                yaml.dump(cfg, file, indent=4)
                print(f'\n >> saved to {config_dir}.')

        self.cfg = cfg
    
    def step(self):
        self.iteration += 1
    
    def save_checkpoint(self, results, save_to='checkpoints'):
        # results = [[worker_id, reward, policy_kwargs, policy_weight],
        #              worker_id, reward, policy_kwargs, policy_weight], ... ]

        # sort results in reward decending order
        results.sort(key=lambda x: x[1], reverse=True)

        # find the best worker and save at every checkpoints
        best_worker = results[0][0]
        best_reward = results[best_worker][1]
        best_policy_kwargs = results[best_worker][2]
        best_policy_weight = results[best_worker][3]

        idx = self.iteration
        checkpoints_dir = os.path.join(self.cfg['logdir'], save_to)
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        else:
            warnings.warn(f'{checkpoints_dir} already exists. Checkpoints may be overwritten.')
        path_to_model = os.path.join(checkpoints_dir, f'Iteration-{idx}')
        
        if os.path.exists(path_to_model):
            warnings.warn(f'{path_to_model} already exists. Files may be overwritten.')
        else:
            os.mkdir(path_to_model)
        
        with open(f'{path_to_model}/best_policy_kwargs.pkl', 'wb') as f:
            cloudpickle.dump(best_policy_kwargs, f)
        with open(f'{path_to_model}/best_policy_weights.pkl', 'wb') as f:
            cloudpickle.dump(best_policy_weight, f)

    def plot_learning_curve(self, all_rewards, rewards_per_worker, neuron_counts):
        # Plot the learning curve with dual y-axes
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot all_rewards on the first y-axis
        ax1.set_xlabel('Environment steps [1e3]', fontsize=20)
        ax1.set_ylabel('Cumulative Reward', fontsize=20)
        ax1.plot(all_rewards, label='Mean Reward')
        ax1.tick_params(axis='y', labelsize=15)

        # Fill between for mean reward range
        iterations = range(len(all_rewards))
        rewards_min = np.min(rewards_per_worker, axis=0)
        rewards_max = np.max(rewards_per_worker, axis=0)
        ax1.fill_between(
            iterations,
            rewards_min,
            rewards_max,
            color="blue",
            alpha=0.2,
            label="Reward Range",
        )

        # Ensure x-axis ticks are integers
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax1.tick_params(axis='x', labelsize=15)
        ax1.set_xticks(np.arange(0, self.cfg['num_iterations']+1, step=100))  # Adjust step as needed

        # Create a second y-axis to plot total_neurons
        ax2 = ax1.twinx()
        ax2.set_ylabel('Total Neurons', fontsize=20)
        ax2.plot(neuron_counts, label='Total Neurons', color='red', linestyle='--')
        ax2.tick_params(axis='y', labelsize=15)

        # Ensure y-axis ticks for total neurons are integers
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        fig.tight_layout()
        plt.grid(True)

        # Add legends and bring them to the front
        legend1 = ax1.legend(loc='upper left', fontsize=14)
        legend2 = ax2.legend(loc='lower right', fontsize=14, bbox_to_anchor=(1, 0.05))
        legend1.set_zorder(10)
        legend2.set_zorder(10)

        # Determine the maximum iteration and format it
        max_iteration = len(all_rewards) * self.cfg['timesteps_per_iteration']  # Assuming each step represents 1000 iterations
        if max_iteration >= 1_000_000:
            iteration_str = f"{max_iteration // 1_000_000}M"
        elif max_iteration >= 1_000:
            iteration_str = f"{max_iteration // 1_000}K"
        else:
            iteration_str = str(max_iteration)

        # Save the plot as an EPS file with the formatted iteration count
        eps_file_path = os.path.join(self.cfg['logdir'], f'learning_curve_{iteration_str}.eps')
        plt.savefig(eps_file_path, format='eps')
        print(f"Learning curve saved as {eps_file_path}")

        # Save the plot as a PNG file with the formatted iteration count
        png_file_path = os.path.join(self.cfg['logdir'], f'learning_curve_{iteration_str}.png')
        plt.savefig(png_file_path, format='png')
        print(f"Learning curve saved as {png_file_path}")

        # Save the training history
        self.save_history()

    def log_training_summary(self, start_time, end_time, best_reward, average_reward, best_policy_arch, best_iteration):
        # Calculate the duration in hours
        duration_seconds = end_time - start_time
        duration_hours = duration_seconds / 3600

        # Create the summary
        summary = (
            f"Training Summary:\n"
            f"-----------------\n"
            f"Total Training Time: {duration_hours:.2f} hours\n"
            f"Ranking Thresholds: {self.cfg['thresholds']}\n"
            f"Slope Threshold: {self.cfg['slope_threshold']}\n"
            f"Best Reward: {best_reward}\n"
            f"Best Iteration: {best_iteration}\n"
            f"Best Policy Architecture: {best_policy_arch}\n"
        )

        # Print the summary
        print(summary)

        # Construct the directory path from the configuration
        logdir = self.cfg['logdir']
        os.makedirs(logdir, exist_ok=True)

        # Log the summary to a file in the specified directory
        summary_path = os.path.join(logdir, 'training_summary.txt')
        with open(summary_path, 'a') as summary_file:
            summary_file.write(summary)

    def log_iteration(self, mean_reward, std_reward, total_neurons):
        # Log the data for each iteration
        self.history["iterations"].append(self.iteration)
        self.history["mean_rewards"].append(mean_reward)
        self.history["std_rewards"].append(std_reward)
        self.history["neuron_counts"].append(total_neurons)

    def save_history(self):
        # Save the history to a JSON file
        history_path = os.path.join(self.cfg['logdir'], 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def plot_learning_curve_from_json(self, json_files, plot_every=10):
        # Initialize the plot
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Iterate over each JSON file
        for json_file in json_files:
            with open(json_file, 'r') as f:
                history = json.load(f)

            # Select every 10th iteration
            iterations = history['iterations'][::plot_every]
            mean_rewards = history['mean_rewards'][::plot_every]
            std_rewards = history['std_rewards'][::plot_every]
            neuron_counts = history['neuron_counts'][::plot_every]

            # Plot mean_rewards on the first y-axis
            ax1.plot(iterations, mean_rewards, label=f'Mean Reward ({os.path.basename(json_file)})')
            ax1.fill_between(iterations, 
                             np.array(mean_rewards) - np.array(std_rewards), 
                             np.array(mean_rewards) + np.array(std_rewards), 
                             alpha=0.2, label=f'Reward Std Dev ({os.path.basename(json_file)})')

        # Set labels and ticks for the first y-axis
        ax1.set_xlabel('Environment steps [1e3]', fontsize=20)
        ax1.set_ylabel('Cumulative Reward', fontsize=20)
        ax1.tick_params(axis='y', labelsize=12)
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax1.tick_params(axis='x', labelsize=12)
        ax1.set_xticks(np.arange(0, self.cfg['num_iterations']+1, step=100))  # Adjust step as needed

        # Create a second y-axis to plot total_neurons
        ax2 = ax1.twinx()
        for json_file in json_files:
            with open(json_file, 'r') as f:
                history = json.load(f)
            neuron_counts = history['neuron_counts'][::plot_every]
            ax2.plot(iterations, neuron_counts, label=f'Total Neurons ({os.path.basename(json_file)})', linestyle='--')

        # Set labels and ticks for the second y-axis
        ax2.set_ylabel('Total Neurons', fontsize=20)
        ax2.tick_params(axis='y', labelsize=12)
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        fig.tight_layout()
        plt.grid(True)

        # Add legends and bring them to the front
        # legend1 = ax1.legend(loc='upper left', fontsize=14)
        # legend2 = ax2.legend(loc='lower right', fontsize=14, bbox_to_anchor=(1, 0.05))
        # legend1.set_zorder(10)
        # legend2.set_zorder(10)

        # Comment out or remove plt.show() to prevent displaying the plot
        plt.show()