import gymnasium as gym
import stable_baselines3
import torch
import os
import numpy as np
import cloudpickle

class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = gym.make(cfg['env_name'], render_mode=cfg['render_mode'])
        self.make_policy(self.env)
        self.layer_outputs = []
    
    @property
    def policy_kwargs(self):
        return self.model.policy_kwargs
    
    @property
    def policy(self):
        return self.model.policy
    
    def make_policy(self, env, policy_kwargs=None, policy_weights=None):
        if policy_kwargs is None: # to initialize
            policy_kwargs = self.cfg['policy_kwargs']

        if isinstance(policy_kwargs['activation_fn'], str): # load from cfg
            activation_fn = getattr(torch.nn, policy_kwargs['activation_fn'])
        else:
            activation_fn = policy_kwargs['activation_fn']

        policy_kwargs = dict(
            activation_fn=activation_fn,
            net_arch=policy_kwargs['net_arch']
        )
        algorithm = getattr(stable_baselines3, self.cfg['algorithm'])
        device = self.cfg['device']
        # self.model = None
        self.model = algorithm("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, device=device)
        if policy_weights is not None:
            self.model.policy.load_state_dict(policy_weights)

    def save_policy(self, save_to=''):
        policy_kwargs = self.model.policy_kwargs
        policy_weights = self.model.policy.state_dict()
        with open(f'{save_to}/policy_kwargs.pkl', 'wb') as f:
            cloudpickle.dump(policy_kwargs, f)
        with open(f'{save_to}/policy_weights.pkl', 'wb') as f:
            cloudpickle.dump(policy_weights, f)

    def load_policy(self, load_from=''):
        for file_name in os.listdir(load_from):
            if 'policy_kwargs.pkl' in file_name:
                with open(os.path.join(load_from, file_name), 'rb') as f:
                    deserialized_kwargs = cloudpickle.load(f)
            if 'policy_weights.pkl' in file_name:
                with open(os.path.join(load_from, file_name), 'rb') as f:
                    deserialized_weights = cloudpickle.load(f)

        self.make_policy(self.env, deserialized_kwargs, deserialized_weights)

    def learn(self, total_timesteps):
        self.model.learn(total_timesteps)

    def predict_with_hidden_logging(self, model, observation):
        """
        Wraps model.predict to log hidden output values.

        Args:
        model: The MLP model.
        observation: The observation for prediction.

        Returns:
        A tuple containing the action and the hidden state values.
        """

        # Access the policy network
        policy = model.policy

        # Register a hook to capture hidden states
        hidden_state = {}

        def hook(module, input, output):
            hidden_state[module] = output.detach() # or output

        # Assuming the MLP extractor has layers you want to hook into
        # and your policy uses 'mlp_extractor'
        handles = []  # Store hook handles for removal
        try:
            for i, layer in enumerate(policy.mlp_extractor.policy_net):
                if isinstance(layer, torch.nn.Linear):
                    handle = layer.register_forward_hook(hook)
                    handles.append(handle)
        except AttributeError:
            print('mlp_extractor not found in policy network')

        # Make the prediction
        action, _states = model.predict(observation, deterministic=True)

        # Remove the hooks using the handles
        for handle in handles:
            handle.remove()

        return action, hidden_state

    def evaluate_policy(self, seed=None, num_eval_episodes=1, num_eval_steps_per_episode=1000, record_video=False):
        total_rewards = []
        self.layer_outputs = []
        
        # Wrap the environment for video recording if record_video is True
        if record_video:
            video_folder = os.path.join(self.cfg['logdir'], 'videos')
            self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder, episode_trigger=lambda x: True)
        
        hidden_states = []
        for _ in range(num_eval_episodes):
            observation, info = self.env.reset(seed=seed)
            episode_reward = 0
            for _ in range(num_eval_steps_per_episode):
                action, hidden_state = self.predict_with_hidden_logging(self.model, observation)
                hidden_states.append(hidden_state)
                observation, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    self.env.reset(seed=seed)
            total_rewards.append(episode_reward)
        
        # Extract hidden states per layer to make hidden_logs {layer: tensor.shape[samples, units]}
        hidden_states_per_layer = {}  
        for hidden_state in hidden_states:
            for key, val in hidden_state.items():
                # print(key, val.shape)
                if key not in hidden_states_per_layer:
                    hidden_states_per_layer[key] = []
                hidden_states_per_layer[key].append(val)
        self.hidden_logs = {}
        for layer, states in hidden_states_per_layer.items():
            self.hidden_logs.update({layer: torch.stack(states).squeeze(1)})
        
        self.env.close()
        return np.mean(total_rewards)
    
    def clear_hidden_logs(self):
        self.hidden_logs = []
