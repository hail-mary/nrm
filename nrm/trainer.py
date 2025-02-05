import torch
import numpy as np
from collections import Counter

class Trainer:
    """
    a set of utility functions for training
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.candidates_to_remove = {}

    
    def average_params(self, params_list):
        avg_params = {}
        for key in params_list[0].keys():
            avg_params[key] = torch.stack([params[key] for params in params_list]).mean(dim=0)
        return avg_params
    
    def calculate_norms(self, pre_avg_params, avg_params):
        pre_avg_norm = torch.cat([p.flatten() for p in pre_avg_params.values()]).norm()
        avg_norm = torch.cat([p.flatten() for p in avg_params.values()]).norm()
        return pre_avg_norm, avg_norm

    def preprocess(self, raw_arch, raw_params):
        # First, preprocess arch
        layer_types = ('policy', 'value')
        archs = tuple(raw_arch.values())
        arch = dict(zip(layer_types, archs))

        # Then, preprocess params
        params = {}
        for layer_type, layers in arch.items():
            for key, val in raw_params.items():
                if layer_type in key and 'weight' in key:
                    params.update({key: val})
                elif layer_type in key and 'bias' in key:
                    params.update({key: val})
                elif 'action' in key and layer_type == 'policy':
                    params.update({key: val})

        # keep auxilary info to reconstruct
        aux = {'log_std': raw_params.pop('log_std'), 
               'action_net.bias': raw_params.pop('action_net.bias'),
               'value_net.bias': raw_params.pop('value_net.bias'),
        }

        return arch, params, aux

    def reconstruct(self, arch, params, aux):
        # First, reconstruct arch
        layer_types = ('pi', 'vf')
        archs = tuple(arch.values())
        processed_arch = dict(zip(layer_types, archs))

        # Then, reconstruct params
        # Missing key(s) in state_dict: "log_std", "action_net.bias", "value_net.bias"
        processed_params = params
        processed_params.update(aux)

        return processed_arch, processed_params

    def remove_indices(self, tensor, indices_to_remove, row_or_col=None):
        if tensor.dim() == 1: # for bias
            # Create a boolean mask indicating which elements to keep
            mask = torch.ones(tensor.shape[0], dtype=torch.bool)
            mask[indices_to_remove] = False
        elif tensor.dim() == 2 and row_or_col == 'row': # for weight
            # Create a boolean mask indicating which rows to keep
            mask = torch.ones_like(tensor, dtype=torch.bool)
            mask[indices_to_remove, :] = False
        elif tensor.dim() == 2 and row_or_col == 'col': # for weight
            # Create a boolean mask indicating which columns to keep
            mask = torch.ones_like(tensor, dtype=torch.bool)
            mask[:, indices_to_remove] = False
        else:
            raise ValueError("Invalid tensor dimension or row_or_col value")

        # Apply the mask to select the desired elements
        new_tensor = tensor[mask]
        if row_or_col == 'row':
            new_units = tensor.shape[0] - len(indices_to_remove)
            new_tensor = new_tensor.reshape(new_units, tensor.shape[1])
        elif row_or_col == 'col':
            new_units = tensor.shape[1] - len(indices_to_remove)
            new_tensor = new_tensor.reshape(tensor.shape[0], new_units)

        return new_tensor

    def modify_network(self, params, arch, rankings):
        thresholds = self.cfg["thresholds"]
        for layer_type, layers in arch.items():
            if layer_type == 'value': break
            for layer_idx in range(len(layers)):
                ranking = rankings[layer_idx]
                if len(ranking) == 0: continue
                threshold = thresholds[layer_idx]
                indices_to_remove = [idx for idx, count in ranking.items() if count >= threshold]
                num_to_drop = len(indices_to_remove)

                # First, modify network architecture
                arch[layer_type][layer_idx] -= num_to_drop

                # Process weights
                weight_key_1 = f"mlp_extractor.{layer_type}_net.{2 * layer_idx}.weight"  # Adjust the key format as per your architecture
                if weight_key_1 in params:
                    params[weight_key_1] = self.remove_indices(params[weight_key_1], indices_to_remove, row_or_col='row')
                
                # Process biases
                bias_key = f"mlp_extractor.{layer_type}_net.{2 * layer_idx}.bias"  # Adjust the key format as per your architecture
                if bias_key in params:
                    params[bias_key] = self.remove_indices(params[bias_key], indices_to_remove)

                # Process weights
                weight_key_2 = f"mlp_extractor.{layer_type}_net.{2 * (layer_idx + 1)}.weight"  # Adjust the key format as per your architecture
                if weight_key_2 in params:
                    params[weight_key_2] = self.remove_indices(params[weight_key_2], indices_to_remove, row_or_col='col')

                # Process output
                output_key = "action_net.weight" if layer_type == 'policy' else "value_net.weight"
                if output_key in params and layer_idx == len(layers) - 1 :
                    params[output_key] = self.remove_indices(params[output_key], indices_to_remove, row_or_col='col')

        return arch, params

    def yield_rankings(self, hidden_logs):
        hidden_stats = {}
        for layer, logs in enumerate(hidden_logs):
            hidden_stats.update({layer: {'mean': logs.mean(dim=0), 'std': logs.std(dim=0)}})
        
        criteria = 'std'
        percentile = 0.1
        for layer, stats in hidden_stats.items():
            if criteria == 'mean':
                stat = stats['mean']
            elif criteria == 'std':
                stat = stats['std']
            threshold = torch.quantile(stat, percentile)
            
            for idx, neuron in enumerate(stat):
                if neuron < threshold:
                    if len(self.candidates_to_remove) == 0:
                        self.candidates_to_remove["layer1"] = []
                        self.candidates_to_remove["layer2"] = []
                    if layer == 0:
                        self.candidates_to_remove["layer1"].append(idx)
                    else:
                        self.candidates_to_remove["layer2"].append(idx)
        
        if len(self.candidates_to_remove) > 0:
            self.rankings = Counter(self.candidates_to_remove["layer1"]), Counter(self.candidates_to_remove["layer2"])
                            
        return self.rankings
    
    def clear_rankings(self):
        self.candidates_to_remove = {}
        self.rankings = None
