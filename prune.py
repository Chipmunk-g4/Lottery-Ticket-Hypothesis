import math

import torch
import torch.nn as nn
import torch.nn.utils.prune as P

from utils import get_layers

def prune_network(model, prune_ratio = 0.1, prune_method = 'l1', prune_type = 'local', layers_type = (nn.Linear, nn.Conv2d)):

    if prune_type == 'local':
        if prune_method == 'l1':
            prune_func = P.l1_unstructured
        elif prune_method == 'random':
            prune_func = P.random_unstructured
        else: raise ValueError

        for prune_ratio, layer in zip(prune_ratio, get_layers(model, layers_type)):
            prune_func(layer, "weight", prune_ratio)
            prune_func(layer, "bias", prune_ratio)

    elif prune_type == 'global':
        if prune_method == 'l1':
            method = P.L1Unstructured
        elif prune_method == 'random':
            method = P.RandomUnstructured

        parameters_to_prune = []
        for layer in get_layers(model, layers_type): parameters_to_prune += [(layer, 'weight'), (layer, 'bias')]

        P.global_unstructured(parameters=parameters_to_prune, pruning_method=method, amount = prune_ratio)


def compute_stats(model, layers_type = (nn.Linear, nn.Conv2d)):
    stats = {}
    total_params = 0
    total_pruned_params = 0

    for layer_ix, layer in enumerate(get_layers(model, layers_type)):
        weight_mask = layer.weight_mask
        bias_mask = layer.bias_mask

        params = weight_mask.numel() + bias_mask.numel()
        pruned_params = ((weight_mask == 0).sum() + (bias_mask == 0).sum()).item()

        total_params += params
        total_pruned_params += pruned_params

        stats[f"layer{layer_ix}_total_params"] = params
        stats[f"layer{layer_ix}_pruned_params"] = pruned_params
        stats[f"layer{layer_ix}_actual_prune_ratio"] = pruned_params / params

    stats["total_params"] = total_params
    stats["total_pruned_params"] = total_pruned_params
    stats["actual_prune_ratio"] = total_pruned_params / total_params

    return stats

def copy_weights(unpruned_model, pruned_model, layers_type):
    zipped = zip(get_layers(unpruned_model, layers_type), get_layers(pruned_model, layers_type))

    for layer_unpruned, layer_pruned in zipped:
        assert check_layer_pruned(layer_pruned)
        assert not check_layer_pruned(layer_unpruned)

        with torch.no_grad():
            layer_pruned.weight_orig.copy_(layer_unpruned.weight)
            layer_pruned.bias_orig.copy_(layer_unpruned.bias)

def check_layer_pruned(layer):
    params = {param_name for param_name, _ in layer.named_parameters()}
    expected_params = {"weight_orig", "bias_orig"}

    return params == expected_params