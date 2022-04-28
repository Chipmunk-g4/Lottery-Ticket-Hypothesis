import torch
import torch.nn as nn

def get_layers(
        model       : nn.Module,
        layers_type : tuple = (nn.Linear, nn.Conv2d)
    ):
    return [module for module in model.modules() if isinstance(module, layers_type)]