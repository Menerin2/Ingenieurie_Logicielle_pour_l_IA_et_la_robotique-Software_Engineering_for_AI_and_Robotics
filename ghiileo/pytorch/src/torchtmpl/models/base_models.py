# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch.nn as nn


def Linear(cfg, input_size, num_classes):
    """
    cfg: a dictionnary with possibly some parameters
    input_size: (C, H, W) input size tensor
    num_classes: int
    """
    in_features = reduce(operator.mul, input_size)

    layers = [
        nn.Flatten(),
        nn.Linear(in_features, num_classes),
    ]

    return nn.Sequential(*layers)


def FFN(cfg, input_size, num_classes):
    """
    cfg: a dictionnary with possibly some parameters
    input_size: (C, H, W) input size tensor
    num_classes: int
    """
    in_features = reduce(operator.mul, input_size)
    
    num_layers = cfg.get("num_layers", 1)
    num_hidden = cfg.get("num_hidden", 32)
    use_dropout = cfg.get("use_dropout", False)
    
    layers = [nn.Flatten()]

    for _ in range(num_layers):
        layers.append(nn.Linear(in_features, num_hidden))
        layers.append(nn.ReLU())
        if use_dropout:
            layers.append(nn.Dropout(p=0.5))
        in_features = num_hidden

    layers.append(nn.Linear(in_features, num_classes))


    return nn.Sequential(*layers)
