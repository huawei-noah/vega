"""Torch activation functions."""
import torch.nn
from modnas.registry.arch_space import register

modules = [
    'ELU',
    'Hardshrink',
    'Hardtanh',
    'LeakyReLU',
    'LogSigmoid',
    'PReLU',
    'ReLU',
    'ReLU6',
    'RReLU',
    'SELU',
    'CELU',
    'Sigmoid',
    'Softplus',
    'Softshrink',
    'Softsign',
    'Tanh',
    'Tanhshrink',
    'Threshold',
]

for name in modules:
    attr = getattr(torch.nn, name, None)
    if attr is not None:
        register(attr)
