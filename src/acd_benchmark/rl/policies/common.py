"""Initialization helpers shared by policy implementations."""

import torch.nn as nn


def init_linear_weights(module: nn.Module) -> None:
    """Apply the benchmark's orthogonal initialization to linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
