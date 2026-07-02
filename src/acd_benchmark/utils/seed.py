import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable pilot runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
