"""Device boundaries for CPU simulation and the PPO learner."""

import copy

import torch


def learner_device(name):
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("The benchmark supports CPU or CUDA learners")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; select an NVIDIA GPU runtime. No CPU fallback.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index unavailable: {device.index}")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return device


def cpu_state_dict(policy):
    return {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }


def cpu_inference_policy(policy):
    return copy.deepcopy(policy).cpu().eval()
