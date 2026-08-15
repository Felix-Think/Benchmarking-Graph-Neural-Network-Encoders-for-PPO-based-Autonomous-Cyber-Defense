import torch.nn as nn 
import torch 

def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)

        if m.bias is not None:
            nn.init.zeros_(m.bias)