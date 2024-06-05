from dataclasses import dataclass, field

import torch


@dataclass
class Experience:
    obs: torch.Tensor
    action: torch.Tensor
    obs_next: torch.Tensor
    reward: torch.Tensor
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)
