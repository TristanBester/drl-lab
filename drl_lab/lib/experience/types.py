from dataclasses import dataclass, field
import torch


@dataclass
class Experience:
    obs: torch.Tensor
    action: torch.Tensor
    obs_next: torch.Tensor
    reward: torch.Tensor
    truncated: bool
    done: bool
    info: dict = field(default_factory=dict)
