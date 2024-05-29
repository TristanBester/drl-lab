from drl_lab.lib.actions.interface import ActionSelector
import torch
import numpy as np


class EpsilonGreedyActionSelector(ActionSelector):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def __call__(self, info: torch.Tensor) -> torch.Tensor:
        """Select an action based on the given information."""
        if np.random.random() < self.epsilon:
            return torch.randint(info.size(-1), (info.size(0),))
        else:
            return torch.argmax(info, dim=-1)
