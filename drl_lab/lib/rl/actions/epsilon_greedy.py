from drl_lab.lib.rl.actions.interface import ActionSelector
import torch
import numpy as np


class EpsilonGreedyActionSelector(ActionSelector):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def __call__(self, info: torch.Tensor) -> torch.Tensor:
        """Select an action based on the given information."""
        if np.random.random() < self.epsilon:
            action = torch.randint(2, (1,))[0]
            return action
        else:
            return torch.argmax(info)