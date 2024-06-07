import numpy as np
import torch

from drl_lab.lib.rl.interfaces import ActionSelector


class EpsilonGreedyActionSelector(ActionSelector):
    def __init__(self, start_val: float, end_val: float, steps: float):
        self.epsilon = start_val
        self.end_val = end_val
        self.steps = steps
        self.decay = (start_val - end_val) / steps

    def decay_epsilon(self):
        """Decay epsilon."""
        self.epsilon = max(self.end_val, self.epsilon - self.decay)

    def __call__(self, info: torch.Tensor) -> torch.Tensor:
        """Select an action based on the given information."""
        if np.random.random() < self.epsilon:
            action = torch.randint(2, (1,))[0]
            return action
        else:
            return torch.argmax(info)
