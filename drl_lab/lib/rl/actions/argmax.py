import torch

from drl_lab.lib.rl.interfaces import ActionSelector


class ArgmaxActionSelector(ActionSelector):
    def __call__(self, info: torch.Tensor) -> torch.Tensor:
        """Reuturn the index of the maximum value in the array."""
        return torch.argmax(info)
