from drl_lab.lib.rl.actions.interface import ActionSelector
import torch


class ArgmaxActionSelector(ActionSelector):
    def __call__(self, info: torch.Tensor) -> torch.Tensor:
        """Reuturn the index of the maximum value in the array."""
        return torch.argmax(info)
