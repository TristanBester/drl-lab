from drl_lab.lib.rl.actions.interface import ActionSelector
import torch


class DistributionActionSelector(ActionSelector):
    """
    Attributes:
        logits: A boolean indicating whether the input is logits or probabilities.
    """

    def __init__(self, logits: bool):
        self.logits = logits

    def __call__(self, info: torch.Tensor) -> torch.Tensor:
        """Sample actions based on the given probabilities."""
        if self.logits:
            return torch.distributions.Categorical(logits=info).sample()
        else:
            return torch.distributions.Categorical(probs=info).sample()
