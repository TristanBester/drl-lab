from abc import ABC, abstractmethod

import torch


class ActionSelector(ABC):
    @abstractmethod
    def __call__(self, info: torch.Tensor) -> torch.Tensor:
        """Select an action based on the given information."""
