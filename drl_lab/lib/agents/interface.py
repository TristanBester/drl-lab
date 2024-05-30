from abc import ABC, abstractmethod
import torch


class Agent(ABC):
    @abstractmethod
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Return actions based on given observations."""
