from abc import ABC, abstractmethod
from typing import Self

from drl_lab.lib.core.types import Experience


class ExperienceGenerator(ABC):
    @abstractmethod
    def __iter__(self) -> Self:
        """Return an iterator object."""
        return self

    @abstractmethod
    def __next__(self) -> Experience:
        """Returns the next environmental experience."""
