from abc import ABC, abstractmethod
from drl_lab.lib.experience.types import Experience


class ExperienceGenerator(ABC):
    @abstractmethod
    def __iter__(self):
        """Return an iterator object."""
        return self

    @abstractmethod
    def __next__(self) -> Experience:
        """Returns the next environmental experience."""
        # NOTE: We can stop iteration at max steps by raising StopIteration
        # if we choose to make this a feature of the experience generator.
