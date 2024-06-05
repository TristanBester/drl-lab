from dataclasses import dataclass, field

from numpy.typing import ArrayLike


@dataclass
class Experience:
    obs: ArrayLike
    action: ArrayLike
    obs_next: ArrayLike
    reward: float
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)
