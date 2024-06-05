import functools
from typing import Callable

from ignite.engine import Engine


class RecordEpisodeStatistics:
    """Decorator to record episode statistics.

    Episode statistics are recorded using exponential moving averages
    which are stored in the PyTorch Ignite engine state (the preferred
    state storage location in ignite).
    """

    def __init__(self, func: Callable, alpha: float = 0.1) -> None:
        """Constructor."""
        self.func = func
        self.alpha = alpha
        self.engine = None

        functools.update_wrapper(self, func)

    def set_engine(self, engine: Engine) -> None:
        """Set the PyTorch Ignite engine.

        Add the required attributed to the engine state.

        Args:
            engine (Engine): PyTorch Ignite engine.

        Returns:
            None
        """
        self.engine = engine
        print("Adding state variables to engine...")
        self.engine.state.ep_returns = 0
        self.engine.state.ep_lengths = 0

    def __call__(self, *args, **kwargs):
        """Decorator for the generator function."""
        if self.engine is None:
            raise ValueError("Engine set before calling the generator.")

        ep_steps = 0
        ep_returns = 0

        for exp in self.func(*args, **kwargs):
            ep_steps += 1
            ep_returns += exp.reward

            if exp.terminated or exp.truncated:
                self.engine.state.ep_lengths = (
                    self.alpha * ep_steps
                    + (1 - self.alpha) * self.engine.state.ep_lengths
                )
                self.engine.state.ep_returns = (
                    self.alpha * ep_returns
                    + (1 - self.alpha) * self.engine.state.ep_returns
                )

                ep_steps = 0
                ep_returns = 0

            yield exp
