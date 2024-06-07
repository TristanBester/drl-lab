from typing import Generator

from ignite.engine import Engine


class RecordEpisodeStatistics:
    """Wrapper for recoding episode statistics.

    Episode statistics are recorded using exponential moving averages
    which are stored in the PyTorch Ignite engine state (the preferred
    state storage location in ignite).
    """

    def __init__(self, generator: Generator, engine: Engine, alpha: float = 0.1):
        self.generator = generator
        self.engine = engine
        self.alpha = alpha

        self.engine.state.ep_returns = 0
        self.engine.state.ep_lengths = 0

        self.ep_steps = 0
        self.ep_returns = 0

    def __iter__(self):
        return self

    def __next__(self):
        exp = next(self.generator)

        self.ep_steps += 1
        self.ep_returns += exp.reward

        if exp.terminated or exp.truncated:
            self.engine.state.ep_lengths = (
                self.alpha * self.ep_steps
                + (1 - self.alpha) * self.engine.state.ep_lengths
            )
            self.engine.state.ep_returns = (
                self.alpha * self.ep_returns
                + (1 - self.alpha) * self.engine.state.ep_returns
            )

            self.ep_steps = 0
            self.ep_returns = 0
        return exp
