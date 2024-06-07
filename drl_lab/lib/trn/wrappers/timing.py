import time
from typing import Generator

from ignite.engine import Engine


class InteractionTimingHandler:
    def __init__(self, generator: Generator, engine: Engine, alpha: float = 0.1):
        self.generator = generator
        self.engine = engine
        self.alpha = alpha

        self.engine.state.interactions_per_second = 0

    def __iter__(self):
        return self

    def __next__(self):
        start_time = time.time()
        exp = next(self.generator)
        end_time = time.time()
        time_delta = end_time - start_time
        steps_per_sec = 1.0 / time_delta

        self.engine.state.interactions_per_second = (
            self.alpha * steps_per_sec
            + (1 - self.alpha) * self.engine.state.interactions_per_second
        )
        return exp
