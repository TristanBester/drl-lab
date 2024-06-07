from ignite.engine import Engine
from ignite.handlers import Timer


class CHandler:
    def __init__(self, timer: Timer, alpha: float = 0.1):
        self.timer = timer
        self.last_call = 0
        self.alpha = alpha

    def __call__(
        self,
        engine: Engine,
    ) -> None:
        time_delta = self.timer.value() - self.last_call
        steps_per_second = 1.0 / time_delta if time_delta > 0 else 0

        if hasattr(engine.state, "training_iterations_per_second"):
            engine.state.training_iterations_per_second = (
                self.alpha * steps_per_second
                + (1 - self.alpha) * engine.state.training_iterations_per_second
            )
        else:
            engine.state.training_iterations_per_second = steps_per_second

        self.last_call = self.timer.value()
