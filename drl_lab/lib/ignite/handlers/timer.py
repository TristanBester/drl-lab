from ignite.engine import Engine
from ignite.handlers import Timer


class TimingHandler:
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


class TrainingTimeFractionHandler:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(
        self,
        engine: Engine,
    ) -> None:
        try:
            interaction_time = engine.state.interactions_per_second**-1
            training_time = engine.state.training_iterations_per_second**-1
        except AttributeError:
            return

        training_fraction = training_time / (training_time + interaction_time)

        if hasattr(engine.state, "training_time_fraction"):
            engine.state.training_time_fraction = (
                self.alpha * training_fraction
                + (1 - self.alpha) * engine.state.training_time_fraction
            )
        else:
            engine.state.training_time_fraction = training_fraction
