from ignite.engine import Engine, Events
from ignite.handlers import TensorboardLogger, Timer, global_step_from_engine


class TimingHandler:
    def __init__(self, timer: Timer, call_freq: int = 100):
        self.timer = timer
        self.last_time = 0
        self.call_freq = call_freq

    def __call__(
        self, engine: Engine, logger: TensorboardLogger, event_name: str | Events
    ) -> None:
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError(
                "Handler 'OutputHandler' works only with TensorboardLogger"
            )

        time_delta = self.timer.value() - self.last_time
        self.last_time = self.timer.value()
        steps_per_sec = 1.0 / time_delta * self.call_freq

        global_step = global_step_from_engine(engine)(None, event_name)
        logger.writer.add_scalar("time", steps_per_sec, global_step)
