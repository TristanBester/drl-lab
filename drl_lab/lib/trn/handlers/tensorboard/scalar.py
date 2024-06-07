from typing import Callable

from ignite.engine import Engine, Events
from ignite.handlers import TensorboardLogger
from ignite.handlers.base_logger import BaseHandler


class ScalarHandler(BaseHandler):
    def __init__(
        self,
        scalar_name: str,
        global_step_transform: Callable,
        section_name: str = None,
    ):
        self.scalar_name = scalar_name
        self.global_step_transform = global_step_transform
        self.section_name = section_name

    def __call__(
        self, engine: Engine, logger: TensorboardLogger, event_name: str | Events
    ):
        if not hasattr(engine.state, self.scalar_name):
            # TODO: put warning here...
            return

        name = (
            f"{self.section_name}/{self.scalar_name}"
            if self.section_name is not None
            else self.scalar_name
        )
        logger.writer.add_scalar(
            name,
            getattr(engine.state, self.scalar_name),
            global_step=self.global_step_transform(engine, event_name),
        )
