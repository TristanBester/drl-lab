class ScalarHandler(BaseHandler):
    def __init__(
        self,
        scalar_name: str,
    ):
        self.scalar_name = scalar_name
        self.global_step_transform = global_step_transform

    def __call__(self, engine: Engine, logger: WandBLogger, event_name: str | Events):
        print(
            f"iteration: {engine.state.iteration}\t{getattr(engine.state, self.scalar_name)}"
        )

        print(self.global_step_transform(engine, event_name))

        logger.log(
            {
                self.scalar_name: getattr(engine.state, self.scalar_name),
            },
            step=self.global_step_transform(engine, event_name),
        )
