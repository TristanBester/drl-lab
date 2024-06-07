import gymnasium as gym
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, Timer
from ignite.handlers.tensorboard_logger import (GradsHistHandler,
                                                GradsScalarHandler,
                                                TensorboardLogger,
                                                WeightsHistHandler,
                                                WeightsScalarHandler)

from drl_lab.lib.core.interfaces import Agent
from drl_lab.lib.ignite.handlers import tensorboard as tbh
from drl_lab.lib.ignite.handlers.agent import (EpsilonDecayHandler,
                                               EpsilonRecorderHandler)
from drl_lab.lib.ignite.handlers.eval import EvalHandler
from drl_lab.lib.ignite.handlers.timer import (TimingHandler,
                                               TrainingTimeFractionHandler)
from drl_lab.lib.ignite.handlers.utils import global_step_transform


def register_epsilon_handlers(
    engine: Engine,
    agent: Agent,
    record_freq: int,
    decay_freq: int,
):
    """Register event handlers for epsilon-greedy exploration."""
    epsilon_recorder = EpsilonRecorderHandler(agent)
    epsilon_decay = EpsilonDecayHandler(agent)

    engine.add_event_handler(
        event_name=Events.ITERATION_COMPLETED(every=record_freq),
        handler=epsilon_recorder,
    )
    engine.add_event_handler(
        event_name=Events.ITERATION_COMPLETED(every=decay_freq),
        handler=epsilon_decay,
    )


def register_checkpointing_handlers(engine: Engine, save_dict: dict, save_freq: int):
    """Register event handlers for checkpointing."""
    checkpointer = Checkpoint(
        to_save=save_dict,
        save_handler="checkpoints/",
        n_saved=5,
        global_step_transform=global_step_transform,
    )
    engine.add_event_handler(
        event_name=Events.ITERATION_COMPLETED(every=save_freq),
        handler=checkpointer,
    )


def register_eval_handlers(
    engine: Engine, agent: Agent, env: gym.Env, success_reward: float, eval_freq: int
):
    evaluator = EvalHandler(agent, env, success_transform=lambda x: x >= success_reward)
    engine.add_event_handler(
        event_name=Events.ITERATION_COMPLETED(every=eval_freq),
        handler=evaluator,
    )


def register_timing_handlers(engine: Engine, time_freq: int):
    timer = Timer()
    timer.attach(
        engine,
        start=Events.STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    training_timer = TimingHandler(timer)
    training_fraction_timer = TrainingTimeFractionHandler(alpha=0.1)

    engine.add_event_handler(
        event_name=Events.ITERATION_COMPLETED(every=time_freq),
        handler=training_timer,
    )
    engine.add_event_handler(
        event_name=Events.ITERATION_COMPLETED(every=time_freq),
        handler=training_fraction_timer,
    )


def register_tensorboard_handlers(
    engine, model, agent, render_env, scalar_freq, weight_freq, grad_freq, record_freq
):
    def attach_scalar_handlers():
        training_iterations_per_second = tbh.ScalarHandler(
            "training_iterations_per_second", global_step_transform, "1-training"
        )
        training_time_fraction = tbh.ScalarHandler(
            "training_time_fraction", global_step_transform, "1-training"
        )
        ep_returns = tbh.ScalarHandler(
            "ep_returns", global_step_transform, "1-training"
        )
        ep_lengths = tbh.ScalarHandler(
            "ep_lengths", global_step_transform, "1-training"
        )
        interactions_per_second = tbh.ScalarHandler(
            "interactions_per_second", global_step_transform, "1-training"
        )
        epsilon = tbh.ScalarHandler("epsilon", global_step_transform, "1-training")
        eval_returns = tbh.ScalarHandler(
            "eval_returns", global_step_transform, "2-evaluation"
        )
        eval_success_rate = tbh.ScalarHandler(
            "eval_success_rate", global_step_transform, "2-evaluation"
        )

        tb_logger.attach(
            engine,
            log_handler=training_iterations_per_second,
            event_name=Events.ITERATION_COMPLETED(every=scalar_freq),
        )
        tb_logger.attach(
            engine,
            log_handler=training_time_fraction,
            event_name=Events.ITERATION_COMPLETED(every=scalar_freq),
        )
        tb_logger.attach(
            engine,
            log_handler=ep_returns,
            event_name=Events.ITERATION_COMPLETED(every=scalar_freq),
        )
        tb_logger.attach(
            engine,
            log_handler=ep_lengths,
            event_name=Events.ITERATION_COMPLETED(every=scalar_freq),
        )
        tb_logger.attach(
            engine,
            log_handler=interactions_per_second,
            event_name=Events.ITERATION_COMPLETED(every=scalar_freq),
        )
        tb_logger.attach(
            engine,
            log_handler=epsilon,
            event_name=Events.ITERATION_COMPLETED(every=scalar_freq),
        )
        tb_logger.attach(
            engine,
            log_handler=eval_returns,
            event_name=Events.ITERATION_COMPLETED(every=scalar_freq),
        )
        tb_logger.attach(
            engine,
            log_handler=eval_success_rate,
            event_name=Events.ITERATION_COMPLETED(every=scalar_freq),
        )

    def attach_weight_handlers():
        weight_scalar = WeightsScalarHandler(model)
        weight_hist = WeightsHistHandler(model)

        tb_logger.attach(
            engine,
            log_handler=weight_scalar,
            event_name=Events.ITERATION_COMPLETED(every=weight_freq),
        )
        tb_logger.attach(
            engine,
            log_handler=weight_hist,
            event_name=Events.ITERATION_COMPLETED(every=weight_freq),
        )

    def attach_grad_handlers():
        grad_scalar = GradsScalarHandler(model)
        grad_hist = GradsHistHandler(model)

        tb_logger.attach(
            engine,
            log_handler=grad_scalar,
            event_name=Events.ITERATION_COMPLETED(every=grad_freq),
        )
        tb_logger.attach(
            engine,
            log_handler=grad_hist,
            event_name=Events.ITERATION_COMPLETED(every=grad_freq),
        )

    def attach_recorder_handlers():
        recorder = tbh.RecordEpisodeHandler(
            agent, render_env, global_step_transform=global_step_transform
        )
        tb_logger.attach(
            engine,
            log_handler=recorder,
            event_name=Events.ITERATION_COMPLETED(every=record_freq),
        )

    tb_logger = TensorboardLogger(log_dir="logs")
    attach_scalar_handlers()
    attach_weight_handlers()
    attach_grad_handlers()
    attach_recorder_handlers()
