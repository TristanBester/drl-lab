import copy
import time

import gymnasium as gym
import torch
import torch.optim as optim
from cpprb import ReplayBuffer
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, Timer
from ignite.handlers.tensorboard_logger import (GradsHistHandler,
                                                GradsScalarHandler,
                                                TensorboardLogger,
                                                WeightsHistHandler,
                                                WeightsScalarHandler)
from tensorboardX.record_writer import re
from torch.utils.checkpoint import CheckpointError

from drl_lab.lib.core.actions import (ArgmaxActionSelector,
                                      EpsilonGreedyActionSelector)
from drl_lab.lib.core.agents.value_agent import ValueAgent
from drl_lab.lib.core.experience.transition import \
    TransitionExperienceGenerator
from drl_lab.lib.core.interfaces import Agent
from drl_lab.lib.ignite.handlers import tensorboard as tbh
from drl_lab.lib.ignite.handlers.agent import (EpsilonDecayHandler,
                                               EpsilonRecorderHandler)
from drl_lab.lib.ignite.handlers.eval import EvalHandler
from drl_lab.lib.ignite.handlers.timer import (TimingHandler,
                                               TrainingTimeFractionHandler)
from drl_lab.lib.ignite.handlers.utils import global_step_transform
from drl_lab.lib.ignite.wrappers import (InteractionTimingHandler,
                                         RecordEpisodeStatistics, timing)
from drl_lab.projects.loss import dqn_loss
from drl_lab.projects.network import DeepQNetwork
from drl_lab.projects.utils import sync_networks


def batch_generator():
    for exp in exp_gen:
        buffer.add(
            obs=exp.obs,
            action=exp.action,
            obs_next=exp.obs_next,
            reward=exp.reward,
            terminated=exp.terminated,
            truncated=exp.truncated,
        )

        if buffer.get_stored_size() > 1000:
            yield buffer.sample(32)


def process_batch(engine: Engine, batch: dict):
    """Process batch."""
    optimizer.zero_grad()
    loss = dqn_loss(
        batch,
        value_net,
        target_net,
        device=device,
    )
    loss.backward()
    optimizer.step()

    if engine.state.iteration % 1000 == 0:
        sync_networks(value_net, target_net)

    engine.state.metrics["loss"] = loss.item()


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


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Networks
    device = torch.device("cpu")
    value_net = DeepQNetwork(obs_dim, n_actions)
    target_net = copy.deepcopy(value_net)
    buffer = ReplayBuffer(
        size=100000,
        env_dict={
            "obs": {"shape": obs_dim},
            "action": {"shape": 1},
            "obs_next": {"shape": obs_dim},
            "reward": {},
            "terminated": {},
            "truncated": {},
        },
    )
    optimizer = optim.Adam(value_net.parameters(), lr=0.0001)

    # Reinforcement Learning
    engine = Engine(process_batch)
    agent = ValueAgent(
        action_selector=EpsilonGreedyActionSelector(
            start_val=1.0, end_val=0.01, steps=250000
        ),
        device=device,
        value_net=value_net,
    )
    eval_agent = ValueAgent(
        action_selector=ArgmaxActionSelector(),
        device=device,
        value_net=value_net,
    )
    eval_env = gym.make("CartPole-v1")
    render_env = gym.make("CartPole-v1", render_mode="rgb_array")
    transition_generator = TransitionExperienceGenerator(env, agent)
    exp_gen = RecordEpisodeStatistics(transition_generator, engine)
    exp_gen = InteractionTimingHandler(exp_gen, engine)

    # Ignite Handlers
    register_epsilon_handlers(engine, agent, 100, 1)
    register_checkpointing_handlers(
        engine, {"model": value_net, "optimizer": optimizer}, 50000
    )
    register_eval_handlers(engine, eval_agent, eval_env, 500, 5000)
    register_timing_handlers(engine, 100)
    register_tensorboard_handlers(
        engine, value_net, agent, render_env, 100, 10000, 1000, 50000
    )

    engine.run(
        batch_generator(),
        max_epochs=100,
        epoch_length=10000,
    )
