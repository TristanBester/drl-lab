import copy

import gymnasium as gym
import torch
import torch.optim as optim
from cpprb import ReplayBuffer
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint
from ignite.handlers.tensorboard_logger import (GradsHistHandler,
                                                GradsScalarHandler,
                                                TensorboardLogger,
                                                WeightsHistHandler,
                                                WeightsScalarHandler)
from torch.optim.lr_scheduler import CosineAnnealingLR

from drl_lab.lib.rl.actions import EpsilonGreedyActionSelector
from drl_lab.lib.rl.agents.value_agent import ValueAgent
from drl_lab.lib.rl.experience.transition import TransitionExperienceGenerator
from drl_lab.lib.trn.handlers import tensorboard as tbh
from drl_lab.lib.trn.handlers.utils import global_step_transform
from drl_lab.lib.trn.wrappers import (InteractionTimingHandler,
                                      RecordEpisodeStatistics)
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
    scheduler.step()

    if engine.state.iteration % 1000 == 0:
        sync_networks(value_net, target_net)

    engine.state.metrics["loss"] = loss.item()


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
    optimizer = optim.Adam(value_net.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=100000, eta_min=0.00001)

    # Reinforcement Learning
    engine = Engine(process_batch)
    agent = ValueAgent(
        action_selector=EpsilonGreedyActionSelector(0.1),
        device=device,
        value_net=value_net,
    )
    transition_generator = TransitionExperienceGenerator(env, agent)
    exp_gen = RecordEpisodeStatistics(transition_generator, engine)
    exp_gen = InteractionTimingHandler(exp_gen, engine)

    # Call attach logging handlers()
    # Then pass hyra config to the logging config attach
    # This will specify how all the loggers should be configured

    # Logging
    tb_logger = TensorboardLogger(log_dir="logs")

    tb_logger.attach(
        engine,
        log_handler=tbh.ScalarHandler(
            "ep_returns", global_step_transform, "1-training"
        ),
        event_name=Events.ITERATION_COMPLETED(every=100),
    )
    tb_logger.attach(
        engine,
        log_handler=tbh.ScalarHandler(
            "ep_lengths", global_step_transform, "1-training"
        ),
        event_name=Events.ITERATION_COMPLETED(every=100),
    )
    tb_logger.attach(
        engine,
        log_handler=tbh.ScalarHandler(
            "interaction_time", global_step_transform, "1-training"
        ),
        event_name=Events.ITERATION_COMPLETED(every=100),
    )

    tb_logger.attach_output_handler(
        engine,
        event_name=Events.ITERATION_COMPLETED(every=100),
        tag="1-training",
        metric_names=["loss"],
    )
    tb_logger.attach(
        engine,
        log_handler=GradsScalarHandler(value_net, tag="2-gradients"),
        event_name=Events.ITERATION_COMPLETED(every=10000),
    )
    tb_logger.attach(
        engine,
        log_handler=GradsHistHandler(value_net, tag="2-gradients"),
        event_name=Events.ITERATION_COMPLETED(every=10000),
    )
    tb_logger.attach(
        engine,
        log_handler=WeightsScalarHandler(value_net, tag="3-weights"),
        event_name=Events.ITERATION_COMPLETED(every=10000),
    )
    tb_logger.attach(
        engine,
        log_handler=WeightsHistHandler(value_net, tag="3-weights"),
        event_name=Events.ITERATION_COMPLETED(every=10000),
    )
    tb_logger.attach_opt_params_handler(
        engine,
        event_name=Events.ITERATION_COMPLETED(every=10000),
        optimizer=optimizer,
        param_name="lr",
        tag="1-training",
    )

    # Checkpointing
    # engine.add_event_handler(
    #    event_name=Events.ITERATION_COMPLETED(every=10000),
    #    handler=Checkpoint(
    #        to_save={"model": value_net, "optimizer": optimizer},
    #        save_handler="checkpoints/",
    #        n_saved=3,
    #        global_step_transform=global_step_transform,
    #    ),
    # )

    engine.run(
        batch_generator(),
        max_epochs=100,
        epoch_length=10000,
    )
