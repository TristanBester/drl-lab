import copy
from typing import Callable

import gymnasium as gym
import torch
import torch.optim as optim
from cpprb import ReplayBuffer
from ignite.engine import Engine, Events
from ignite.handlers import (TensorboardLogger, WandBLogger,
                             global_step_from_engine)
from ignite.handlers.base_logger import BaseHandler
from torch.optim.lr_scheduler import CosineAnnealingLR
from wandb.sdk import wandb_watch

from drl_lab.lib.rl.actions import EpsilonGreedyActionSelector
from drl_lab.lib.rl.agents.value_agent import ValueAgent
from drl_lab.lib.rl.experience.transition import TransitionExperienceGenerator
from drl_lab.lib.trn.wrappers.ep_stats import RecordEpisodeStatistics
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

    return {
        "loss": loss.item(),
    }


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

    agent = ValueAgent(
        action_selector=EpsilonGreedyActionSelector(0.1),
        device=device,
        value_net=value_net,
    )

    engine = Engine(process_batch)

    exp_gen = RecordEpisodeStatistics(
        TransitionExperienceGenerator(env, agent),
        engine,
    )

    # tb_logger = TensorboardLogger(log_dir="logs/run_1")

    # tb_logger.attach(
    #    engine,
    #    log_handler=ScalarHandler(scalar_name="ep_returns"),
    #    event_name=Events.ITERATION_COMPLETED(every=1000),
    # )

    wandb_logger = WandBLogger(project="drl_lab")

    wandb_logger.attach(
        engine,
        log_handler=ScalarHandler(scalar_name="ep_returns"),
        event_name=Events.ITERATION_COMPLETED(every=1000),
    )

    engine.run(batch_generator())
