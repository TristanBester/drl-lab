import copy
from typing import Callable

import gymnasium as gym
import torch
import torch.optim as optim
from cpprb import ReplayBuffer
from ignite.engine import Engine
from torch.optim.lr_scheduler import CosineAnnealingLR

from drl_lab.lib.rl.actions import EpsilonGreedyActionSelector
from drl_lab.lib.rl.agents.value_agent import ValueAgent
from drl_lab.lib.rl.experience.transition import (
    EpisodeStatisticsAggregator, TransitionExperienceGenerator)
from drl_lab.lib.trn.handlers.tensorboard import attach_tensorboard_logger
from drl_lab.projects.loss import dqn_loss
from drl_lab.projects.network import DeepQNetwork
from drl_lab.projects.utils import add_max_to_engine_state, sync_networks


def create_batch_processor() -> Callable:
    """Factory function for the DQN batch processor."""

    def process_batch(engine: Engine, batch: dict):
        optimizer.zero_grad()
        loss = dqn_loss(
            batch,
            value_net,
            target_net,
            device=device,
        )
        loss.backward()
        optimizer.step()
        # We should probably step on epoch level
        # We can the also step on val loss
        scheduler.step()

        add_max_to_engine_state(engine, "max_return", stats_aggregator.returns)

        if engine.state.iteration % 1000 == 0:
            sync_networks(value_net, target_net)

        return {
            "loss": loss.item(),
            "epsilon": agent.action_selector.epsilon,
            "returns": stats_aggregator.returns,
            "lengths": stats_aggregator.lengths,
        }

    return process_batch


def create_batch_generator() -> Callable:
    """Factory function for the DQN batch generator."""

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

    return batch_generator


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

    # Training infrastructure
    agent = ValueAgent(
        action_selector=EpsilonGreedyActionSelector(0.1),
        device=device,
        value_net=value_net,
    )
    stats_aggregator = EpisodeStatisticsAggregator()
    exp_gen = TransitionExperienceGenerator(env, agent, stats_aggregator)

    # Setup engine
    batch_generator = create_batch_generator()
    batch_processor = create_batch_processor()
    engine = Engine(batch_processor)

    # Setup tensorboard logging
    attach_tensorboard_logger(
        engine=engine,
        log_dir="logs/",
        model=value_net,
        optimizer=optimizer,
        output_transform=lambda output: {
            "loss": output["loss"],
            "epsilon": output["epsilon"],
            "returns": output["returns"],
            "lengths": output["lengths"],
        },
    )

    # Run trainin
    engine.run(
        batch_generator(),
        max_epochs=100,
        epoch_length=10000,
    )
