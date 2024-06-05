import copy
from typing import Callable

import gymnasium as gym
import torch
import torch.optim as optim
from cpprb import ReplayBuffer
from ignite.engine import Engine, Events
from torch.optim.lr_scheduler import CosineAnnealingLR

from drl_lab.lib.rl.actions import EpsilonGreedyActionSelector
from drl_lab.lib.rl.agents.value_agent import ValueAgent
from drl_lab.lib.rl.experience.transition import (
    EpisodeStatisticsAggregator, TransitionExperienceGenerator)
from drl_lab.lib.trn.decorators.statistics import RecordEpisodeStatistics
from drl_lab.lib.trn.handlers.tensorboard import attach_tensorboard_logger
from drl_lab.projects.loss import dqn_loss
from drl_lab.projects.network import DeepQNetwork
from drl_lab.projects.utils import add_max_to_engine_state, sync_networks


@RecordEpisodeStatistics
def experience_generator():
    for exp in exp_gen:
        yield exp


def batch_generator():
    """Batch generator."""
    for exp in experience_generator():
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

    # Training infrastructure
    agent = ValueAgent(
        action_selector=EpsilonGreedyActionSelector(0.1),
        device=device,
        value_net=value_net,
    )

    exp_gen = TransitionExperienceGenerator(env, agent)

    engine = Engine(process_batch)

    # Handlers
    @engine.on(Events.ITERATION_COMPLETED(every=1000))
    def log_metrics():
        print(
            f"Iteration: {engine.state.iteration}\tLength: {engine.state.ep_lengths}\tReward: {engine.state.ep_returns}"
        )

    experience_generator.set_engine(engine)
    engine.run(
        batch_generator(),
        max_epochs=100,
        epoch_length=10000,
    )
