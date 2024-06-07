import copy
import functools
from typing import Callable

import gymnasium as gym
import torch
import torch.optim as optim
from cpprb import ReplayBuffer
from ignite.engine import Engine
from torch.optim.lr_scheduler import CosineAnnealingLR

from drl_lab.lib.core.actions import EpsilonGreedyActionSelector
from drl_lab.lib.core.agents.value_agent import ValueAgent
from drl_lab.lib.core.experience.transition import (
    EpisodeStatisticsAggregator, TransitionExperienceGenerator)
from drl_lab.lib.ignite.handlers.tensorboard import attach_tensorboard_logger
from drl_lab.projects.loss import dqn_loss
from drl_lab.projects.network import DeepQNetwork
from drl_lab.projects.utils import add_max_to_engine_state, sync_networks


class RecordStatistics:
    def __init__(self, func, alpha=0.1, record_epsilon=True):
        self.func = func
        self.alpha = alpha
        self.record_epsilon = record_epsilon
        self.engine = None

        functools.update_wrapper(self, func)

    def set_engine(self, engine):
        self.engine = engine

        self.engine.ep_returns = 0
        self.engine.ep_steps = 0

        if self.record_epsilon:
            self.engine.epsilon = 0

    def __call__(self, *args, **kwargs):
        if self.engine is None:
            raise ValueError("Engine set before calling the generator.")

        ep_steps = 0
        ep_returns = 0

        for exp in self.func(*args, **kwargs):
            ep_steps += 1
            ep_returns += exp.reward

            if exp.terminated or exp.truncated:
                self.engine.ep_steps = (
                    self.alpha * ep_steps + (1 - self.alpha) * self.engine.ep_steps
                )
                self.engine.ep_returns = (
                    self.alpha * ep_returns + (1 - self.alpha) * self.engine.ep_returns
                )

                if self.record_epsilon:
                    pass

                ep_steps = 0
                ep_returns = 0

            yield exp


@RecordStatistics
def batch_generator():
    """this is a generator function."""
    for exp in exp_gen:
        yield exp


class Engine:
    def __init__(self):
        self.name = 1


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    engine = Engine()

    # Networks
    device = torch.device("cpu")
    value_net = DeepQNetwork(obs_dim, n_actions)

    # Training infrastructure
    agent = ValueAgent(
        action_selector=EpsilonGreedyActionSelector(0.1),
        device=device,
        value_net=value_net,
    )
    exp_gen = TransitionExperienceGenerator(env, agent)

    batch_generator.set_engine(engine)

    for i, j in enumerate(batch_generator()):
        if i == 100:
            break
        if i % 10 == 0:
            print(engine.ep_steps, engine.ep_returns)
