from drl_lab.lib.experience.types import Experience
import torch
from drl_lab.lib.experience.interface import ExperienceGenerator
from drl_lab.lib.agents.interface import Agent
import gymnasium as gym
import numpy as np


class EpisodeStatisticsAggregator:
    def __init__(self, max_length: int = 10000, min_export_size: int = 100):
        self.max_length = max_length
        self.ep_counter = 0
        self.min_export_size = min_export_size
        self._clear_buffers()

    def _clear_buffers(self):
        self.elements_in_buffer = 0
        self.returns_buffer = np.zeros(self.max_length)
        self.lengths_buffer = np.zeros(self.max_length)
        # FPS stuff

    def record_episode(self, return_: float, length: int):
        self.returns_buffer[self.elements_in_buffer % self.max_length] = return_
        self.lengths_buffer[self.elements_in_buffer % self.max_length] = length
        self.elements_in_buffer += 1
        self.ep_counter += 1

    def export_required(self) -> bool:
        return self.elements_in_buffer >= self.min_export_size

    def export(self):
        returns = self.returns_buffer[: self.elements_in_buffer]
        lengths = self.lengths_buffer[: self.elements_in_buffer]
        self._clear_buffers()
        return returns, lengths


class TransitionExperienceGenerator(ExperienceGenerator):
    def __init__(
        self,
        env: gym.Env,
        agent: Agent,
        stat_aggregator: EpisodeStatisticsAggregator,
    ):
        """Constructor."""
        self.agent = agent
        self.env = env

        self.stat_aggregator = stat_aggregator

        self.ep_steps = 0
        self.ep_returns = 0.0

        self.obs, _ = env.reset()

    def __iter__(self):
        """Returns an iterator object."""
        return self

    def __next__(self) -> Experience:
        """Returns the next environmental experience."""
        obs = torch.Tensor(self.obs)
        action = self.agent(obs)
        obs_next, reward, terminated, truncated, info = self.env.step(action)
        done = truncated or terminated

        self.ep_returns += float(reward)
        self.ep_steps += 1

        experience = Experience(
            self.obs, action, obs_next, reward, truncated, done, info
        )

        if done:
            self.obs, _ = self.env.reset()
            self.stat_aggregator.record_episode(self.ep_returns, self.ep_steps)
            self.ep_returns = 0
            self.ep_steps = 0
        else:
            self.obs = obs_next
        return experience
