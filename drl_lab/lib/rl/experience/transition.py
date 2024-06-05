import gymnasium as gym
import torch

from drl_lab.lib.rl.agents.interface import Agent
from drl_lab.lib.rl.experience.interface import ExperienceGenerator
from drl_lab.lib.rl.experience.types import Experience


class EpisodeStatisticsAggregator:
    def __init__(self, window_size: int = 2):
        """Constructor."""
        self.alpha = 2 / (window_size + 1)

        self.returns_ema = 0.0
        self.lengths_ema = 0.0

    def record_episode(self, return_: float, length: int):
        """Record the episode."""
        self.returns_ema = self.alpha * return_ + (1 - self.alpha) * self.returns_ema
        self.lengths_ema = self.alpha * length + (1 - self.alpha) * self.lengths_ema

    @property
    def returns(self) -> float:
        """Returns the average return."""
        return self.returns_ema

    @property
    def lengths(self) -> float:
        """Returns the average length."""
        return self.lengths_ema


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

        self.ep_returns += float(reward)
        self.ep_steps += 1

        experience = Experience(
            self.obs,
            action,
            obs_next,
            reward,
            terminated,
            truncated,
            info,
        )

        if terminated or truncated:
            self.obs, _ = self.env.reset()
            self.stat_aggregator.record_episode(self.ep_returns, self.ep_steps)
            self.ep_returns = 0
            self.ep_steps = 0
        else:
            self.obs = obs_next
        return experience
