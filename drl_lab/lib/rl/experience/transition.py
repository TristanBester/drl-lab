import gymnasium as gym
import torch

from drl_lab.lib.rl.interfaces import Agent, ExperienceGenerator
from drl_lab.lib.rl.types import Experience


class TransitionExperienceGenerator(ExperienceGenerator):
    def __init__(
        self,
        env: gym.Env,
        agent: Agent,
    ):
        """Constructor."""
        self.agent = agent
        self.env = env

        self.obs, _ = env.reset()

    def __iter__(self):
        """Returns an iterator object."""
        return self

    def __next__(self) -> Experience:
        """Returns the next environmental experience."""
        obs = torch.Tensor(self.obs)
        action = self.agent(obs)
        obs_next, reward, terminated, truncated, info = self.env.step(action)

        experience = Experience(
            self.obs,
            action,
            obs_next,
            float(reward),
            terminated,
            truncated,
            info,
        )

        if terminated or truncated:
            self.obs, _ = self.env.reset()
        else:
            self.obs = obs_next
        return experience
