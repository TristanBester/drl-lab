from drl_lab.lib.experience.types import Experience
import torch
from drl_lab.lib.experience.interface import ExperienceGenerator
from drl_lab.lib.agents.interface import Agent
import gymnasium as gym
import numpy as np
from collections import deque


class TransitionExperienceGenerator(ExperienceGenerator):
    def __init__(self, env: gym.Env, agent: Agent):
        """Constructor."""
        self.agent = agent
        self.env = env

        self.ep_steps = 0
        self.max_ep_steps = -1
        self.step_counter = 0
        self.export_freq = 10
        self.ep_counter = 0
        self.returns_buffer = deque(maxlen=10000)
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
        self.step_counter += 1
        self.ep_steps += 1

        experience = Experience(
            self.obs, action, obs_next, reward, truncated, done, info
        )

        if done:
            self.obs, _ = self.env.reset()
            self.returns_buffer.append(self.ep_returns)

            if self.ep_steps > self.max_ep_steps:
                self.max_ep_steps = self.ep_steps

            self.ep_returns = 0
            self.ep_counter += 1
            self.ep_steps = 0
        else:
            self.obs = obs_next
        return experience

    def export_required(self) -> bool:
        """Returns True if export is required."""
        return self.ep_counter % self.export_freq == 0 and len(self.returns_buffer) > 0

    def export_returns(self) -> np.ndarray:
        """Returns the returns buffer."""
        # TODO: replace with circular buffer as np array
        buffer = np.array(list(self.returns_buffer))
        self.returns_buffer.clear()
        return buffer
