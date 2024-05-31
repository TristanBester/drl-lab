from drl_lab.lib.experience.types import Experience
from drl_lab.lib.experience.interface import ExperienceGenerator
from drl_lab.lib.agents.interface import Agent
import gymnasium as gym
from collections import deque


class TransitionExperienceGenerator(ExperienceGenerator):
    def __init__(self, env: gym.Env, agent: Agent):
        """Constructor."""
        self.agent = agent
        self.env = env

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
        # action = self.agent(self.obs)
        action = 1
        obs_next, reward, done, truncated, info = self.env.step(action)

        experience = Experience(
            self.obs, action, obs_next, reward, truncated, done, info
        )
        self.obs = obs_next

        if done:
            self.obs, _ = self.env.reset()
            self.returns_buffer.append(self.ep_returns)
            self.ep_returns = 0
            self.ep_counter += 1
        else:
            self.ep_returns += reward
        return experience

    def export_required(self) -> bool:
        """Returns True if export is required."""
        return self.ep_counter % self.export_freq == 0 and len(self.returns_buffer) > 0

    def export_returns(self) -> list[float]:
        """Returns the returns buffer."""
        buffer = list(self.returns_buffer)
        self.returns_buffer.clear()
        return buffer
