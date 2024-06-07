from typing import Callable

import gymnasium as gym
import torch
from ignite.engine import Engine

from drl_lab.lib.rl.interfaces import Agent


class EvalHandler:
    def __init__(
        self,
        agent: Agent,
        env: gym.Env,
        success_transform: Callable | None = None,
        alpha=0.1,
    ):
        self.agent = agent
        self.env = env
        self.success_transform = success_transform
        self.alpha = alpha

    def __call__(self, engine: Engine):
        return_ = self._evaluate()

        if not hasattr(engine.state, "eval_returns"):
            engine.state.eval_returns = 0
        else:
            engine.state.eval_returns = (
                self.alpha * return_ + (1 - self.alpha) * engine.state.eval_returns
            )

        if self.success_transform is not None:
            if not hasattr(engine.state, "eval_success_rate"):
                engine.state.eval_success_rate = 0
            else:
                engine.state.eval_success_rate = (
                    self.alpha * self.success_transform(return_)
                    + (1 - self.alpha) * engine.state.eval_success_rate
                )

    def _evaluate(self):
        obs, _ = self.env.reset()
        done = False
        return_ = 0

        while not done:
            obs = torch.Tensor(obs)
            action = self.agent(obs)
            obs_next, reward, terminated, truncated, _ = self.env.step(action)

            done = terminated or truncated
            obs = obs_next

            return_ += float(reward)
        return return_
