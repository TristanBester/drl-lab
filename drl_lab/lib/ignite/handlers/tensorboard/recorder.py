from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from ignite.engine import Engine, Events
from ignite.handlers import TensorboardLogger
from ignite.handlers.base_logger import BaseHandler

from drl_lab.lib.core.interfaces import Agent


class RecordEpisodeHandler(BaseHandler):
    def __init__(self, agent: Agent, env: gym.Env, global_step_transform: Callable):
        self.agent = agent
        self.env = env
        self.global_step_transform = global_step_transform

    def __call__(
        self,
        engine: Engine,
        logger: TensorboardLogger,
        event_name: Events | str,
    ):
        frames = []
        obs, _ = self.env.reset()
        done = False
        frames.append(self.env.render())

        while not done:
            obs = torch.Tensor(obs)
            action = self.agent(obs)
            obs_next, _, terminated, truncated, _ = self.env.step(action)

            done = terminated or truncated
            obs = obs_next

            frames.append(self.env.render())

        frames = np.array(frames)
        frames = frames.transpose(0, 3, 1, 2)
        frames = np.expand_dims(frames, axis=0)

        logger.writer.add_video(
            tag="episode",
            vid_tensor=frames,
            global_step=self.global_step_transform(engine, event_name),
        )

        del engine, event_name
