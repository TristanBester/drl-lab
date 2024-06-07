import copy

import gymnasium as gym
import torch
import torch.optim as optim
from cpprb import ReplayBuffer
from ignite.engine import Engine

from drl_lab.lib.core.actions import (ArgmaxActionSelector,
                                      EpsilonGreedyActionSelector)
from drl_lab.lib.core.agents.value_agent import ValueAgent
from drl_lab.lib.core.experience.transition import \
    TransitionExperienceGenerator
from drl_lab.lib.ignite.wrappers import (InteractionTimingHandler,
                                         RecordEpisodeStatistics)
from drl_lab.projects.model.loss import dqn_loss
from drl_lab.projects.model.network import DeepQNetwork
from drl_lab.projects.model.utils import sync_networks
from drl_lab.projects.train.handlers import (register_checkpointing_handlers,
                                             register_epsilon_handlers,
                                             register_eval_handlers,
                                             register_tensorboard_handlers,
                                             register_timing_handlers)


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

    if engine.state.iteration % 1000 == 0:
        sync_networks(value_net, target_net)

    engine.state.metrics["loss"] = loss.item()


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
    optimizer = optim.Adam(value_net.parameters(), lr=0.0001)

    # Reinforcement Learning
    engine = Engine(process_batch)
    agent = ValueAgent(
        action_selector=EpsilonGreedyActionSelector(
            start_val=1.0,
            end_val=0.01,
            steps=250000,
        ),
        device=device,
        value_net=value_net,
    )
    eval_agent = ValueAgent(
        action_selector=ArgmaxActionSelector(),
        device=device,
        value_net=value_net,
    )

    eval_env = gym.make("CartPole-v1")
    render_env = gym.make("CartPole-v1", render_mode="rgb_array")
    transition_generator = TransitionExperienceGenerator(env, agent)
    exp_gen = RecordEpisodeStatistics(transition_generator, engine)
    exp_gen = InteractionTimingHandler(exp_gen, engine)

    # Ignite Handlers
    register_epsilon_handlers(engine, agent, 100, 1)
    register_checkpointing_handlers(
        engine, {"model": value_net, "optimizer": optimizer}, 50000
    )
    register_eval_handlers(engine, eval_agent, eval_env, 500, 5000)
    register_timing_handlers(engine, 100)
    register_tensorboard_handlers(
        engine, value_net, agent, render_env, 100, 10000, 1000, 50000
    )

    engine.run(
        batch_generator(),
        max_epochs=100,
        epoch_length=10000,
    )
