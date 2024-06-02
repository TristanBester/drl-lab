from os import statvfs_result

from torch.types import Storage
from drl_lab.lib.experience.transition import TransitionExperienceGenerator
from drl_lab.lib.agents.value_agent import ValueAgent
from drl_lab.lib.actions import EpsilonGreedyActionSelector
import gymnasium as gym
import torch
from cpprb import ReplayBuffer, train
import copy
import torch.optim as optim
from drl_lab.projects.network import DeepQNetwork
from drl_lab.lib.experience.transition import EpisodeStatisticsAggregator
from drl_lab.projects.utils import sync_networks
from drl_lab.projects.loss import dqn_loss

from ignite.engine import Engine, Events


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = torch.device("cpu")
    value_net = DeepQNetwork(obs_dim, n_actions)
    target_net = copy.deepcopy(value_net)

    agent = ValueAgent(
        action_selector=EpsilonGreedyActionSelector(0.1),
        device=device,
        value_net=value_net,
    )

    stats_aggregator = EpisodeStatisticsAggregator()
    exp_gen = TransitionExperienceGenerator(env, agent, stats_aggregator)
    buffer = ReplayBuffer(
        size=100000,
        env_dict={
            "obs": {"shape": obs_dim},
            "action": {"shape": 1},
            "obs_next": {"shape": obs_dim},
            "reward": {},
            "truncated": {},
            "done": {},
        },
    )
    optimizer = optim.Adam(value_net.parameters(), lr=0.0001)

    def process_batch(engine, batch):
        """Process one batch of data."""
        optimizer.zero_grad()
        loss = dqn_loss(batch, value_net, target_net, device)
        loss.backward()
        optimizer.step()

        if engine.state.iteration % 1000 == 0:
            sync_networks(value_net, target_net)

        return {
            "loss": loss.item(),
            "epsilon": agent.action_selector.epsilon,
        }

    def batch_generator():
        for exp in exp_gen:
            buffer.add(
                obs=exp.obs,
                action=exp.action,
                obs_next=exp.obs_next,
                reward=exp.reward,
                truncated=exp.truncated,
                done=exp.done,
            )

            if buffer.get_stored_size() > 1000:
                yield buffer.sample(32)

    trainer = Engine(process_batch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def on_epoch_completed(engine):
        print(f"Epoch {engine.state.epoch} completed")
        if stats_aggregator.export_required():
            returns, lengths = stats_aggregator.export()
            print(
                f"Epoch {engine.state.epoch} complete. Mean return: {returns.mean()}. Mean length: {lengths.mean()}"
            )

    trainer.run(batch_generator(), max_epochs=10, epoch_length=10000)
