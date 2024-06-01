from inspect import getmro
import time

from torch._dynamo.utils import object_has_getattribute
from drl_lab.lib.experience.transition import TransitionExperienceGenerator
from drl_lab.lib.experience.interface import ExperienceGenerator
import torch.nn.functional as F
from drl_lab.lib.agents.value_agent import ValueAgent
from drl_lab.lib.actions import ArgmaxActionSelector, EpsilonGreedyActionSelector
import gymnasium as gym
import torch
import torch.nn as nn
from cpprb import ReplayBuffer
import copy
import torch.optim as optim
from ignite.engine import Engine, Events, EventEnum, State


class EpisodeReturnsEvents(EventEnum):
    EPISODE_COMPLETED = "episode_completed"
    OTHER_EVENT = "other_event"


class EpisodeReturnsHandler:
    def __init__(self, exp_gen: ExperienceGenerator):
        self.exp_gen = exp_gen

    def __call__(self, engine: Engine):
        # TODO: Make this less disgusting...
        # TODO: Can we make the exp_gen only track current episode stats and remove all complexity from theses classes?
        print(f"called with ep_conter: {self.exp_gen.ep_counter}")
        if not self.exp_gen.export_required():
            return

        returns = self.exp_gen.export_returns()
        print(returns)
        for return_ in returns:
            # NOTE: I think this is wack - not in docs
            # Just set public attributes dynamcially on engine state
            engine.state.episode = getattr(engine.state, "episode", 0) + 1
            engine.state.metrics["return"] = return_
            engine.fire_event(EpisodeReturnsEvents.EPISODE_COMPLETED)

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        engine.register_events(*EpisodeReturnsEvents)
        State.event_to_attr[EpisodeReturnsEvents.EPISODE_COMPLETED] = "episode"


class DeepQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sync_networks(value_net: nn.Module, target_net: nn.Module):
    """Sync the weights between two networks."""
    target_net.load_state_dict(value_net.state_dict())


def dqn_loss(
    batch: dict, value_net: nn.Module, target_net: nn.Module, device: torch.device
) -> torch.Tensor:
    """Compute the DQN loss."""
    obs = torch.tensor(batch["obs"], dtype=torch.float32).to(device)
    action = torch.tensor(batch["action"], dtype=torch.long).to(device)
    obs_next = torch.tensor(batch["obs_next"], dtype=torch.float32).to(device)
    reward = torch.tensor(batch["reward"], dtype=torch.float32).to(device)
    truncated = torch.tensor(batch["truncated"], dtype=torch.bool).to(device)
    done = torch.tensor(batch["done"], dtype=torch.bool).to(device)

    # Estimated Q-values
    q_values = value_net(obs)
    selected_q_values = q_values.gather(dim=1, index=action)

    # Target Q-values
    next_q_values = target_net(obs_next)
    max_next_q_values = next_q_values.max(dim=1)[0].unsqueeze(1)

    # Override the target Q-values for the terminal states
    # terminal_mask = (done & ~truncated).flatten()
    # max_next_q_values[terminal_mask] = 0.0
    max_next_q_values[done] = 0

    # Compute loss
    td_target = reward + 0.99 * max_next_q_values
    loss = F.mse_loss(selected_q_values, td_target)
    return loss


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
    experience_generator = TransitionExperienceGenerator(env, agent)
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
        print("processing batch...")
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

    # Create the engine
    engine = Engine(process_batch)

    # Add the event handlers to the engine
    handler = EpisodeReturnsHandler(exp_gen=experience_generator)
    handler.attach(engine)

    # The add_event_handler syntax will probably be cleaner here
    # Then we can move all handlers somewhere else and just attach them to the engine at runtime
    @engine.on(EpisodeReturnsEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        ep_return = trainer.state.metrics["return"]
        print(f"Episode {trainer.state.episode} completed with return {ep_return}")
        time.sleep(3)

    # Start the engine
    def batch_generator(exp_gen: ExperienceGenerator, buffer: ReplayBuffer):
        for exp in exp_gen:
            buffer.add(
                obs=exp.obs,
                action=exp.action,
                obs_next=exp.obs_next,
                reward=exp.reward,
                truncated=exp.truncated,
                done=exp.done,
            )

            if buffer.get_stored_size() < 32:
                continue
            yield buffer.sample(32)

    engine.run(batch_generator(experience_generator, buffer), max_epochs=100000)
