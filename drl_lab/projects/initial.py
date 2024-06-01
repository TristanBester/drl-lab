from drl_lab.lib.experience.transition import TransitionExperienceGenerator
import torch.nn.functional as F
from drl_lab.lib.agents.value_agent import ValueAgent
from drl_lab.lib.actions import ArgmaxActionSelector, EpsilonGreedyActionSelector
import gymnasium as gym
import torch
import torch.nn as nn
from cpprb import ReplayBuffer
import copy
import torch.optim as optim


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
    env = gym.make("CartPole-v1", render_mode=None)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device = torch.device("cpu")
    value_net = DeepQNetwork(obs_dim, n_actions)
    target_net = copy.deepcopy(value_net)

    agent = ValueAgent(
        action_selector=EpsilonGreedyActionSelector(epsilon=0.1),
        device=torch.device("cpu"),
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

    for exp in experience_generator:
        buffer.add(
            obs=exp.obs,
            action=exp.action,
            obs_next=exp.obs_next,
            reward=exp.reward,
            truncated=exp.truncated,
            done=exp.done,
        )

        if buffer.get_stored_size() < 1000:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(32)
        loss = dqn_loss(batch, value_net, target_net, device)
        loss.backward()
        optimizer.step()

        if experience_generator.step_counter % 1000 == 0:
            print(f"Syning networks at step {experience_generator.step_counter}")
            sync_networks(value_net, target_net)

        if experience_generator.export_required():
            returns = experience_generator.export_returns().mean()
            eps = experience_generator.ep_counter
            print(
                f"Episode: {eps}, Returns: {returns}, Max steps: {experience_generator.max_ep_steps}"
            )
