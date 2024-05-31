from drl_lab.lib.experience.transition import TransitionExperienceGenerator
from drl_lab.lib.agents.value_agent import ValueAgent
from drl_lab.lib.actions import ArgmaxActionSelector, EpsilonGreedyActionSelector
import gymnasium as gym
import torch
import torch.nn as nn
from cpprb import ReplayBuffer


class DeepQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super(DeepQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode=None)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    value_net = DeepQNetwork(obs_dim, n_actions)

    agent = ValueAgent(
        action_selector=EpsilonGreedyActionSelector(epsilon=0.1),
        device=torch.device("cpu"),
        value_net=value_net,
    )
    experience_generator = TransitionExperienceGenerator(env, agent)

    buffer = ReplayBuffer(
        size=1000,
        env_dict={
            "obs": {"shape": obs_dim},
            "action": {"shape": 1},
            "obs_next": {"shape": obs_dim},
            "reward": {},
            "truncated": {},
            "done": {},
        },
    )

    for exp in experience_generator:
        buffer.add(
            obs=exp.obs,
            action=exp.action,
            obs_next=exp.obs_next,
            reward=exp.reward,
            truncated=exp.truncated,
            done=exp.done,
        )

        if experience_generator.ep_counter > 10:
            print(buffer.sample(2))
            break

    #    if experience_generator.export_required():
    #        returns = experience_generator.export_returns()
    #        print(f"Episodes: {experience_generator.ep_counter}, Returns: {returns}")
    #    if experience_generator.ep_counter > 100:
    #        break
