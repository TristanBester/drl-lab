from drl_lab.lib.experience.transition import TransitionExperienceGenerator
from drl_lab.lib.agents.value_agent import ValueAgent
from drl_lab.lib.actions import EpsilonGreedyActionSelector
import gymnasium as gym
import torch
from cpprb import ReplayBuffer
import copy
import torch.optim as optim
from drl_lab.projects.network import DeepQNetwork


def create_process_function():
    """Factory function.

    This function will be used to create the function called by the ignite engine.
    As the setup of this function is a complex process, we employ the factory patterns.

    All aspects of the processing function must be paramterised: network, env, optim etc.
    This allows a single general function to brige the gap between the objects associated with
    a convetional RL training loop where all objects are directly managed by the user and a training
    loop which is fully managed by the ignite engine.
    """
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

    def process_function(engine, batch):
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

    return process_batch
