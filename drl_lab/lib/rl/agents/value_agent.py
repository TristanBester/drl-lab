import torch
import torch.nn as nn

from drl_lab.lib.rl.interfaces import ActionSelector, Agent


class ValueAgent(Agent):
    def __init__(
        self,
        action_selector: ActionSelector,
        device: torch.device,
        value_net: nn.Module,
    ) -> None:
        self.action_selector = action_selector
        self.device = device
        self.value_net = value_net

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device)
        q_values = self.value_net(obs)
        actions = self.action_selector(q_values)
        actions = actions.cpu().numpy()
        return actions
