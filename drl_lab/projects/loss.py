import torch
import torch.nn as nn
import torch.nn.functional as F


def dqn_loss(
    batch: dict,
    value_net: nn.Module,
    target_net: nn.Module,
    device: torch.device,
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
