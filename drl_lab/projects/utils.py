import torch.nn as nn


def sync_networks(value_net: nn.Module, target_net: nn.Module):
    """Sync the weights between two networks."""
    target_net.load_state_dict(value_net.state_dict())
