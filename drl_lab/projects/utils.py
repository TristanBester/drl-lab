import torch.nn as nn
from typing import Any
from ignite.engine import Engine


def sync_networks(value_net: nn.Module, target_net: nn.Module):
    """Sync the weights between two networks."""
    target_net.load_state_dict(value_net.state_dict())


def add_max_to_engine_state(engine: Engine, attr: str, value: Any):
    """Add a new attribute to the engine state."""
    if not hasattr(engine.state, attr):
        setattr(engine.state, attr, value)
    else:
        setattr(
            engine.state,
            attr,
            max(getattr(engine.state, attr), value),
        )
