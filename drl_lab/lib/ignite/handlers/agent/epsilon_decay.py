from ignite.engine import Engine

from drl_lab.lib.core.actions import EpsilonGreedyActionSelector
from drl_lab.lib.core.interfaces import Agent


class EpsilonDecayHandler:
    def __init__(self, agent: Agent):
        self.agent = agent

    def __call__(self, _: Engine):
        if not isinstance(self.agent.action_selector, EpsilonGreedyActionSelector):
            raise TypeError(
                "EpsilonHandler is only compatible with EpsilonGreedyActionSelector"
            )

        self.agent.action_selector.decay_epsilon()
