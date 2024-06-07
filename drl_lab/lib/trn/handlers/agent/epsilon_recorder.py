from ignite.engine import Engine

from drl_lab.lib.rl.actions import EpsilonGreedyActionSelector
from drl_lab.lib.rl.interfaces import Agent


class EpsilonRecorderHandler:
    def __init__(self, agent: Agent):
        self.agent = agent

    def __call__(self, engine: Engine):
        if not isinstance(self.agent.action_selector, EpsilonGreedyActionSelector):
            raise TypeError(
                "EpsilonHandler is only compatible with EpsilonGreedyActionSelector"
            )

        engine.state.epsilon = self.agent.action_selector.epsilon
