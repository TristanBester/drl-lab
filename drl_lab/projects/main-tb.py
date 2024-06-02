from os import stat, statvfs_result

from torch.types import Storage
from drl_lab.lib.experience.transition import TransitionExperienceGenerator
from drl_lab.lib.agents.value_agent import ValueAgent
from drl_lab.lib.actions import EpsilonGreedyActionSelector
import gymnasium as gym
import torch
from cpprb import ReplayBuffer, train
import copy
import torch.optim as optim
from drl_lab.projects.network import DeepQNetwork
from drl_lab.lib.experience.transition import EpisodeStatisticsAggregator
from drl_lab.projects.utils import sync_networks
from drl_lab.projects.loss import dqn_loss
from ignite.handlers.tensorboard_logger import *
from ignite.handlers import Checkpoint, global_step_from_engine, Timer, EarlyStopping
from torch.optim.lr_scheduler import StepLR
from ignite.engine import Engine, Events


class TimingHandler:
    def __init__(self, timer: Timer, call_freq: int = 100):
        self.timer = timer
        self.last_time = 0
        self.call_freq = call_freq

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name) -> None:
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError(
                "Handler 'OutputHandler' works only with TensorboardLogger"
            )

        time_delta = self.timer.value() - self.last_time
        self.last_time = self.timer.value()
        steps_per_sec = 1.0 / time_delta * self.call_freq

        global_step = global_step_from_engine(trainer)(None, event_name)
        logger.writer.add_scalar("time", steps_per_sec, global_step)


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

    stats_aggregator = EpisodeStatisticsAggregator()
    exp_gen = TransitionExperienceGenerator(env, agent, stats_aggregator)
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
    optimizer = optim.Adam(value_net.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10_000, gamma=0.5)

    def process_batch(engine, batch):
        """Process one batch of data."""
        optimizer.zero_grad()
        loss = dqn_loss(batch, value_net, target_net, device)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if engine.state.iteration % 1000 == 0:
            sync_networks(value_net, target_net)

        if not hasattr(engine.state, "max_return"):
            engine.state.max_return = 0
        else:
            engine.state.max_return = max(
                engine.state.max_return, stats_aggregator.returns
            )

        # agent.action_selector.epsilon = max(
        #    0.01, agent.action_selector.epsilon - 0.000
        # )

        # Should be something like for stat in stats add to response
        return {
            "loss": loss.item(),
            "epsilon": agent.action_selector.epsilon,
            "returns": stats_aggregator.returns,
            "lengths": stats_aggregator.lengths,
        }

    def batch_generator():
        for exp in exp_gen:
            buffer.add(
                obs=exp.obs,
                action=exp.action,
                obs_next=exp.obs_next,
                reward=exp.reward,
                truncated=exp.truncated,
                done=exp.done,
            )

            if buffer.get_stored_size() > 1000:
                yield buffer.sample(32)

    trainer = Engine(process_batch)

    tb_logger = TensorboardLogger(log_dir="logs/run_1")

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda output: {
            "loss": output["loss"],
            "epsilon": output["epsilon"],
            "returns": output["returns"],
            "lengths": output["lengths"],
        },
    )
    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
    tb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_STARTED,
        optimizer=optimizer,
        param_name="lr",  # optional
    )

    # Attach the logger to the trainer to log model's weights norm after each iteration
    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        log_handler=WeightsScalarHandler(value_net),
    )

    # Attach the logger to the trainer to log model's weights as a histogram after each epoch
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=WeightsHistHandler(value_net),
    )

    # Attach the logger to the trainer to log model's gradients norm after each iteration
    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        log_handler=GradsScalarHandler(value_net),
    )

    # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=GradsHistHandler(value_net),
    )

    # Timing
    elapsed_time = Timer()

    elapsed_time.attach(
        trainer,
        start=Events.STARTED,  # Start timer at the beginning of training
        resume=Events.ITERATION_STARTED,  # Resume timer at the beginning of each epoch
        pause=Events.ITERATION_COMPLETED,  # Pause timer at the end of each epoch
        step=Events.ITERATION_COMPLETED,  # Step (update) timer at the end of each epoch
    )

    tb_logger.attach(
        trainer,
        log_handler=TimingHandler(elapsed_time),
        event_name=Events.ITERATION_COMPLETED(every=100),
    )

    # Model checkpointing
    to_save = {"model": value_net, "optimizer": optimizer, "trainer": trainer}
    checkpoint_dir = "checkpoints/"

    checkpoint = Checkpoint(
        to_save,
        checkpoint_dir,
        n_saved=2,
        global_step_transform=global_step_from_engine(trainer),
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)

    # Early stopping
    handler = EarlyStopping(
        patience=3,
        score_function=lambda e: e.state.max_return,
        trainer=trainer,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

    trainer.run(batch_generator(), max_epochs=100, epoch_length=10000)
    tb_logger.close()
