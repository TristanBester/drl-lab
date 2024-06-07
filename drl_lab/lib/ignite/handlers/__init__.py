import os
from typing import Callable

import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers.tensorboard_logger import (GradsHistHandler,
                                                GradsScalarHandler,
                                                TensorboardLogger,
                                                WeightsHistHandler,
                                                WeightsScalarHandler)
from torch.optim import Optimizer


def register_tb_handlers(
    engine: Engine,
    output_transform: Callable,
    log_dir: str = "logs/",
    log_lr: bool = True,
    log_weight_norm: bool = True,
    log_gradient_norm: bool = True,
    log_weight_hist: bool = True,
    log_gradient_hist: bool = True,
    model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
) -> None:
    """Configure and attach TensorBoard logger to the engine."""
    run_name = _get_next_run_name(log_dir)
    log_dir = os.path.join(log_dir, run_name)
    tb_logger = TensorboardLogger(log_dir=log_dir)

    _attach_output_handler(engine, tb_logger, output_transform)

    if log_lr:
        assert optimizer is not None, "Optimizer must be provided to log learning rate."
        _attach_lr_handler(engine, tb_logger, optimizer)

    if log_weight_norm:
        assert model is not None, "Model must be provided to log weight norm."
        _attach_weight_norm_handler(engine, tb_logger, model)

    if log_gradient_norm:
        assert model is not None, "Model must be provided to log gradient norm."
        _attach_grad_norm_handler(engine, tb_logger, model)

    if log_weight_hist:
        assert model is not None, "Model must be provided to log weight histogram."
        _attach_weight_hist_handler(engine, tb_logger, model)

    if log_gradient_hist:
        assert model is not None, "Model must be provided to log gradient histogram."
        _attach_grad_hist_handler(engine, tb_logger, model)


def _get_next_run_name(log_dir: str):
    """Get the next run name."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    runs = [
        int(run.split("_")[1])
        for run in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, run))
    ]
    if runs:
        return f"run_{max(runs) + 1}"
    return "run_1"


def _attach_output_handler(
    engine: Engine,
    tb_logger: TensorboardLogger,
    output_transform: Callable,
):
    """Attach output handler to the engine."""
    tb_logger.attach_output_handler(
        engine,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=output_transform,
    )


def _attach_lr_handler(
    engine: Engine, tb_logger: TensorboardLogger, optimizer: Optimizer
):
    """Attach learning rate handler to the engine."""
    tb_logger.attach_opt_params_handler(
        engine,
        event_name=Events.ITERATION_STARTED,
        optimizer=optimizer,
        param_name="lr",
    )


def _attach_weight_norm_handler(
    engine: Engine, tb_logger: TensorboardLogger, model: nn.Module
):
    """Attach weight norm handler to the engine."""
    tb_logger.attach(
        engine,
        event_name=Events.ITERATION_COMPLETED,
        log_handler=WeightsScalarHandler(model),
    )


def _attach_weight_hist_handler(
    engine: Engine, tb_logger: TensorboardLogger, model: nn.Module
):
    """Attach weight histogram handler to the engine."""
    tb_logger.attach(
        engine,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=WeightsHistHandler(model),
    )


def _attach_grad_norm_handler(
    engine: Engine, tb_logger: TensorboardLogger, model: nn.Module
):
    """Attach gradient norm handler to the engine."""
    tb_logger.attach(
        engine,
        event_name=Events.ITERATION_COMPLETED,
        log_handler=GradsScalarHandler(model),
    )


def _attach_grad_hist_handler(
    engine: Engine, tb_logger: TensorboardLogger, model: nn.Module
):
    """Attach gradient histogram handler to the engine."""
    tb_logger.attach(
        engine,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=GradsHistHandler(model),
    )
