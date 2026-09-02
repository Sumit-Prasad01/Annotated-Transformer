import torch
from torch.optim.lr_scheduler import LambdaLR
from utils.logger import logger
from utils.custom_exception import CustomException


def rate(step: int, model_size: int, factor: float, warmup: int) -> float:
    """
    Learning rate schedule function from 'Attention Is All You Need':
    lrate = factor * (d_model ^ -0.5 * min(step ^ -0.5, step * warmup ^ -1.5))
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def get_std_opt(model: torch.nn.Module, d_model: int = 512, factor: float = 1.0, 
                warmup: int = 4000, lr: float = 1.0) -> tuple:
    """
    Construct standard Adam optimizer with LambdaLR learning rate scheduler.
    """
    try:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(step, model_size=d_model, factor=factor, warmup=warmup),
        )
        return optimizer, lr_scheduler
    except Exception as e:
        logger.error("Error constructing standard optimizer/scheduler.")
        raise CustomException("Failed to construct optimizer and scheduler", e)


class DummyOptimizer(torch.optim.Optimizer):
    """Dummy optimizer for evaluation/testing loops."""
    def __init__(self):
        self.param_groups = [{"lr": 0}]

    def step(self):
        pass

    def zero_grad(self, set_to_none=False):
        pass


class DummyScheduler:
    """Dummy scheduler for evaluation/testing loops."""
    def step(self):
        pass
