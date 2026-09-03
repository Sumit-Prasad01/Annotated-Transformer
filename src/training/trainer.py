import time
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from utils.logger import logger
from utils.custom_exception import CustomException
from src.training.scheduler import DummyOptimizer, DummyScheduler


@dataclass
class TrainState:
    """Track training metrics and steps."""
    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


def run_epoch(
    data_iter,
    model: nn.Module,
    loss_compute,
    optimizer,
    scheduler,
    mode: str = "train",
    accum_iter: int = 1,
    train_state: TrainState = TrainState(),
    device: torch.device = None,
) -> tuple:
    """
    Train a single epoch over data_iter using model and loss_compute.
    Handles gradient accumulation across `accum_iter` batches.
    """
    try:
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        n_accum = 0

        for i, batch in enumerate(data_iter):
            if device is not None:
                batch.src = batch.src.to(device)
                batch.src_mask = batch.src_mask.to(device)
                if batch.tgt is not None:
                    batch.tgt = batch.tgt.to(device)
                    batch.tgt_y = batch.tgt_y.to(device)
                    batch.tgt_mask = batch.tgt_mask.to(device)

            out = model.forward(
                batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
            )
            loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

            if mode == "train" or mode == "train_cum":
                loss_node.backward()
                train_state.step += 1
                train_state.samples += batch.src.shape[0]
                train_state.tokens += batch.ntokens
                if i % accum_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    n_accum += 1
                    train_state.accum_step += 1
                scheduler.step()

            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens

            if i % 40 == 1 and (mode == "train" or mode == "train_cum"):
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                logger.info(
                    f"Epoch Step: {i:6d} | Accum Step: {n_accum:6d} | "
                    f"Loss: {loss / batch.ntokens:6.2f} | "
                    f"Tokens / Sec: {tokens / elapsed:7.1f} | Learning Rate: {lr:6.1e}"
                )
                start = time.time()
                tokens = 0

            del out
            del loss
            del loss_node

        return total_loss / total_tokens, train_state
    except Exception as e:
        logger.error(f"Error during epoch execution (mode={mode}).")
        raise CustomException("Failed during epoch execution", e)


def train_model(
    model: nn.Module,
    train_dataloader,
    val_dataloader,
    loss_compute,
    val_loss_compute,
    optimizer,
    scheduler,
    num_epochs: int = 8,
    accum_iter: int = 10,
    save_path: str = "outputs/model.pt",
    device: torch.device = None,
    use_mlflow: bool = True,
) -> nn.Module:
    """
    Main single-GPU / CPU model training orchestration loop with MLflow metric logging.
    """
    try:
        if device is not None:
            model = model.to(device)

        train_state = TrainState()
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            logger.info(f"--- Epoch {epoch + 1}/{num_epochs} ---")
            model.train()
            train_loss, train_state = run_epoch(
                data_iter=train_dataloader,
                model=model,
                loss_compute=loss_compute,
                optimizer=optimizer,
                scheduler=scheduler,
                mode="train",
                accum_iter=accum_iter,
                train_state=train_state,
                device=device,
            )

            model.eval()
            with torch.no_grad():
                val_loss, _ = run_epoch(
                    data_iter=val_dataloader,
                    model=model,
                    loss_compute=val_loss_compute,
                    optimizer=DummyOptimizer(),
                    scheduler=DummyScheduler(),
                    mode="eval",
                    device=device,
                )

            val_loss = float(val_loss)
            try:
                val_perplexity = math.exp(val_loss)
            except OverflowError:
                val_perplexity = float("inf")

            logger.info(
                f"Epoch {epoch + 1} Complete -> Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Perplexity (PPL): {val_perplexity:.4f}"
            )

            # Log metrics to MLflow if active
            if use_mlflow:
                try:
                    import mlflow
                    if mlflow.active_run():
                        current_lr = optimizer.param_groups[0]["lr"]
                        mlflow.log_metric("train_loss", float(train_loss), step=epoch + 1)
                        mlflow.log_metric("val_loss", val_loss, step=epoch + 1)
                        mlflow.log_metric("val_perplexity", val_perplexity, step=epoch + 1)
                        mlflow.log_metric("learning_rate", float(current_lr), step=epoch + 1)
                except Exception as ml_err:
                    logger.warning(f"Could not log metrics to MLflow: {ml_err}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                logger.info(f"Saved best model checkpoint to {save_path}")

        return model
    except Exception as e:
        logger.error("Error in train_model training loop.")
        raise CustomException("Failed during model training", e)
