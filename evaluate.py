import os
import argparse
import torch
from utils.helper import read_yaml
from utils.logger import logger
from utils.custom_exception import CustomException
from src.data import create_dataloaders, Vocab
from src.models import make_model
from src.training import LabelSmoothing, SimpleLossCompute, run_epoch, DummyOptimizer, DummyScheduler


def evaluate(config_path: str = "config/config.yaml", checkpoint_path: str = None):
    try:
        config = read_yaml(config_path)
        output_dir = config.get("output_dir", "outputs")
        vocab_path = config.get("vocab_path", os.path.join(output_dir, "vocab.pt"))

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")

        vocab_data = torch.load(vocab_path)
        vocab_src = Vocab(vocab_data["src_stoi"], vocab_data["src_itos"])
        vocab_tgt = Vocab(vocab_data["tgt_stoi"], vocab_data["tgt_itos"])

        if checkpoint_path is None:
            checkpoint_path = os.path.join(output_dir, f"{config.get('checkpoint_prefix', 'model_')}best.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        model = make_model(
            src_vocab=len(vocab_src),
            tgt_vocab=len(vocab_tgt),
            N=config.get("num_layers", 6),
            d_model=config.get("d_model", 512),
            d_ff=config.get("d_ff", 2048),
            h=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1),
        )

        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        logger.info(f"Loaded checkpoint state from {checkpoint_path}")

        _, val_loader = create_dataloaders(
            vocab_src=vocab_src,
            vocab_tgt=vocab_tgt,
            batch_size=config.get("batch_size", 32),
            max_padding=config.get("max_padding", 72),
            pad_id=vocab_tgt["<blank>"],
        )

        pad_id = vocab_tgt["<blank>"]
        criterion = LabelSmoothing(
            size=len(vocab_tgt),
            padding_idx=pad_id,
            smoothing=config.get("label_smoothing", 0.1),
        )
        val_loss_compute = SimpleLossCompute(model.generator, criterion)

        with torch.no_grad():
            val_loss, _ = run_epoch(
                data_iter=val_loader,
                model=model,
                loss_compute=val_loss_compute,
                optimizer=DummyOptimizer(),
                scheduler=DummyScheduler(),
                mode="eval",
            )

        logger.info(f"Evaluation Complete -> Validation Loss: {val_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

    except Exception as e:
        logger.error("Error running evaluation script.")
        raise CustomException("Failed to run evaluation script", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Annotated Transformer Model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)
