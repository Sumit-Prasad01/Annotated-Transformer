import os
import argparse
import torch
from utils.helper import read_yaml
from utils.logger import logger
from utils.custom_exception import CustomException
from src.data import build_vocabulary, create_dataloaders, Vocab
from src.models import make_model
from src.training import LabelSmoothing, SimpleLossCompute, get_std_opt, train_model


def main(config_path: str = "config/config.yaml"):
    try:
        logger.info(f"Loading configuration from {config_path}...")
        config = read_yaml(config_path)

        # Set device and random seed
        cfg_device = config.get("device", "auto")
        if cfg_device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(cfg_device)
            
        logger.info(f"Using compute device: {device}")
        if device.type == "cuda":
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)} | Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        seed = config.get("seed", 42)
        torch.manual_seed(seed)

        output_dir = config.get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        vocab_path = config.get("vocab_path", os.path.join(output_dir, "vocab.pt"))

        # Build or load vocabularies
        if os.path.exists(vocab_path):
            logger.info(f"Loading vocabulary from {vocab_path}...")
            vocab_data = torch.load(vocab_path)
            vocab_src = Vocab(vocab_data["src_stoi"], vocab_data["src_itos"])
            vocab_tgt = Vocab(vocab_data["tgt_stoi"], vocab_data["tgt_itos"])
        else:
            logger.info("Building vocabulary from Multi30k dataset...")
            vocab_src, vocab_tgt = build_vocabulary(min_freq=config.get("vocab_min_freq", 2))
            torch.save(
                {
                    "src_stoi": vocab_src.get_stoi(),
                    "src_itos": vocab_src.get_itos(),
                    "tgt_stoi": vocab_tgt.get_stoi(),
                    "tgt_itos": vocab_tgt.get_itos(),
                },
                vocab_path,
            )
            logger.info(f"Saved vocabulary to {vocab_path}")

        # Create DataLoaders
        train_loader, val_loader = create_dataloaders(
            vocab_src=vocab_src,
            vocab_tgt=vocab_tgt,
            batch_size=config.get("batch_size", 16),
            max_padding=config.get("max_padding", 72),
            pad_id=vocab_tgt["<blank>"],
        )

        # Build Transformer Model
        model = make_model(
            src_vocab=len(vocab_src),
            tgt_vocab=len(vocab_tgt),
            N=config.get("num_layers", 6),
            d_model=config.get("d_model", 512),
            d_ff=config.get("d_ff", 2048),
            h=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1),
        )

        # Define Loss Functions
        pad_id = vocab_tgt["<blank>"]
        criterion = LabelSmoothing(
            size=len(vocab_tgt),
            padding_idx=pad_id,
            smoothing=config.get("label_smoothing", 0.1),
        )
        loss_compute = SimpleLossCompute(model.generator, criterion)
        val_loss_compute = SimpleLossCompute(model.generator, criterion)

        # Optimizer & Scheduler
        optimizer, scheduler = get_std_opt(
            model=model,
            d_model=config.get("d_model", 512),
            factor=config.get("base_lr", 1.0),
            warmup=config.get("warmup", 3000),
        )

        # Launch Training
        checkpoint_path = os.path.join(output_dir, f"{config.get('checkpoint_prefix', 'model_')}best.pt")
        logger.info("Starting model training...")
        train_model(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            loss_compute=loss_compute,
            val_loss_compute=val_loss_compute,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config.get("num_epochs", 8),
            accum_iter=config.get("accum_iter", 20),
            save_path=checkpoint_path,
            device=device,
        )
        logger.info(f"Training completed successfully. Model saved to {checkpoint_path}")

    except Exception as e:
        logger.error("Error executing main training pipeline.")
        raise CustomException("Failed to run training script", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Annotated Transformer Model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
