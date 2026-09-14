import os
import argparse
import torch
import mlflow
import mlflow.pytorch
from utils.helper import read_yaml
from utils.logger import logger
from utils.custom_exception import CustomException
from src.data import build_vocabulary, create_dataloaders, load_tokenizers, Vocab
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

        # Load Byte-Level BPE Tokenizers
        tok_cfg = config.get("tokenizer", {})
        src_tok_path = tok_cfg.get("src_path", os.path.join(output_dir, "bpe_de.json"))
        tgt_tok_path = tok_cfg.get("tgt_path", os.path.join(output_dir, "bpe_en.json"))
        vocab_size = tok_cfg.get("vocab_size", 8000)

        tok_de, tok_en = load_tokenizers(src_path=src_tok_path, tgt_path=tgt_tok_path, vocab_size=vocab_size)
        vocab_src = Vocab(tok_de.get_vocab(), tok_de.get_inverse_vocab())
        vocab_tgt = Vocab(tok_en.get_vocab(), tok_en.get_inverse_vocab())

        # Save synchronized vocabulary mapping
        torch.save(
            {
                "src_stoi": vocab_src.get_stoi(),
                "src_itos": vocab_src.get_itos(),
                "tgt_stoi": vocab_tgt.get_stoi(),
                "tgt_itos": vocab_tgt.get_itos(),
            },
            vocab_path,
        )
        logger.info(f"Synchronized BPE vocabulary saved to {vocab_path} (Src: {len(vocab_src)}, Tgt: {len(vocab_tgt)})")

        # Create DataLoaders
        train_loader, val_loader = create_dataloaders(
            tokenizer_src=tok_de,
            tokenizer_tgt=tok_en,
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

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Transformer model constructed. Total parameters: {total_params:,}")

        # MLflow Setup
        mlflow_cfg = config.get("mlflow", {})
        use_mlflow = mlflow_cfg.get("enabled", True)

        if use_mlflow:
            # Opt-out of file store exception fallback for compatibility
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
            exp_name = mlflow_cfg.get("experiment_name", "Annotated_Transformer_Multi30k")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(exp_name)
            logger.info(f"MLflow experiment '{exp_name}' set with tracking URI '{tracking_uri}'.")

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

        checkpoint_path = os.path.join(output_dir, f"{config.get('checkpoint_prefix', 'model_')}best.pt")

        def run_training_pipeline():
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
                use_mlflow=use_mlflow,
            )

        if use_mlflow:
            with mlflow.start_run():
                # Log hyperparameter dictionary to MLflow
                mlflow.log_params({
                    "num_epochs": config.get("num_epochs", 8),
                    "batch_size": config.get("batch_size", 16),
                    "accum_iter": config.get("accum_iter", 20),
                    "base_lr": config.get("base_lr", 1.0),
                    "warmup": config.get("warmup", 3000),
                    "d_model": config.get("d_model", 512),
                    "d_ff": config.get("d_ff", 2048),
                    "num_layers": config.get("num_layers", 6),
                    "num_heads": config.get("num_heads", 8),
                    "dropout": config.get("dropout", 0.1),
                    "label_smoothing": config.get("label_smoothing", 0.1),
                    "max_padding": config.get("max_padding", 72),
                    "src_vocab_size": len(vocab_src),
                    "tgt_vocab_size": len(vocab_tgt),
                    "total_parameters": total_params,
                    "device": str(device),
                    "seed": seed,
                })

                run_training_pipeline()

                # Log artifacts
                if os.path.exists(config_path):
                    mlflow.log_artifact(config_path, artifact_path="config")
                if os.path.exists(vocab_path):
                    mlflow.log_artifact(vocab_path, artifact_path="vocabulary")
                if os.path.exists(src_tok_path):
                    mlflow.log_artifact(src_tok_path, artifact_path="tokenizer")
                if os.path.exists(tgt_tok_path):
                    mlflow.log_artifact(tgt_tok_path, artifact_path="tokenizer")
                if os.path.exists(checkpoint_path):
                    mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
                    
                logger.info("Successfully logged run parameters, metrics, and artifacts to MLflow.")
        else:
            run_training_pipeline()

        logger.info(f"Training completed successfully. Model saved to {checkpoint_path}")

    except Exception as e:
        logger.error("Error executing main training pipeline.")
        raise CustomException("Failed to run training script", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Annotated Transformer Model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
