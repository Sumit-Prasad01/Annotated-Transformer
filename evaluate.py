import os
import argparse
import torch
import mlflow
from utils.helper import read_yaml
from utils.logger import logger
from utils.custom_exception import CustomException
from src.data import create_dataloaders, load_tokenizers, Vocab
from src.models import make_model
from src.training import LabelSmoothing, SimpleLossCompute
from src.evaluation import evaluate_model


def evaluate(
    config_path: str = "config/config.yaml",
    checkpoint_path: str = None,
    decoding: str = None,
    beam_size: int = None,
    max_samples: int = None,
):
    try:
        config = read_yaml(config_path)
        output_dir = config.get("output_dir", "outputs")
        tok_cfg = config.get("tokenizer", {})
        src_tok_path = tok_cfg.get("src_path", os.path.join(output_dir, "bpe_de.json"))
        tgt_tok_path = tok_cfg.get("tgt_path", os.path.join(output_dir, "bpe_en.json"))

        tok_de, tok_en = load_tokenizers(src_path=src_tok_path, tgt_path=tgt_tok_path)
        vocab_src = Vocab(tok_de.get_vocab(), tok_de.get_inverse_vocab())
        vocab_tgt = Vocab(tok_en.get_vocab(), tok_en.get_inverse_vocab())

        if checkpoint_path is None:
            checkpoint_path = os.path.join(output_dir, f"{config.get('checkpoint_prefix', 'model_')}best.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}. Please train the model first.")

        # Determine compute device
        cfg_device = config.get("device", "auto")
        if cfg_device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(cfg_device)

        logger.info(f"Using compute device for evaluation: {device}")

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

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info(f"Successfully loaded checkpoint state from {checkpoint_path}")

        # Create Validation DataLoader
        _, val_loader = create_dataloaders(
            tokenizer_src=tok_de,
            tokenizer_tgt=tok_en,
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

        # Decoding parameters
        eval_decoding = decoding if decoding is not None else config.get("eval_decoding", "greedy")
        eval_beam_size = beam_size if beam_size is not None else config.get("eval_beam_size", 5)
        eval_max_samples = max_samples if max_samples is not None else config.get("eval_max_samples", None)
        max_decode_len = config.get("max_decode_len", 72)

        # Execute full model evaluation (Loss, Perplexity, BLEU Score)
        metrics = evaluate_model(
            model=model,
            val_loader=val_loader,
            vocab_src=vocab_src,
            vocab_tgt=vocab_tgt,
            val_loss_compute=val_loss_compute,
            device=device,
            tokenizer_tgt=tok_en,
            decoding=eval_decoding,
            max_len=max_decode_len,
            beam_size=eval_beam_size,
            max_samples=eval_max_samples,
        )

        val_loss = metrics["val_loss"]
        perplexity = metrics["perplexity"]
        bleu_score = metrics["bleu_score"]

        # Print Evaluation Summary
        print("\n" + "=" * 50)
        print("         MODEL EVALUATION RESULTS         ")
        print("=" * 50)
        print(f"  Validation Loss : {val_loss:.4f}")
        print(f"  Perplexity (PPL): {perplexity:.4f}")
        print(f"  BLEU Score      : {bleu_score:.2f}")
        print(f"  Decoding Strategy: {eval_decoding}" + (f" (beam_size={eval_beam_size})" if eval_decoding == "beam" else ""))
        print("=" * 50 + "\n")

        # MLflow Metric Logging
        mlflow_cfg = config.get("mlflow", {})
        if mlflow_cfg.get("enabled", True):
            try:
                os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
                tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
                exp_name = mlflow_cfg.get("experiment_name", "Annotated_Transformer_Multi30k")
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(exp_name)

                with mlflow.start_run(run_name="Evaluation"):
                    mlflow.log_metrics({
                        "eval_val_loss": val_loss,
                        "eval_perplexity": perplexity,
                        "eval_bleu_score": bleu_score,
                    })
                    mlflow.log_params({
                        "eval_decoding": eval_decoding,
                        "eval_beam_size": eval_beam_size,
                        "eval_max_samples": str(eval_max_samples),
                    })
                logger.info("Logged evaluation metrics to MLflow.")
            except Exception as ml_err:
                logger.warning(f"Could not log metrics to MLflow: {ml_err}")

        return metrics

    except Exception as e:
        logger.error("Error running evaluation script.")
        raise CustomException("Failed to run evaluation script", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Annotated Transformer Model (Loss, Perplexity, BLEU)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")

    decode_group = parser.add_mutually_exclusive_group()
    decode_group.add_argument("--greedy", action="store_true", help="Use greedy decoding for BLEU evaluation")
    decode_group.add_argument("--beam", action="store_true", help="Use beam search decoding for BLEU evaluation")

    parser.add_argument("--beam_size", type=int, default=None, help="Beam width when using --beam")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of validation samples for BLEU evaluation")

    args = parser.parse_args()

    decoding_choice = None
    if args.beam:
        decoding_choice = "beam"
    elif args.greedy:
        decoding_choice = "greedy"

    evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        decoding=decoding_choice,
        beam_size=args.beam_size,
        max_samples=args.max_samples,
    )
