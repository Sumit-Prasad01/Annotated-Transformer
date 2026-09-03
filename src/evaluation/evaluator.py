import torch
import torch.nn as nn
from tqdm import tqdm
from utils.logger import logger
from utils.custom_exception import CustomException
from src.training.trainer import run_epoch
from src.training.scheduler import DummyOptimizer, DummyScheduler
from src.inference.generator import greedy_decode, beam_search_decode
from src.evaluation.metrics import calculate_perplexity, calculate_bleu


def evaluate_model(
    model: nn.Module,
    val_loader,
    vocab_src,
    vocab_tgt,
    val_loss_compute,
    device: torch.device = None,
    decoding: str = "greedy",
    max_len: int = 72,
    beam_size: int = 5,
    max_samples: int = None,
) -> dict:
    """
    Run full model evaluation over validation dataloader:
    1. Computes Validation Loss.
    2. Computes Perplexity (PPL).
    3. Generates translations and computes BLEU score.

    Args:
        model (nn.Module): The Transformer model.
        val_loader: DataLoader for validation data.
        vocab_src: Source language Vocab instance.
        vocab_tgt: Target language Vocab instance.
        val_loss_compute: Loss computation callable (e.g. SimpleLossCompute).
        device (torch.device): CUDA or CPU device.
        decoding (str): "greedy" or "beam" search decoding.
        max_len (int): Maximum sequence generation length.
        beam_size (int): Beam width if decoding == "beam".
        max_samples (int): Max number of samples to generate for BLEU (None for all).

    Returns:
        dict: Evaluation metrics containing 'val_loss', 'perplexity', and 'bleu_score'.
    """
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        model.eval()

        logger.info("Computing validation loss...")
        with torch.no_grad():
            val_loss, _ = run_epoch(
                data_iter=val_loader,
                model=model,
                loss_compute=val_loss_compute,
                optimizer=DummyOptimizer(),
                scheduler=DummyScheduler(),
                mode="eval",
                device=device,
            )

        val_loss = float(val_loss)
        perplexity = calculate_perplexity(val_loss)
        logger.info(f"Validation Loss: {val_loss:.4f} | Perplexity: {perplexity:.4f}")

        # Compute BLEU Score via generation
        logger.info(f"Generating translations for BLEU evaluation (decoding={decoding})...")
        hypotheses = []
        references = []

        special_tokens = {"<s>", "</s>", "<blank>", "<pad>", "<unk>"}
        itos_tgt = vocab_tgt.get_itos()

        start_symbol = vocab_tgt["<s>"]
        end_symbol = vocab_tgt["</s>"]

        total_samples_evaluated = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="BLEU Evaluation"):
                src_batch = batch.src.to(device)
                src_mask_batch = batch.src_mask.to(device)
                tgt_y_batch = batch.tgt_y

                batch_size = src_batch.size(0)

                for b in range(batch_size):
                    if max_samples is not None and total_samples_evaluated >= max_samples:
                        break

                    src_single = src_batch[b : b + 1]
                    src_mask_single = src_mask_batch[b : b + 1]

                    if decoding == "beam":
                        out_tokens = beam_search_decode(
                            model,
                            src_single,
                            src_mask_single,
                            max_len=max_len,
                            start_symbol=start_symbol,
                            end_symbol=end_symbol,
                            beam_size=beam_size,
                        )
                    else:
                        out_tokens = greedy_decode(
                            model,
                            src_single,
                            src_mask_single,
                            max_len=max_len,
                            start_symbol=start_symbol,
                            end_symbol=end_symbol,
                        )

                    # Extract hypothesis word tokens
                    hyp_words = []
                    for idx in out_tokens[0]:
                        tok = itos_tgt.get(idx.item(), "")
                        if tok not in special_tokens:
                            hyp_words.append(tok)

                    # Extract reference word tokens
                    ref_words = []
                    for idx in tgt_y_batch[b]:
                        tok = itos_tgt.get(idx.item(), "")
                        if tok not in special_tokens:
                            ref_words.append(tok)

                    hypotheses.append(hyp_words)
                    references.append([ref_words])

                    total_samples_evaluated += 1

                if max_samples is not None and total_samples_evaluated >= max_samples:
                    break

        bleu_score = calculate_bleu(references, hypotheses)
        logger.info(f"BLEU Score: {bleu_score:.2f} (evaluated on {total_samples_evaluated} samples)")

        return {
            "val_loss": val_loss,
            "perplexity": perplexity,
            "bleu_score": bleu_score,
            "samples_evaluated": total_samples_evaluated,
        }

    except Exception as e:
        logger.error("Error executing model evaluation.")
        raise CustomException("Failed during model evaluation", e)
