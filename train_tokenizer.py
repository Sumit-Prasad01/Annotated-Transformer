import argparse
import os
from datasets import load_dataset
from src.data.bpe_tokenizer import ByteLevelBPETokenizer
from utils.logger import logger
from utils.custom_exception import CustomException


def train_tokenizers(
    vocab_size: int = 4000,
    output_dir: str = "outputs",
    min_freq: int = 2,
    shared: bool = False,
    dataset_name: str = "bentrevett/multi30k",
) -> dict:
    """
    Train scratch Byte-Level BPE tokenizers on the Multi30k dataset.

    Args:
        vocab_size (int): Target vocabulary size per tokenizer.
        output_dir (str): Directory where tokenizer JSON files will be stored.
        min_freq (int): Minimum pair frequency for BPE merges.
        shared (bool): If True, trains a single shared German+English tokenizer.
        dataset_name (str): HuggingFace dataset name.

    Returns:
        dict: Mapping of tokenizer instances {'src': tok_de, 'tgt': tok_en} (or {'shared': tok}).
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Loading dataset '{dataset_name}' (train split)...")
        dataset = load_dataset(dataset_name, split="train")

        german_corpus = [sample["de"] for sample in dataset]
        english_corpus = [sample["en"] for sample in dataset]
        logger.info(f"Loaded {len(german_corpus)} German sentences and {len(english_corpus)} English sentences.")

        if shared:
            logger.info(f"Training shared Byte-Level BPE tokenizer (Target vocab: {vocab_size})...")
            shared_corpus = german_corpus + english_corpus
            shared_tok = ByteLevelBPETokenizer()
            shared_tok.train(shared_corpus, vocab_size=vocab_size, min_frequency=min_freq)

            shared_path = os.path.join(output_dir, "bpe_shared.json")
            shared_tok.save(shared_path)
            logger.info(f"Shared tokenizer saved to {shared_path} (Vocab size: {len(shared_tok)})")

            # Verification test
            sample_de = german_corpus[0]
            sample_en = english_corpus[0]
            assert shared_tok.decode(shared_tok.encode(sample_de, add_special_tokens=False)) == sample_de
            assert shared_tok.decode(shared_tok.encode(sample_en, add_special_tokens=False)) == sample_en
            logger.info("Verification passed for shared tokenizer.")

            return {"src": shared_tok, "tgt": shared_tok, "shared": shared_tok}

        else:
            # 1. Train German Tokenizer
            logger.info(f"--- Training German (Source) BPE Tokenizer (Target vocab: {vocab_size}) ---")
            tok_de = ByteLevelBPETokenizer()
            tok_de.train(german_corpus, vocab_size=vocab_size, min_frequency=min_freq)
            path_de = os.path.join(output_dir, "bpe_de.json")
            tok_de.save(path_de)
            logger.info(f"German tokenizer saved to {path_de} (Vocab size: {len(tok_de)})")

            # 2. Train English Tokenizer
            logger.info(f"--- Training English (Target) BPE Tokenizer (Target vocab: {vocab_size}) ---")
            tok_en = ByteLevelBPETokenizer()
            tok_en.train(english_corpus, vocab_size=vocab_size, min_frequency=min_freq)
            path_en = os.path.join(output_dir, "bpe_en.json")
            tok_en.save(path_en)
            logger.info(f"English tokenizer saved to {path_en} (Vocab size: {len(tok_en)})")

            # Verification test
            sample_de = german_corpus[0]
            sample_en = english_corpus[0]
            assert tok_de.decode(tok_de.encode(sample_de, add_special_tokens=False)) == sample_de
            assert tok_en.decode(tok_en.encode(sample_en, add_special_tokens=False)) == sample_en
            logger.info("Lossless roundtrip verification passed for both tokenizers!")

            return {"src": tok_de, "tgt": tok_en}

    except Exception as e:
        logger.error("Error executing tokenizer training pipeline.")
        raise CustomException("Failed to train tokenizers", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Scratch Byte-Level BPE Tokenizer for Multi30k")
    parser.add_argument("--vocab_size", type=int, default=4000, help="Target vocabulary size per tokenizer (default: 4000)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save tokenizer JSON files (default: outputs)")
    parser.add_argument("--min_freq", type=int, default=2, help="Minimum pair frequency for BPE merges (default: 2)")
    parser.add_argument("--shared", action="store_true", help="Train a single shared German+English tokenizer")
    parser.add_argument("--dataset", type=str, default="bentrevett/multi30k", help="Dataset name on HuggingFace")

    args = parser.parse_args()

    train_tokenizers(
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        min_freq=args.min_freq,
        shared=args.shared,
        dataset_name=args.dataset,
    )
