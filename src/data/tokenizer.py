import os
from typing import Tuple
from src.data.bpe_tokenizer import ByteLevelBPETokenizer
from utils.logger import logger
from utils.custom_exception import CustomException


def load_tokenizers(
    src_path: str = "outputs/bpe_de.json",
    tgt_path: str = "outputs/bpe_en.json",
    vocab_size: int = 8000,
) -> Tuple[ByteLevelBPETokenizer, ByteLevelBPETokenizer]:
    """
    Load scratch Byte-Level BPE tokenizers for German (src) and English (tgt).
    Automatically triggers training if tokenizer files are not yet present on disk.

    Args:
        src_path (str): Path to the German BPE tokenizer JSON.
        tgt_path (str): Path to the English BPE tokenizer JSON.
        vocab_size (int): Target vocabulary size if training is triggered.

    Returns:
        tuple: (tok_de, tok_en) ByteLevelBPETokenizer instances.
    """
    try:
        if not os.path.exists(src_path) or not os.path.exists(tgt_path):
            logger.info(f"Tokenizer files not found at '{src_path}' and/or '{tgt_path}'. Training now...")
            from train_tokenizer import train_tokenizers
            out_dir = os.path.dirname(src_path) or "outputs"
            train_tokenizers(vocab_size=vocab_size, output_dir=out_dir)

        tok_de = ByteLevelBPETokenizer.load(src_path)
        tok_en = ByteLevelBPETokenizer.load(tgt_path)
        logger.info(f"Byte-Level BPE tokenizers loaded successfully. Src size: {len(tok_de)}, Tgt size: {len(tok_en)}")
        return tok_de, tok_en
    except Exception as e:
        logger.error("Error loading Byte-Level BPE tokenizers.")
        raise CustomException("Failed to load Byte-Level BPE tokenizers", e)


def tokenize(text: str, tokenizer) -> list:
    """
    Tokenize a given string using a tokenizer instance.
    Supports ByteLevelBPETokenizer as well as fallback tokenizer objects.
    """
    try:
        if isinstance(tokenizer, ByteLevelBPETokenizer):
            return tokenizer.tokenize(text)
        elif hasattr(tokenizer, "tokenize"):
            return tokenizer.tokenize(text)
        elif hasattr(tokenizer, "tokenizer"):
            return [token.text for token in tokenizer.tokenizer(text)]
        else:
            return list(text.split())
    except Exception as e:
        logger.error("Error tokenizing text.")
        raise CustomException("Failed to tokenize text", e)
