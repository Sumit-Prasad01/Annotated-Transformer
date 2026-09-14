import os
import torch
from typing import Optional, Dict
from src.data.tokenizer import load_tokenizers
from utils.logger import logger
from utils.custom_exception import CustomException


class Vocab:
    """
    Vocabulary class for managing token-to-ID (stoi) and ID-to-token (itos) mappings.
    Compatible with Byte-Level BPE tokenizers and checkpoint serialization.
    """

    def __init__(self, token_to_id: Optional[Dict[str, int]] = None, 
                 id_to_token: Optional[Dict[int, str]] = None, 
                 unk_token: str = "<unk>"):
        self.stoi = token_to_id if token_to_id is not None else {}
        self.itos = id_to_token if id_to_token is not None else {}
        self.unk_token = unk_token
        self.default_index = self.stoi.get(unk_token, 0)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.default_index)

    def __call__(self, tokens: list) -> list:
        return [self[token] for token in tokens]

    def __len__(self) -> int:
        return len(self.stoi)

    def get_itos(self) -> dict:
        return self.itos

    def get_stoi(self) -> dict:
        return self.stoi

    def set_default_index(self, index: int):
        self.default_index = index

    def save(self, filepath: str):
        """Save vocabulary mapping to a file."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save({"stoi": self.stoi, "itos": self.itos}, filepath)
            logger.info(f"Saved vocabulary to {filepath}")
        except Exception as e:
            logger.error(f"Error saving vocabulary to {filepath}")
            raise CustomException(f"Failed to save vocabulary to {filepath}", e)

    @classmethod
    def load(cls, filepath: str):
        """Load vocabulary mapping from a file."""
        try:
            data = torch.load(filepath)
            logger.info(f"Loaded vocabulary from {filepath}")
            return cls(token_to_id=data["stoi"], id_to_token=data["itos"])
        except Exception as e:
            logger.error(f"Error loading vocabulary from {filepath}")
            raise CustomException(f"Failed to load vocabulary from {filepath}", e)


def build_vocabulary(
    src_tokenizer_path: str = "outputs/bpe_de.json",
    tgt_tokenizer_path: str = "outputs/bpe_en.json",
    vocab_size: int = 8000,
    min_freq: int = 2,
    dataset_name: str = "bentrevett/multi30k",
):
    """
    Build source (German) and target (English) vocabularies from trained Byte-Level BPE tokenizers.
    Special tokens: <unk>=0, pad (<blank>)=1, <s>=2, </s>=3.
    """
    try:
        tok_de, tok_en = load_tokenizers(
            src_path=src_tokenizer_path,
            tgt_path=tgt_tokenizer_path,
            vocab_size=vocab_size,
        )

        vocab_src = Vocab(tok_de.get_vocab(), tok_de.get_inverse_vocab())
        vocab_tgt = Vocab(tok_en.get_vocab(), tok_en.get_inverse_vocab())

        logger.info(f"Vocabularies built from BPE tokenizers. Src size: {len(vocab_src)}, Tgt size: {len(vocab_tgt)}")
        return vocab_src, vocab_tgt
    except Exception as e:
        logger.error("Error building vocabulary from BPE tokenizers.")
        raise CustomException("Failed to build vocabulary", e)
