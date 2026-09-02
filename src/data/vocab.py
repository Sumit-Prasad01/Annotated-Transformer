import os
import torch
from collections import Counter
from datasets import load_dataset
from src.data.tokenizer import load_tokenizers, tokenize
from utils.logger import logger
from utils.custom_exception import CustomException


class Vocab:
    """
    Vocabulary class for managing token-to-ID (stoi) and ID-to-token (itos) mappings.
    """

    def __init__(self, token_to_id=None, id_to_token=None, unk_token="<unk>"):
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


def build_vocabulary(min_freq: int = 2, dataset_name: str = "bentrevett/multi30k"):
    """
    Build source (German) and target (English) vocabularies from dataset.
    Special tokens: <unk>=0, pad (<blank>)=1, <s>=2, </s>=3
    """
    try:
        spacy_de, spacy_en = load_tokenizers()
        
        def tokenize_de(text):
            return tokenize(text, spacy_de)

        def tokenize_en(text):
            return tokenize(text, spacy_en)

        logger.info(f"Building vocabularies from dataset {dataset_name}...")
        dataset = load_dataset(dataset_name)
        
        counter_de = Counter()
        counter_en = Counter()

        for example in dataset["train"]:
            counter_de.update(tokenize_de(example["de"]))
            counter_en.update(tokenize_en(example["en"]))

        specials = ["<unk>", "<blank>", "<s>", "</s>"]

        def make_vocab_dict(counter, min_freq):
            stoi = {tok: idx for idx, tok in enumerate(specials)}
            idx = len(specials)
            for tok, freq in counter.items():
                if freq >= min_freq and tok not in stoi:
                    stoi[tok] = idx
                    idx += 1
            itos = {idx: tok for tok, idx in stoi.items()}
            return stoi, itos

        de_stoi, de_itos = make_vocab_dict(counter_de, min_freq)
        en_stoi, en_itos = make_vocab_dict(counter_en, min_freq)

        vocab_src = Vocab(de_stoi, de_itos)
        vocab_tgt = Vocab(en_stoi, en_itos)

        logger.info(f"Vocabularies built successfully. Src size: {len(vocab_src)}, Tgt size: {len(vocab_tgt)}")
        return vocab_src, vocab_tgt
    except Exception as e:
        logger.error("Error building vocabulary.")
        raise CustomException("Failed to build vocabulary", e)
