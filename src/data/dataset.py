import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import pad
from datasets import load_dataset
from src.data.batch import Batch
from src.data.tokenizer import load_tokenizers, tokenize
from src.data.vocab import Vocab
from utils.logger import logger
from utils.custom_exception import CustomException


def data_gen(V: int, batch_size: int, nbatches: int):
    """
    Generate synthetic data for a src-tgt copy task (useful for sanity testing & debugging).
    """
    try:
        for _ in range(nbatches):
            data = torch.randint(1, V, size=(batch_size, 10))
            data[:, 0] = 1  # Start token
            src = data.requires_grad_(False)
            tgt = data.requires_grad_(False)
            yield Batch(src, tgt, pad=0)
    except Exception as e:
        logger.error("Error generating synthetic copy task data.")
        raise CustomException("Failed to generate synthetic data", e)


class Multi30kDataset(Dataset):
    """
    PyTorch Dataset wrapper for Multi30k German-English translation task.
    """

    def __init__(self, split: str = "train", dataset_name: str = "bentrevett/multi30k"):
        try:
            self.data = load_dataset(dataset_name, split=split)
        except Exception as e:
            logger.error(f"Error loading split '{split}' from dataset {dataset_name}.")
            raise CustomException(f"Failed to load dataset split '{split}'", e)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        return {"de": item["de"], "en": item["en"]}


def collate_fn(batch, vocab_src: Vocab, vocab_tgt: Vocab, spacy_de, spacy_en, 
               max_padding: int = 72, pad_id: int = 1) -> Batch:
    """
    Collate function to process raw text samples into padded Tensor Batches.
    """
    try:
        bs_id = vocab_tgt["<s>"]
        eos_id = vocab_tgt["</s>"]

        src_list, tgt_list = [], []
        for sample in batch:
            src_tokens = [vocab_src["<s>"]] + [vocab_src[token] for token in tokenize(sample["de"], spacy_de)] + [vocab_src["</s>"]]
            tgt_tokens = [bs_id] + [vocab_tgt[token] for token in tokenize(sample["en"], spacy_en)] + [eos_id]

            src_list.append(torch.tensor(src_tokens, dtype=torch.long))
            tgt_list.append(torch.tensor(tgt_tokens, dtype=torch.long))

        # Pad sequences up to max_padding
        padded_src = []
        padded_tgt = []
        for src_tensor, tgt_tensor in zip(src_list, tgt_list):
            src_padded = pad(src_tensor, (0, max_padding - len(src_tensor)), value=pad_id)[:max_padding]
            tgt_padded = pad(tgt_tensor, (0, max_padding - len(tgt_tensor)), value=pad_id)[:max_padding]
            padded_src.append(src_padded)
            padded_tgt.append(tgt_padded)

        src_batch = torch.stack(padded_src)
        tgt_batch = torch.stack(padded_tgt)

        return Batch(src_batch, tgt_batch, pad=pad_id)
    except Exception as e:
        logger.error("Error in collate_fn processing batch.")
        raise CustomException("Failed in collate_fn processing batch", e)


def create_dataloaders(vocab_src: Vocab, vocab_tgt: Vocab, batch_size: int = 32, 
                       max_padding: int = 72, pad_id: int = 1):
    """
    Create PyTorch DataLoaders for train and validation splits.
    """
    try:
        spacy_de, spacy_en = load_tokenizers()

        train_dataset = Multi30kDataset(split="train")
        val_dataset = Multi30kDataset(split="validation")

        collate = lambda b: collate_fn(b, vocab_src, vocab_tgt, spacy_de, spacy_en, max_padding, pad_id)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

        logger.info(f"Created DataLoaders (Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)})")
        return train_dataloader, val_dataloader
    except Exception as e:
        logger.error("Error creating DataLoaders.")
        raise CustomException("Failed to create DataLoaders", e)
