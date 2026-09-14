import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import pad
from datasets import load_dataset
from src.data.batch import Batch
from src.data.bpe_tokenizer import ByteLevelBPETokenizer
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


def collate_fn(batch, tokenizer_src=None, tokenizer_tgt=None, 
               vocab_src: Vocab = None, vocab_tgt: Vocab = None,
               max_padding: int = 72, pad_id: int = 1) -> Batch:
    """
    Collate function to process raw text samples into padded Tensor Batches.
    Leverages ByteLevelBPETokenizer for fast subword encoding.
    """
    try:
        src_list, tgt_list = [], []
        for sample in batch:
            if isinstance(tokenizer_src, ByteLevelBPETokenizer) and isinstance(tokenizer_tgt, ByteLevelBPETokenizer):
                src_tokens = tokenizer_src.encode(sample["de"], add_special_tokens=True)
                tgt_tokens = tokenizer_tgt.encode(sample["en"], add_special_tokens=True)
            elif tokenizer_src is not None and tokenizer_tgt is not None and hasattr(tokenizer_src, "encode"):
                src_tokens = tokenizer_src.encode(sample["de"], add_special_tokens=True)
                tgt_tokens = tokenizer_tgt.encode(sample["en"], add_special_tokens=True)
            else:
                # Fallback to word-level tokenize + vocab mapping
                src_tokens = [vocab_src["<s>"]] + [vocab_src[tok] for tok in tokenize(sample["de"], tokenizer_src)] + [vocab_src["</s>"]]
                tgt_tokens = [vocab_tgt["<s>"]] + [vocab_tgt[tok] for tok in tokenize(sample["en"], tokenizer_tgt)] + [vocab_tgt["</s>"]]

            src_list.append(torch.tensor(src_tokens, dtype=torch.long))
            tgt_list.append(torch.tensor(tgt_tokens, dtype=torch.long))

        # Pad sequences up to max_padding
        padded_src = []
        padded_tgt = []
        for src_tensor, tgt_tensor in zip(src_list, tgt_list):
            src_padded = pad(src_tensor, (0, max(0, max_padding - len(src_tensor))), value=pad_id)[:max_padding]
            tgt_padded = pad(tgt_tensor, (0, max(0, max_padding - len(tgt_tensor))), value=pad_id)[:max_padding]
            padded_src.append(src_padded)
            padded_tgt.append(tgt_padded)

        src_batch = torch.stack(padded_src)
        tgt_batch = torch.stack(padded_tgt)

        return Batch(src_batch, tgt_batch, pad=pad_id)
    except Exception as e:
        logger.error("Error in collate_fn processing batch.")
        raise CustomException("Failed in collate_fn processing batch", e)


def create_dataloaders(vocab_src: Vocab = None, vocab_tgt: Vocab = None, 
                       tokenizer_src: ByteLevelBPETokenizer = None,
                       tokenizer_tgt: ByteLevelBPETokenizer = None,
                       batch_size: int = 32, 
                       max_padding: int = 72, pad_id: int = 1):
    """
    Create PyTorch DataLoaders for train and validation splits using Byte-Level BPE.
    """
    try:
        if tokenizer_src is None or tokenizer_tgt is None:
            tokenizer_src, tokenizer_tgt = load_tokenizers()

        if vocab_src is None:
            vocab_src = Vocab(tokenizer_src.get_vocab(), tokenizer_src.get_inverse_vocab())
        if vocab_tgt is None:
            vocab_tgt = Vocab(tokenizer_tgt.get_vocab(), tokenizer_tgt.get_inverse_vocab())

        train_dataset = Multi30kDataset(split="train")
        val_dataset = Multi30kDataset(split="validation")

        collate = lambda b: collate_fn(
            b,
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt,
            vocab_src=vocab_src,
            vocab_tgt=vocab_tgt,
            max_padding=max_padding,
            pad_id=pad_id,
        )

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

        logger.info(f"Created DataLoaders (Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)})")
        return train_dataloader, val_dataloader
    except Exception as e:
        logger.error("Error creating DataLoaders.")
        raise CustomException("Failed to create DataLoaders", e)
