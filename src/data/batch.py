import torch
from utils.logger import logger
from utils.custom_exception import CustomException


def subsequent_mask(size: int) -> torch.Tensor:
    """
    Mask out subsequent positions in decoder self-attention (causal mask).
    Returns a boolean tensor of shape (1, size, size).
    """
    try:
        attn_shape = (1, size, size)
        subsequent_mask_tensor = torch.triu(torch.ones(attn_shape), diagonal=1).type(
            torch.uint8
        )
        return subsequent_mask_tensor == 0
    except Exception as e:
        logger.error("Error generating subsequent mask.")
        raise CustomException("Failed to generate subsequent mask", e)


class Batch:
    """
    Object for holding a batch of data with mask during training.
    """

    def __init__(self, src: torch.Tensor, tgt: torch.Tensor = None, pad: int = 2):
        """
        src: tensor of shape (batch_size, src_len)
        tgt: tensor of shape (batch_size, tgt_len)
        pad: padding token index (default: 2 for <blank>)
        """
        try:
            self.src = src
            self.src_mask = (src != pad).unsqueeze(-2)
            if tgt is not None:
                self.tgt = tgt[:, :-1]
                self.tgt_y = tgt[:, 1:]
                self.tgt_mask = self.make_std_mask(self.tgt, pad)
                self.ntokens = (self.tgt_y != pad).data.sum()
        except Exception as e:
            logger.error("Error constructing Batch object.")
            raise CustomException("Failed to construct Batch object", e)

    @staticmethod
    def make_std_mask(tgt: torch.Tensor, pad: int) -> torch.Tensor:
        """
        Create a mask to hide padding tokens and future words.
        """
        try:
            tgt_mask = (tgt != pad).unsqueeze(-2)
            tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
                tgt_mask.data
            )
            return tgt_mask
        except Exception as e:
            logger.error("Error in make_std_mask.")
            raise CustomException("Failed in make_std_mask", e)
