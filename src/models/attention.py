import math
import copy
import torch
import torch.nn as nn
from utils.logger import logger
from utils.custom_exception import CustomException


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
              mask: torch.Tensor = None, dropout: nn.Dropout = None):
    """Compute 'Scaled Dot Product Attention'"""
    try:
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
    except Exception as e:
        logger.error("Error in scaled dot product attention computation.")
        raise CustomException("Failed during attention computation", e)


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention mechanism."""

    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        """
        Take in model size and number of heads.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0, f"d_model ({d_model}) must be divisible by h ({h})"
        
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        """
        Implements Multi-Head Attention.
        query, key, value shapes: (batch_size, seq_len, d_model)
        mask shape: (batch_size, 1, seq_len) or (batch_size, 1, seq_len, seq_len)
        """
        try:
            if mask is not None:
                # Same mask applied to all h heads.
                mask = mask.unsqueeze(1)
            nbatches = query.size(0)

            # 1) Do all the linear projections in batch from d_model => h x d_k
            query, key, value = [
                lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                for lin, x in zip(self.linears, (query, key, value))
            ]

            # 2) Apply attention on all the projected vectors in batch.
            x, self.attn = attention(
                query, key, value, mask=mask, dropout=self.dropout
            )

            # 3) "Concat" using a view and apply final linear projection.
            x = (
                x.transpose(1, 2)
                .contiguous()
                .view(nbatches, -1, self.h * self.d_k)
            )
            del query
            del key
            del value
            return self.linears[-1](x)
        except Exception as e:
            logger.error("Error in MultiHeadedAttention forward pass.")
            raise CustomException("Failed in MultiHeadedAttention forward pass", e)
