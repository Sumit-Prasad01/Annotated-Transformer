import torch
import torch.nn as nn
from src.models.attention import clones
from utils.logger import logger
from utils.custom_exception import CustomException


class LayerNorm(nn.Module):
    """
    Construct a Layer Normalization module.
    """

    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)
            return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        except Exception as e:
            logger.error("Error in LayerNorm forward pass.")
            raise CustomException("Failed in LayerNorm forward pass", e)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by layer normalization.
    Note: Pre-LN architecture (norm is applied before sublayer).
    """

    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: callable) -> torch.Tensor:
        try:
            return x + self.dropout(sublayer(self.norm(x)))
        except Exception as e:
            logger.error("Error in SublayerConnection forward pass.")
            raise CustomException("Failed in SublayerConnection forward pass", e)


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).
    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.w_2(self.dropout(self.w_1(x).relu()))
        except Exception as e:
            logger.error("Error in PositionwiseFeedForward forward pass.")
            raise CustomException("Failed in PositionwiseFeedForward forward pass", e)


class EncoderLayer(nn.Module):
    """
    Encoder Layer composed of self-attention and feed-forward sublayers.
    """

    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        try:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward)
        except Exception as e:
            logger.error("Error in EncoderLayer forward pass.")
            raise CustomException("Failed in EncoderLayer forward pass", e)


class DecoderLayer(nn.Module):
    """
    Decoder Layer composed of self-attention, cross-attention (src-attn), and feed-forward sublayers.
    """

    def __init__(self, size: int, self_attn: nn.Module, src_attn: nn.Module, 
                 feed_forward: nn.Module, dropout: float):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, 
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        try:
            m = memory
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            return self.sublayer[2](x, self.feed_forward)
        except Exception as e:
            logger.error("Error in DecoderLayer forward pass.")
            raise CustomException("Failed in DecoderLayer forward pass", e)
