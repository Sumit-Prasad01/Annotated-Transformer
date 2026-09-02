import math
import torch
import torch.nn as nn
from utils.logger import logger
from utils.custom_exception import CustomException


class Embeddings(nn.Module):
    """
    Token Embedding layer with sqrt(d_model) scaling.
    """

    def __init__(self, d_model: int, vocab: int):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.lut(x) * math.sqrt(self.d_model)
        except Exception as e:
            logger.error("Error in Embeddings forward pass.")
            raise CustomException("Failed in Embeddings forward pass", e)


class PositionalEncoding(nn.Module):
    """
    Implement the Sinusoidal Positional Encoding function.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = x + self.pe[:, : x.size(1)].requires_grad_(False)
            return self.dropout(x)
        except Exception as e:
            logger.error("Error in PositionalEncoding forward pass.")
            raise CustomException("Failed in PositionalEncoding forward pass", e)
