import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from src.models.attention import clones
from src.models.layers import LayerNorm
from utils.logger import logger
from utils.custom_exception import CustomException


class Encoder(nn.Module):
    """
    Core Encoder stack composed of N EncoderLayers and a final LayerNorm.
    """

    def __init__(self, layer: nn.Module, N: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        try:
            for layer in self.layers:
                x = layer(x, mask)
            return self.norm(x)
        except Exception as e:
            logger.error("Error in Encoder forward pass.")
            raise CustomException("Failed in Encoder forward pass", e)


class Decoder(nn.Module):
    """
    Core Decoder stack composed of N DecoderLayers and a final LayerNorm.
    """

    def __init__(self, layer: nn.Module, N: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, 
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        try:
            for layer in self.layers:
                x = layer(x, memory, src_mask, tgt_mask)
            return self.norm(x)
        except Exception as e:
            logger.error("Error in Decoder forward pass.")
            raise CustomException("Failed in Decoder forward pass", e)


class Generator(nn.Module):
    """
    Standard linear projection head + log_softmax generation step.
    """

    def __init__(self, d_model: int, vocab: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return log_softmax(self.proj(x), dim=-1)
        except Exception as e:
            logger.error("Error in Generator forward pass.")
            raise CustomException("Failed in Generator forward pass", e)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture for Seq2Seq processing.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, 
                 src_embed: nn.Module, tgt_embed: nn.Module, generator: Generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """Process masked src and target sequences."""
        try:
            return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
        except Exception as e:
            logger.error("Error in EncoderDecoder forward pass.")
            raise CustomException("Failed in EncoderDecoder forward pass", e)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, 
               tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
