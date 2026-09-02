import copy
import torch.nn as nn
from src.models.attention import MultiHeadedAttention
from src.models.embeddings import Embeddings, PositionalEncoding
from src.models.layers import PositionwiseFeedForward, EncoderLayer, DecoderLayer
from src.models.encoder_decoder import Encoder, Decoder, Generator, EncoderDecoder
from utils.logger import logger
from utils.custom_exception import CustomException


def make_model(
    src_vocab: int, 
    tgt_vocab: int, 
    N: int = 6, 
    d_model: int = 512, 
    d_ff: int = 2048, 
    h: int = 8, 
    dropout: float = 0.1
) -> EncoderDecoder:
    """
    Construct a full Transformer EncoderDecoder model from hyperparameters.
    Initializes parameters with Xavier / Glorot uniform distribution.
    """
    try:
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab),
        )

        # Initialize parameters with Glorot / fan_avg
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        logger.info(f"Successfully constructed Transformer model (N={N}, d_model={d_model}, h={h}).")
        return model
    except Exception as e:
        logger.error("Error in constructing Transformer model.")
        raise CustomException("Failed to construct Transformer model", e)
