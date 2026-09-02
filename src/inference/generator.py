import torch
from src.data.batch import subsequent_mask
from src.data.tokenizer import load_tokenizers, tokenize
from src.data.vocab import Vocab
from utils.logger import logger
from utils.custom_exception import CustomException


def greedy_decode(model: torch.nn.Module, src: torch.Tensor, src_mask: torch.Tensor, 
                  max_len: int, start_symbol: int) -> torch.Tensor:
    """
    Autoregressive greedy decode step for Transformer Seq2Seq model.
    """
    try:
        memory = model.encode(src, src_mask)
        ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len - 1):
            out = model.decode(
                memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
            )
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat(
                [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
        return ys
    except Exception as e:
        logger.error("Error in greedy_decode autoregressive generation.")
        raise CustomException("Failed in greedy_decode generation", e)


class Translator:
    """
    High-level string-to-string translation pipeline using trained Transformer model.
    """

    def __init__(self, model: torch.nn.Module, vocab_src: Vocab, vocab_tgt: Vocab, 
                 max_len: int = 72, device: torch.device = None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.max_len = max_len
        self.spacy_de, _ = load_tokenizers()
        self.model.eval()

    def translate(self, text: str) -> str:
        """
        Translate an input German text string into English.
        """
        try:
            tokens = [self.vocab_src["<s>"]] + [self.vocab_src[t] for t in tokenize(text, self.spacy_de)] + [self.vocab_src["</s>"]]
            src = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
            src_mask = (src != self.vocab_src["<blank>"]).unsqueeze(-2).to(self.device)

            with torch.no_grad():
                out_tokens = greedy_decode(
                    self.model, src, src_mask, max_len=self.max_len, start_symbol=self.vocab_tgt["<s>"]
                )

            itos = self.vocab_tgt.get_itos()
            translated_words = []
            for idx in out_tokens[0]:
                token_str = itos.get(idx.item(), "")
                if token_str in ["<s>", "</s>", "<blank>"]:
                    continue
                translated_words.append(token_str)

            return " ".join(translated_words)
        except Exception as e:
            logger.error(f"Error translating sentence '{text}'.")
            raise CustomException(f"Failed to translate sentence '{text}'", e)
