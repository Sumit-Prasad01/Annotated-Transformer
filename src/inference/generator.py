import torch
import torch.nn.functional as F
from src.data.batch import subsequent_mask
from src.data.bpe_tokenizer import ByteLevelBPETokenizer
from src.data.tokenizer import load_tokenizers, tokenize
from src.data.vocab import Vocab
from utils.logger import logger
from utils.custom_exception import CustomException


def greedy_decode(model: torch.nn.Module, src: torch.Tensor, src_mask: torch.Tensor,
                  max_len: int, start_symbol: int, end_symbol: int) -> torch.Tensor:
    """
    Autoregressive greedy decode step for Transformer Seq2Seq model.
    Stops early once the end-of-sequence token is generated.
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
            if next_word == end_symbol:
                break
        return ys
    except Exception as e:
        logger.error("Error in greedy_decode autoregressive generation.")
        raise CustomException("Failed in greedy_decode generation", e)


def beam_search_decode(model: torch.nn.Module, src: torch.Tensor, src_mask: torch.Tensor,
                        max_len: int, start_symbol: int, end_symbol: int,
                        beam_size: int = 5, length_penalty: float = 0.6) -> torch.Tensor:
    """
    Beam search decode for Transformer Seq2Seq model.

    Keeps `beam_size` candidate sequences at each step instead of committing to
    a single greedy choice, then returns the best-scoring completed sequence.

    length_penalty: >1.0 favors longer sequences less harshly by normalizing
    log-prob by length. Google NMT style: ((5 + len) / 6) ** alpha.
    """
    try:
        device = src.device
        memory = model.encode(src, src_mask)  # (1, src_len, d_model)

        # Each beam entry: (sequence tensor, cumulative log-prob score)
        sequences = [(torch.zeros(1, 1).fill_(start_symbol).type_as(src.data), 0.0)]
        completed = []

        for step in range(max_len - 1):
            all_candidates = []

            for seq, score in sequences:
                # If this beam already ended, keep it as-is (don't extend further)
                if seq[0, -1].item() == end_symbol:
                    completed.append((seq, score))
                    continue

                out = model.decode(
                    memory, src_mask, seq, subsequent_mask(seq.size(1)).type_as(src.data)
                )
                log_probs = F.log_softmax(model.generator(out[:, -1]), dim=1)  # (1, vocab)

                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=1)

                for k in range(beam_size):
                    next_word = topk_ids[0, k].view(1, 1)
                    next_score = score + topk_log_probs[0, k].item()
                    new_seq = torch.cat([seq, next_word.type_as(src.data)], dim=1)
                    all_candidates.append((new_seq, next_score))

            if not all_candidates:
                # All beams already completed
                break

            # Apply length normalization for ranking, keep top beam_size candidates
            def normalized_score(item):
                seq, score = item
                length = seq.size(1)
                lp = ((5 + length) / 6) ** length_penalty
                return score / lp

            all_candidates.sort(key=normalized_score, reverse=True)
            sequences = all_candidates[:beam_size]

            # Stop early if we already have enough completed beams that beat all active ones
            if len(completed) >= beam_size:
                best_completed = max(completed, key=normalized_score)
                best_active = max(sequences, key=normalized_score)
                if normalized_score(best_completed) >= normalized_score(best_active):
                    break

        # Add any remaining active (unfinished) beams as fallback candidates
        completed.extend(sequences)

        def normalized_score(item):
            seq, score = item
            length = seq.size(1)
            lp = ((5 + length) / 6) ** length_penalty
            return score / lp

        best_seq, _ = max(completed, key=normalized_score)
        return best_seq

    except Exception as e:
        logger.error("Error in beam_search_decode autoregressive generation.")
        raise CustomException("Failed in beam_search_decode generation", e)


class Translator:
    """
    High-level string-to-string translation pipeline using trained Transformer model
    and scratch Byte-Level BPE tokenizers.
    """

    def __init__(self, model: torch.nn.Module, vocab_src: Vocab = None, vocab_tgt: Vocab = None,
                 tokenizer_src: ByteLevelBPETokenizer = None, tokenizer_tgt: ByteLevelBPETokenizer = None,
                 max_len: int = 72, device: torch.device = None,
                 decoding: str = "greedy", beam_size: int = 5):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        if self.tokenizer_src is None or self.tokenizer_tgt is None:
            self.tokenizer_src, self.tokenizer_tgt = load_tokenizers()

        self.vocab_src = vocab_src if vocab_src is not None else Vocab(self.tokenizer_src.get_vocab(), self.tokenizer_src.get_inverse_vocab())
        self.vocab_tgt = vocab_tgt if vocab_tgt is not None else Vocab(self.tokenizer_tgt.get_vocab(), self.tokenizer_tgt.get_inverse_vocab())
        self.max_len = max_len
        self.decoding = decoding  # "greedy" or "beam"
        self.beam_size = beam_size
        self.model.eval()

    def translate(self, text: str) -> str:
        """
        Translate an input German text string into English using Byte-Level BPE.
        """
        try:
            tokens = self.tokenizer_src.encode(text, add_special_tokens=True)
            src = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
            src_mask = (src != self.vocab_src["<blank>"]).unsqueeze(-2).to(self.device)

            with torch.no_grad():
                if self.decoding == "beam":
                    out_tokens = beam_search_decode(
                        self.model, src, src_mask, max_len=self.max_len,
                        start_symbol=self.vocab_tgt["<s>"],
                        end_symbol=self.vocab_tgt["</s>"],
                        beam_size=self.beam_size,
                    )
                else:
                    out_tokens = greedy_decode(
                        self.model, src, src_mask, max_len=self.max_len,
                        start_symbol=self.vocab_tgt["<s>"],
                        end_symbol=self.vocab_tgt["</s>"],
                    )

            # Decode target token IDs into natural English text
            token_ids = [idx.item() for idx in out_tokens[0]]
            return self.tokenizer_tgt.decode(token_ids, skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"Error translating sentence '{text}'.")
            raise CustomException(f"Failed to translate sentence '{text}'", e)