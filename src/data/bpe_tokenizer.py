import json
import os
import regex
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple
from utils.logger import logger
from utils.custom_exception import CustomException


def bytes_to_unicode() -> Dict[int, str]:
    """
    Returns a reversible bijection mapping between all 256 byte values
    and 256 unique printable Unicode characters (GPT-2 style byte-level encoding).
    This ensures no control characters or whitespace issues when merging tokens.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def unicode_to_bytes() -> Dict[str, int]:
    """
    Inverse mapping of bytes_to_unicode.
    """
    b2u = bytes_to_unicode()
    return {v: k for k, v in b2u.items()}


# Standard pre-tokenization regex pattern for splitting text into words, numbers, and contractions
BPE_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class ByteLevelBPETokenizer:
    """
    A pure-Python Byte-Level Byte Pair Encoding (BPE) Tokenizer built from scratch.
    
    Features:
    - 100% self-contained: No dependency on HuggingFace tokenizers or SentencePiece.
    - Zero out-of-vocabulary (<unk>) rate: Any arbitrary UTF-8 string is representable.
    - Lossless roundtrip detokenization.
    - Full support for special tokens: <unk>, <blank>, <s>, </s>.
    - Word-level encoding cache for blazing-fast inference and batch encoding.
    """

    SPECIAL_TOKENS = {
        "<unk>": 0,
        "<blank>": 1,
        "<s>": 2,
        "</s>": 3,
    }

    def __init__(self, vocab: Optional[Dict[str, int]] = None, merges: Optional[List[Tuple[str, str]]] = None):
        self.b2u = bytes_to_unicode()
        self.u2b = unicode_to_bytes()
        self.pat = regex.compile(BPE_SPLIT_PATTERN)

        self.unk_id = self.SPECIAL_TOKENS["<unk>"]
        self.pad_id = self.SPECIAL_TOKENS["<blank>"]
        self.bos_id = self.SPECIAL_TOKENS["<s>"]
        self.eos_id = self.SPECIAL_TOKENS["</s>"]

        if vocab is not None and merges is not None:
            self.vocab: Dict[str, int] = dict(vocab)
            self.merges: List[Tuple[str, str]] = [tuple(m) for m in merges]
            self.inverse_vocab: Dict[int, str] = {idx: tok for tok, idx in self.vocab.items()}
            self.ranks: Dict[Tuple[str, str], int] = {m: i for i, m in enumerate(self.merges)}
        else:
            self._init_base_vocab()

        self._word_cache: Dict[str, List[str]] = {}

    def _init_base_vocab(self):
        """Initialize the base vocabulary with special tokens and all 256 individual byte characters."""
        self.vocab: Dict[str, int] = {}
        for token, idx in self.SPECIAL_TOKENS.items():
            self.vocab[token] = idx

        # Base 256 bytes mapped to unicode
        next_idx = len(self.vocab)
        for b in range(256):
            char = self.b2u[b]
            if char not in self.vocab:
                self.vocab[char] = next_idx
                next_idx += 1

        self.inverse_vocab = {idx: tok for tok, idx in self.vocab.items()}
        self.merges = []
        self.ranks = {}
        self._word_cache = {}

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def get_inverse_vocab(self) -> Dict[int, str]:
        return self.inverse_vocab

    def train(self, corpus: Iterable[str], vocab_size: int = 4000, min_frequency: int = 2):
        """
        Train Byte-Level BPE on a corpus of text strings.

        Args:
            corpus: Iterable of raw text sentences.
            vocab_size: Target vocabulary size (including special tokens and base 256 bytes).
            min_frequency: Minimum pair frequency required to create a merge.
        """
        try:
            logger.info(f"Starting Byte-Level BPE training (Target vocab size: {vocab_size}, Min freq: {min_frequency})...")
            self._init_base_vocab()

            # Step 1: Pre-tokenize corpus into words and count word frequencies
            word_freqs = Counter()
            num_lines = 0
            for line in corpus:
                num_lines += 1
                for word in self.pat.findall(line):
                    # Map word bytes to unicode characters
                    encoded_word = "".join(self.b2u[b] for b in word.encode("utf-8"))
                    word_freqs[encoded_word] += 1

            logger.info(f"Pre-tokenized {num_lines} lines into {len(word_freqs)} unique words.")

            # Step 2: Index words, counts, pairs, and word occurrences
            words: List[List[str]] = [list(w) for w in word_freqs.keys()]
            counts: List[int] = list(word_freqs.values())

            from collections import defaultdict
            pairs: Counter = Counter()
            pair_to_words: Dict[Tuple[str, str], Set[int]] = defaultdict(set)

            for wid, w in enumerate(words):
                cnt = counts[wid]
                for i in range(len(w) - 1):
                    p = (w[i], w[i + 1])
                    pairs[p] += cnt
                    pair_to_words[p].add(wid)

            num_merges_target = vocab_size - len(self.vocab)
            logger.info(f"Initial base vocab size: {len(self.vocab)}. Merges to learn: {num_merges_target}.")

            for step in range(num_merges_target):
                if not pairs:
                    logger.info("No more adjacent pairs found. Stopping early.")
                    break

                best_pair, best_freq = pairs.most_common(1)[0]
                if best_freq < min_frequency:
                    logger.info(f"Most frequent pair '{best_pair}' has frequency {best_freq} < min_freq {min_frequency}. Stopping.")
                    break

                # Create new token from the best pair
                new_token = best_pair[0] + best_pair[1]
                new_token_id = len(self.vocab)
                self.vocab[new_token] = new_token_id
                self.inverse_vocab[new_token_id] = new_token
                self.merges.append(best_pair)
                self.ranks[best_pair] = step

                # Inverted index update: only update words containing best_pair
                p0, p1 = best_pair
                affected_wids = list(pair_to_words[best_pair])

                for wid in affected_wids:
                    w = words[wid]
                    cnt = counts[wid]

                    # Remove old pairs for this word
                    for i in range(len(w) - 1):
                        p = (w[i], w[i + 1])
                        pairs[p] -= cnt
                        pair_to_words[p].discard(wid)
                        if pairs[p] <= 0:
                            del pairs[p]

                    # Apply merge to this word
                    new_w: List[str] = []
                    i = 0
                    while i < len(w):
                        if i < len(w) - 1 and w[i] == p0 and w[i + 1] == p1:
                            new_w.append(new_token)
                            i += 2
                        else:
                            new_w.append(w[i])
                            i += 1
                    words[wid] = new_w

                    # Add new pairs for this updated word
                    for i in range(len(new_w) - 1):
                        p = (new_w[i], new_w[i + 1])
                        pairs[p] += cnt
                        pair_to_words[p].add(wid)

                if best_pair in pairs:
                    del pairs[best_pair]
                if best_pair in pair_to_words:
                    del pair_to_words[best_pair]

                if (step + 1) % 500 == 0 or (step + 1) == num_merges_target:
                    logger.info(f"Trained {step + 1}/{num_merges_target} merges. Current vocab size: {len(self.vocab)}.")

            logger.info(f"Byte-Level BPE training complete! Total vocabulary size: {len(self.vocab)}, Total merges: {len(self.merges)}.")
            self._word_cache.clear()

        except Exception as e:
            logger.error("Error during Byte-Level BPE training.")
            raise CustomException("Failed to train Byte-Level BPE tokenizer", e)

    def _bpe_encode_word(self, word_chars: str) -> List[str]:
        """Apply BPE merge rules iteratively to a single word."""
        if word_chars in self._word_cache:
            return self._word_cache[word_chars]

        # Break word into initial single-character symbols
        symbols = list(word_chars)
        if len(symbols) <= 1:
            self._word_cache[word_chars] = symbols
            return symbols

        while len(symbols) > 1:
            # Find all adjacent pairs in current symbols
            min_rank = float("inf")
            best_pair_idx = -1

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.ranks.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    best_pair_idx = i

            if best_pair_idx == -1:
                # No more merges can be applied
                break

            # Merge the best pair
            p0 = symbols[best_pair_idx]
            p1 = symbols[best_pair_idx + 1]
            merged_token = p0 + p1
            symbols = symbols[:best_pair_idx] + [merged_token] + symbols[best_pair_idx + 2:]

        self._word_cache[word_chars] = symbols
        return symbols

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize an input string into a list of BPE subword tokens.
        """
        try:
            tokens: List[str] = []
            for word in self.pat.findall(text):
                # Convert word bytes to unicode representation
                word_chars = "".join(self.b2u[b] for b in word.encode("utf-8"))
                subwords = self._bpe_encode_word(word_chars)
                tokens.extend(subwords)
            return tokens
        except Exception as e:
            logger.error(f"Error tokenizing text '{text[:50]}...'.")
            raise CustomException("Failed to tokenize text", e)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode an input string into token IDs.

        Args:
            text: Input string.
            add_special_tokens: If True, wraps sequence with <s> and </s>.

        Returns:
            List of integer token IDs.
        """
        try:
            subwords = self.tokenize(text)
            ids = [self.vocab.get(tok, self.unk_id) for tok in subwords]

            if add_special_tokens:
                ids = [self.bos_id] + ids + [self.eos_id]

            return ids
        except Exception as e:
            logger.error("Error encoding text into token IDs.")
            raise CustomException("Failed to encode text", e)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs back into a UTF-8 string.
        Reverses the byte-to-unicode mapping losslessly.

        Args:
            token_ids: List of integer token IDs.
            skip_special_tokens: If True, discards <s>, </s>, <blank>, <unk>.

        Returns:
            Reconstructed natural text string.
        """
        try:
            special_ids = {self.unk_id, self.pad_id, self.bos_id, self.eos_id}
            subword_chars = []

            for tid in token_ids:
                if skip_special_tokens and tid in special_ids:
                    continue
                token_str = self.inverse_vocab.get(tid, "")
                if skip_special_tokens and token_str in self.SPECIAL_TOKENS:
                    continue
                subword_chars.append(token_str)

            # Combine all unicode characters
            full_unicode = "".join(subword_chars)

            # Map unicode characters back to original raw bytes
            byte_values = bytearray()
            for ch in full_unicode:
                if ch in self.u2b:
                    byte_values.append(self.u2b[ch])
                else:
                    # Fallback for unexpected characters
                    byte_values.extend(ch.encode("utf-8"))

            return byte_values.decode("utf-8", errors="replace")

        except Exception as e:
            logger.error("Error decoding token IDs.")
            raise CustomException("Failed to decode token IDs", e)

    def save(self, filepath: str):
        """
        Save tokenizer state (vocab, merges, special tokens) to a JSON file.
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            data = {
                "special_tokens": self.SPECIAL_TOKENS,
                "vocab": self.vocab,
                "merges": self.merges,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved ByteLevelBPETokenizer to {filepath} (Vocab size: {len(self.vocab)})")
        except Exception as e:
            logger.error(f"Error saving tokenizer to {filepath}.")
            raise CustomException(f"Failed to save tokenizer to {filepath}", e)

    @classmethod
    def load(cls, filepath: str) -> "ByteLevelBPETokenizer":
        """
        Load tokenizer state from a JSON file.
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Tokenizer file not found at {filepath}")

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            vocab = data["vocab"]
            merges = [tuple(m) for m in data["merges"]]

            tokenizer = cls(vocab=vocab, merges=merges)
            logger.info(f"Loaded ByteLevelBPETokenizer from {filepath} (Vocab size: {len(tokenizer)})")
            return tokenizer
        except Exception as e:
            logger.error(f"Error loading tokenizer from {filepath}.")
            raise CustomException(f"Failed to load tokenizer from {filepath}", e)
