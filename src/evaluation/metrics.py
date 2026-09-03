import math
from collections import Counter
from utils.logger import logger
from utils.custom_exception import CustomException


def calculate_perplexity(loss: float) -> float:
    """
    Calculate Perplexity (PPL) from cross-entropy / label smoothing loss per token.

    PPL = exp(loss)

    Args:
        loss (float): Average loss per target token.

    Returns:
        float: Perplexity metric value.
    """
    try:
        if loss < 0:
            logger.warning(f"Negative loss value encountered ({loss}); returning 1.0 for perplexity.")
            return 1.0
        return math.exp(loss)
    except OverflowError:
        logger.warning(f"Loss value too large ({loss}) causing exponent overflow; returning float('inf').")
        return float("inf")
    except Exception as e:
        logger.error("Error computing perplexity.")
        raise CustomException("Failed to calculate perplexity", e)


def _custom_bleu(references: list, hypotheses: list, max_order: int = 4, smooth: bool = True) -> float:
    """
    Pure Python fallback implementation of corpus BLEU-4 with brevity penalty and smoothing.

    Args:
        references (list): List of lists of reference token lists: [[ [ref_tok1, ref_tok2, ...], ... ], ...]
        hypotheses (list): List of hypothesis token lists: [ [hyp_tok1, hyp_tok2, ...], ... ]
        max_order (int): Maximum n-gram order (default: 4).
        smooth (bool): Whether to apply add-1 smoothing to n-gram precision.

    Returns:
        float: BLEU score scaled to [0.0, 100.0].
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    hyp_len = 0
    ref_len = 0

    for refs, hyp in zip(references, hypotheses):
        hyp_len += len(hyp)
        # Choose reference length closest to hypothesis length
        ref_lens = [len(r) for r in refs]
        best_ref_len = min(ref_lens, key=lambda r: (abs(r - len(hyp)), r))
        ref_len += best_ref_len

        for order in range(1, max_order + 1):
            hyp_ngrams = Counter(
                tuple(hyp[i:i + order]) for i in range(len(hyp) - order + 1)
            )

            # Maximum counts across all references
            max_ref_counts = Counter()
            for ref in refs:
                ref_ngrams = Counter(
                    tuple(ref[i:i + order]) for i in range(len(ref) - order + 1)
                )
                for ngram, count in ref_ngrams.items():
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

            # Clipped matches
            clipped_counts = {
                ngram: min(count, max_ref_counts[ngram])
                for ngram, count in hyp_ngrams.items()
            }

            matches_by_order[order - 1] += sum(clipped_counts.values())
            possible_matches_by_order[order - 1] += sum(hyp_ngrams.values())

    precisions = [0.0] * max_order
    for i in range(max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
            else:
                precisions[i] = 0.0

    if min(precisions) <= 0.0:
        return 0.0

    # Geometric mean of precisions
    log_prec_sum = sum(math.log(p) for p in precisions) / max_order
    geo_mean = math.exp(log_prec_sum)

    # Brevity penalty
    if hyp_len == 0:
        return 0.0
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - ref_len / hyp_len)

    return bp * geo_mean * 100.0


def calculate_bleu(references: list, hypotheses: list) -> float:
    """
    Calculate BLEU score for candidate hypothesis sequences against reference sequences.

    Tries external packages in order: sacrebleu -> nltk -> custom fallback.

    Args:
        references: List of reference token lists (or list of list of tokens),
                    e.g. [[["a", "cat"], ["the", "cat"]], [["a", "dog"]]]
        hypotheses: List of hypothesis token lists, e.g. [["a", "cat"], ["a", "dog"]]

    Returns:
        float: BLEU score scaled to [0.0, 100.0].
    """
    try:
        if not hypotheses or not references:
            logger.warning("Empty hypotheses or references provided for BLEU calculation.")
            return 0.0

        # Try sacrebleu
        try:
            import sacrebleu
            # Format input strings for sacrebleu
            # sys_stream: list of hypothesis strings
            sys_stream = [h if isinstance(h, str) else " ".join(h) for h in hypotheses]

            # ref_streams: list of lists of reference strings
            # Determine maximum number of references per sample
            num_refs = max(len(r) if isinstance(r, list) else 1 for r in references)
            ref_streams = []
            for ref_idx in range(num_refs):
                stream = []
                for sample_refs in references:
                    if isinstance(sample_refs, list) and ref_idx < len(sample_refs):
                        r = sample_refs[ref_idx]
                        stream.append(r if isinstance(r, str) else " ".join(r))
                    else:
                        r = sample_refs[0] if isinstance(sample_refs, list) else sample_refs
                        stream.append(r if isinstance(r, str) else " ".join(r))
                ref_streams.append(stream)

            bleu_result = sacrebleu.corpus_bleu(sys_stream, ref_streams)
            logger.info(f"Computed BLEU score using sacrebleu: {bleu_result.score:.2f}")
            return float(bleu_result.score)
        except ImportError:
            pass

        # Try NLTK
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            # Ensure references and hypotheses are lists of token lists
            formatted_refs = []
            for sample_refs in references:
                if isinstance(sample_refs[0], str):
                    formatted_refs.append([sample_refs[0].split()])
                else:
                    formatted_refs.append([r if isinstance(r, list) else r.split() for r in sample_refs])

            formatted_hyps = [h if isinstance(h, list) else h.split() for h in hypotheses]

            sf = SmoothingFunction().method1
            nltk_bleu = corpus_bleu(formatted_refs, formatted_hyps, smoothing_function=sf) * 100.0
            logger.info(f"Computed BLEU score using NLTK: {nltk_bleu:.2f}")
            return float(nltk_bleu)
        except ImportError:
            pass

        # Fallback to custom pure Python implementation
        logger.info("Using custom Python fallback for BLEU calculation.")
        formatted_refs = []
        for sample_refs in references:
            if isinstance(sample_refs[0], str):
                formatted_refs.append([sample_refs[0].split()])
            else:
                formatted_refs.append([r if isinstance(r, list) else r.split() for r in sample_refs])
        formatted_hyps = [h if isinstance(h, list) else h.split() for h in hypotheses]

        bleu_score = _custom_bleu(formatted_refs, formatted_hyps)
        logger.info(f"Computed BLEU score using custom fallback: {bleu_score:.2f}")
        return float(bleu_score)

    except Exception as e:
        logger.error("Error calculating BLEU score.")
        raise CustomException("Failed to calculate BLEU score", e)
