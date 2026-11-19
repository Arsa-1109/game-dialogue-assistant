"""Core pipeline for the Game Dialogue Assistant (Milestones 1-4)."""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent / "corpus" / "npc_dialogues.txt"

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<UNK>"

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
_SPEAKER_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 _-]{0,30}):\s*(.+)$")


# --- Milestone 1: Collect and preprocess a corpus of game dialogues.
def clean_text(text: str) -> str:
    """Normalize quotes, whitespace, and casing for dialogue preprocessing."""
    normalized = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return re.sub(r"\s+", " ", normalized).strip().lower()


def tokenize(text: str) -> list[str]:
    """Clean text and return lowercase word tokens (preserving internal apostrophes)."""
    return _TOKEN_RE.findall(clean_text(text))


def parse_dialogue_lines(raw_text: str) -> list[tuple[str | None, str]]:
    """Parse raw script text into (speaker_or_none, dialogue_text) pairs."""
    entries: list[tuple[str | None, str]] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SPEAKER_RE.match(line)
        if match:
            entries.append((match.group(1).strip(), match.group(2).strip()))
        else:
            entries.append((None, line))
    return entries


def list_characters(corpus_path: str | Path | None = None) -> list[str]:
    """Return sorted unique character/speaker names present in the corpus."""
    path = Path(corpus_path) if corpus_path is not None else DEFAULT_CORPUS_PATH
    raw_text = path.read_text(encoding="utf-8")
    speakers = {
        speaker
        for speaker, _ in parse_dialogue_lines(raw_text)
        if speaker is not None
    }
    return sorted(speakers)


def preprocess_corpus(
    raw_text: str,
    character: str | None = None,
) -> list[list[str]]:
    """Clean, filter (optional), sentence-split, and tokenize dialogue corpus."""
    entries = parse_dialogue_lines(raw_text)
    if character is not None:
        target = character.strip().lower()
        available = sorted({spk for spk, _ in entries if spk is not None})
        entries = [
            (spk, line)
            for spk, line in entries
            if spk is not None and spk.lower() == target
        ]
        if not entries:
            avail_str = ", ".join(available) if available else "none"
            raise ValueError(
                f"No dialogue lines found for character '{character}'. "
                f"Available characters: {avail_str}"
            )

    tokenized_sentences: list[list[str]] = []
    for _, line in entries:
        for fragment in _SENTENCE_SPLIT_RE.split(line):
            tokens = tokenize(fragment)
            if tokens:
                tokenized_sentences.append(tokens)
    return tokenized_sentences


def load_corpus(
    corpus_path: str | Path | None = None,
    character: str | None = None,
) -> list[list[str]]:
    """Load and preprocess a dialogue script file into tokenized sentences."""
    path = Path(corpus_path) if corpus_path is not None else DEFAULT_CORPUS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    raw_text = path.read_text(encoding="utf-8")
    return preprocess_corpus(raw_text, character=character)


def milestone_1(
    corpus_path: str | Path | None = None,
    character: str | None = None,
) -> list[list[str]]:
    """Milestone 1 entrypoint: collect and preprocess the game dialogue corpus."""
    return load_corpus(corpus_path=corpus_path, character=character)


# --- Milestone 2: Build the n-gram language model with smoothing.
class NGramModel:
    """N-gram language model with add-k (Laplace) smoothing and order backoff."""

    def __init__(self, n: int = 3, smoothing: float = 1.0):
        if n < 1:
            raise ValueError("n must be at least 1")
        if smoothing <= 0:
            raise ValueError("smoothing (k) must be > 0")
        self.n = n
        self.smoothing = float(smoothing)
        self.vocab: set[str] = set()
        self.sorted_vocab: list[str] = []
        self._ngram_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {}
        self._context_totals: dict[int, dict[tuple[str, ...], int]] = {}
        self.sentence_count = 0
        self.token_count = 0

    def fit(self, sentences: list[list[str]]) -> "NGramModel":
        """Train n-gram counts across orders 1..n from tokenized sentences."""
        non_empty = [s for s in sentences if s]
        if not non_empty:
            raise ValueError("Cannot train NGramModel on an empty corpus.")

        self.vocab = {END_TOKEN, UNK_TOKEN}
        self._ngram_counts = {order: defaultdict(Counter) for order in range(1, self.n + 1)}
        self._context_totals = {order: defaultdict(int) for order in range(1, self.n + 1)}
        self.sentence_count = len(non_empty)
        self.token_count = sum(len(s) for s in non_empty)

        for sentence in non_empty:
            for tok in sentence:
                self.vocab.add(tok)

        for sentence in non_empty:
            for order in range(1, self.n + 1):
                pad_len = order - 1
                padded = ([START_TOKEN] * pad_len) + sentence + [END_TOKEN]
                for idx in range(pad_len, len(padded)):
                    context = tuple(padded[idx - pad_len : idx])
                    target = padded[idx]
                    self._ngram_counts[order][context][target] += 1
                    self._context_totals[order][context] += 1

        self.sorted_vocab = sorted(self.vocab)
        return self

    def _normalize_context(
        self,
        context: tuple[str, ...] | list[str],
        order: int | None = None,
    ) -> tuple[str, ...]:
        target_order = self.n if order is None else order
        ctx_len = target_order - 1
        if ctx_len <= 0:
            return ()
        tokens = list(context)
        if len(tokens) < ctx_len:
            tokens = ([START_TOKEN] * (ctx_len - len(tokens))) + tokens
        else:
            tokens = tokens[-ctx_len:]
        return tuple(tokens)

    def probability(
        self,
        word: str,
        context: tuple[str, ...] | list[str] = (),
        order: int | None = None,
    ) -> float:
        """Compute add-k smoothed P(word | context) = (count + k) / (total + k * |V|)."""
        if not self.vocab:
            raise RuntimeError("NGramModel must be fitted before querying probabilities.")
        target_order = self.n if order is None else order
        ctx = self._normalize_context(context, order=target_order)
        token = word if word in self.vocab else UNK_TOKEN
        counts = self._ngram_counts[target_order].get(ctx)
        word_count = counts[token] if counts else 0
        total = self._context_totals[target_order].get(ctx, 0)
        denom = total + self.smoothing * len(self.vocab)
        return (word_count + self.smoothing) / denom

    def probabilities(
        self,
        context: tuple[str, ...] | list[str] = (),
        order: int | None = None,
    ) -> dict[str, float]:
        """Return full smoothed probability distribution over the vocabulary."""
        if not self.vocab:
            raise RuntimeError("NGramModel must be fitted before querying probabilities.")
        target_order = self.n if order is None else order
        ctx = self._normalize_context(context, order=target_order)
        counts = self._ngram_counts[target_order].get(ctx)
        total = self._context_totals[target_order].get(ctx, 0)
        denom = total + self.smoothing * len(self.vocab)
        return {
            word: ((counts[word] if counts else 0) + self.smoothing) / denom
            for word in self.sorted_vocab
        }

    def perplexity(self, sentences: list[list[str]]) -> float:
        """Compute corpus perplexity under the smoothed n-gram model."""
        if not self.vocab:
            raise RuntimeError("NGramModel must be fitted before computing perplexity.")
        log_prob_sum = 0.0
        total_events = 0
        pad_len = self.n - 1
        for sentence in sentences:
            if not sentence:
                continue
            padded = ([START_TOKEN] * pad_len) + sentence + [END_TOKEN]
            for idx in range(pad_len, len(padded)):
                ctx = tuple(padded[idx - pad_len : idx])
                target = padded[idx]
                prob = self.probability(target, ctx)
                log_prob_sum += math.log(prob)
                total_events += 1
        if total_events == 0:
            raise ValueError("Cannot compute perplexity on empty sentences.")
        return math.exp(-log_prob_sum / total_events)

    def has_context_match(
        self,
        context: tuple[str, ...] | list[str],
        require_continuation: bool = False,
    ) -> bool:
        """Return True at sentence start or when any n-gram order >= 2 matches context."""
        if not context:
            return True
        for order in range(self.n, 1, -1):
            ctx = self._normalize_context(context, order=order)
            counts = self._ngram_counts.get(order, {}).get(ctx)
            if counts and (not require_continuation or any(tok != END_TOKEN for tok in counts)):
                return True
        return False

    def predict_next(
        self,
        context: tuple[str, ...] | list[str] = (),
        temperature: float = 0.85,
        top_k: int = 8,
        rng: random.Random | None = None,
        allow_end: bool = True,
        end_boost: float = 1.0,
        exclude_tokens: set[str] | None = None,
    ) -> str:
        """Predict next token using interpolated smoothed n-gram backoff."""
        if not self.vocab:
            raise RuntimeError("NGramModel must be fitted before prediction.")
        chooser = rng or random.Random()

        matched_orders: list[tuple[int, tuple[str, ...], Counter[str]]] = []
        for order in range(self.n, 1, -1):
            ctx = self._normalize_context(context, order=order)
            counts = self._ngram_counts[order].get(ctx)
            if counts and (allow_end or any(tok != END_TOKEN for tok in counts)):
                matched_orders.append((order, ctx, counts))

        scored: dict[str, float] = defaultdict(float)

        if matched_orders:
            top_order, _, top_counts = matched_orders[0]
            top_non_end = sum(1 for tok in top_counts if tok != END_TOKEN)
            if len(matched_orders) == 1 or top_non_end >= 4:
                active_orders = [matched_orders[0]]
                order_weights = [1.0]
            else:
                active_orders = matched_orders
                raw_weights = []
                for order, _, _ in active_orders:
                    if order == self.n:
                        raw_weights.append(0.72)
                    elif order == self.n - 1:
                        raw_weights.append(0.24)
                    else:
                        raw_weights.append(0.04)
                w_sum = sum(raw_weights)
                order_weights = [w / w_sum for w in raw_weights]

            for (order, ctx, counts), order_w in zip(active_orders, order_weights):
                total = self._context_totals[order].get(ctx, 0)
                denom = total + self.smoothing * len(self.vocab)
                raw_probs = {
                    word: (count + self.smoothing) / denom
                    for word, count in counts.items()
                    if allow_end or word != END_TOKEN
                }
                norm = sum(raw_probs.values()) or 1.0
                for word, prob in raw_probs.items():
                    scored[word] += order_w * (prob / norm)
        else:
            unigram_counts = self._ngram_counts[1].get((), Counter())
            total = self._context_totals[1].get((), 0)
            denom = total + self.smoothing * len(self.vocab)
            for word, count in unigram_counts.items():
                if not allow_end and word == END_TOKEN:
                    continue
                scored[word] = (count + self.smoothing) / denom

        if allow_end and end_boost != 1.0 and END_TOKEN in scored:
            scored[END_TOKEN] *= end_boost

        if exclude_tokens:
            filtered = {w: s for w, s in scored.items() if w not in exclude_tokens}
            if filtered:
                scored = filtered

        candidates = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        if not candidates:
            return END_TOKEN if allow_end else self.sorted_vocab[0]

        if top_k > 0:
            candidates = candidates[:top_k]

        if temperature <= 0:
            return candidates[0][0]

        weights = [prob ** (1.0 / temperature) for _, prob in candidates]
        words = [word for word, _ in candidates]
        return chooser.choices(words, weights=weights, k=1)[0]


def milestone_2(
    sentences: list[list[str]] | None = None,
    n: int = 3,
    smoothing: float = 1.0,
) -> NGramModel:
    """Milestone 2 entrypoint: build and fit the smoothed n-gram language model."""
    corpus = milestone_1() if sentences is None else sentences
    return NGramModel(n=n, smoothing=smoothing).fit(corpus)




def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")
