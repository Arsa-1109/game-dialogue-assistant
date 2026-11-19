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


# --- Milestone 3: Implement dialogue generation logic using the model.
def _format_word(word: str, is_sentence_start: bool) -> str:
    if word == "i":
        return "I"
    if word.startswith("i'"):
        return "I" + word[1:]
    if is_sentence_start and word:
        return word[0].upper() + word[1:]
    return word


_DANGLING_END_WORDS = {
    "and", "or", "but", "the", "a", "an", "to", "of", "in", "for", "with",
    "from", "at", "by", "on", "if", "that", "as", "will", "would", "could",
    "should", "can", "may", "might", "my", "your", "our", "their", "his",
    "her", "its", "this", "these", "those", "very", "so", "is", "are",
    "was", "were", "be", "has", "have", "had", "into", "near", "before",
    "after", "when", "while", "where", "who", "which", "than",
}


def detokenize(
    sentences: list[list[str]],
    terminators: list[str] | None = None,
) -> str:
    """Format tokenized sentences into readable NPC dialogue."""
    formatted_sentences: list[str] = []
    for idx, sent in enumerate(sentences):
        if not sent:
            continue
        words = [
            _format_word(tok, is_sentence_start=(word_idx == 0))
            for word_idx, tok in enumerate(sent)
        ]
        term = terminators[idx] if terminators and idx < len(terminators) else "."
        formatted_sentences.append(" ".join(words) + term)
    return " ".join(formatted_sentences)


def generate_dialogue(
    model: NGramModel,
    prompt: str = "",
    length: int = 18,
    temperature: float = 0.85,
    top_k: int = 8,
    seed: int | None = None,
) -> str:
    """Generate a coherent NPC dialogue snippet starting from `prompt`."""
    if length < 1:
        raise ValueError("length must be at least 1")

    rng = random.Random(seed)
    stripped = prompt.strip()
    prompt_term = stripped[-1] if stripped and stripped[-1] in {".", "?", "!"} else None
    prompt_tokens = tokenize(prompt)

    if len(prompt_tokens) >= length:
        return detokenize([prompt_tokens[:length]], [prompt_term or "."])

    sentences: list[list[str]] = []
    terminators: list[str] = []

    if prompt_term is not None and prompt_tokens:
        sentences.append(list(prompt_tokens))
        terminators.append(prompt_term)
        current_sentence: list[str] = []
        prompt_sentence_active = False
    elif len(prompt_tokens) >= 3 and not model.has_context_match(prompt_tokens, require_continuation=True):
        sentences.append(list(prompt_tokens))
        terminators.append(".")
        current_sentence = []
        prompt_sentence_active = False
    else:
        current_sentence = list(prompt_tokens)
        prompt_sentence_active = bool(prompt_tokens)

    total_words = len(prompt_tokens)
    min_target = min(length, max(total_words + 4, length // 2, 6))
    ended_naturally = False

    while total_words < length:
        if current_sentence:
            min_sent_len = max(3, len(prompt_tokens) + 1) if prompt_sentence_active else 3
            allow_end = len(current_sentence) >= min_sent_len
        else:
            allow_end = False

        end_boost = 2.5 if (allow_end and total_words >= max(min_target, length - 4)) else 1.0

        exclude_tokens: set[str] = set()
        if current_sentence:
            last_tok = current_sentence[-1]
            exclude_tokens.add(last_tok)
            for idx in range(len(current_sentence) - 1):
                if current_sentence[idx] == last_tok:
                    exclude_tokens.add(current_sentence[idx + 1])

        next_tok = model.predict_next(
            current_sentence,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
            allow_end=allow_end,
            end_boost=end_boost,
            exclude_tokens=exclude_tokens,
        )
        if next_tok == END_TOKEN:
            if current_sentence:
                sentences.append(current_sentence)
                terminators.append(".")
                current_sentence = []
                prompt_sentence_active = False
            if total_words >= min_target:
                ended_naturally = True
                break
            continue

        current_sentence.append(next_tok)
        total_words += 1

    if current_sentence:
        if not ended_naturally:
            min_keep = (len(prompt_tokens) + 1) if prompt_sentence_active else 3
            while len(current_sentence) > min_keep and current_sentence[-1] in _DANGLING_END_WORDS:
                current_sentence.pop()
        sentences.append(current_sentence)
        terminators.append(".")

    return detokenize(sentences, terminators)


def generate_options(
    model: NGramModel,
    prompt: str = "",
    count: int = 3,
    length: int = 18,
    temperature: float = 0.9,
    top_k: int = 8,
    seed: int | None = None,
) -> list[str]:
    """Generate multiple dialogue options for a given starting phrase."""
    if count < 1:
        raise ValueError("count must be at least 1")

    results: list[str] = []
    seen: set[str] = set()
    max_attempts = count * 10

    for attempt in range(max_attempts):
        attempt_seed = None if seed is None else seed + attempt * 31
        retry_round = attempt // count
        cur_temp = min(1.5, temperature + 0.12 * retry_round)
        cur_top_k = top_k + 2 * retry_round
        snippet = generate_dialogue(
            model,
            prompt=prompt,
            length=length,
            temperature=cur_temp,
            top_k=cur_top_k,
            seed=attempt_seed,
        )
        if snippet not in seen:
            seen.add(snippet)
            results.append(snippet)
            if len(results) == count:
                break

    while len(results) < count:
        attempt_seed = None if seed is None else seed + (max_attempts + len(results)) * 31
        results.append(
            generate_dialogue(
                model,
                prompt=prompt,
                length=length,
                temperature=min(1.5, temperature + 0.3),
                top_k=top_k + 6,
                seed=attempt_seed,
            )
        )

    return results


def milestone_3(
    prompt: str = "Halt traveler",
    length: int = 18,
    options: int = 1,
    model: NGramModel | None = None,
    seed: int | None = 42,
) -> list[str]:
    """Milestone 3 entrypoint: generate dialogue snippets from a trained model."""
    active_model = milestone_2() if model is None else model
    return generate_options(
        active_model,
        prompt=prompt,
        count=options,
        length=length,
        seed=seed,
    )


# --- Stretch helper: Save generated dialogues to a file.
def save_dialogues(
    output_path: str | Path,
    prompt: str,
    dialogues: list[str],
    metadata: dict | None = None,
) -> Path:
    """Save generated dialogue options to a .json or .txt/.md file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = metadata or {}

    if path.suffix.lower() == ".json":
        payload = {
            "prompt": prompt,
            "metadata": meta,
            "dialogues": dialogues,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    lines = [
        f"Prompt: {prompt or '(none)'}",
    ]
    if meta:
        meta_parts = [f"{k}={v}" for k, v in meta.items() if v is not None]
        if meta_parts:
            lines.append("Settings: " + ", ".join(meta_parts))
    lines.append("")
    for idx, option in enumerate(dialogues, start=1):
        lines.append(f"{idx}. {option}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path




def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")
