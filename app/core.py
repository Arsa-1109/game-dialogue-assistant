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




def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")
