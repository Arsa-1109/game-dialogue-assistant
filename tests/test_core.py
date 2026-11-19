"""Unit and CLI tests for the Game Dialogue Assistant (Milestones 1-4)."""
import json

import pytest

from app import cli
from app.core import (
    list_characters,
    load_corpus,
    milestone_1,
    preprocess_corpus,
    tokenize,
)


# Milestone 1: Collect and preprocess a corpus of game dialogues.
def test_milestone_1():
    assert tokenize("Halt, traveler! Don't cross the king's bridge.") == [
        "halt",
        "traveler",
        "don't",
        "cross",
        "the",
        "king's",
        "bridge",
    ]

    sentences = milestone_1()
    assert len(sentences) >= 100
    assert all(isinstance(sent, list) and len(sent) > 0 for sent in sentences)

    characters = list_characters()
    assert {"Blacksmith", "Guard", "Innkeeper", "Mage", "Merchant"}.issubset(set(characters))

    guard_sentences = load_corpus(character="Guard")
    assert 10 <= len(guard_sentences) < len(sentences)


def test_preprocess_corpus_filtering_and_errors():
    sample = (
        "# comment line\n"
        "Guard: Halt, traveler! State your name.\n"
        "Merchant: Fresh potions for sale!\n"
    )
    all_sents = preprocess_corpus(sample)
    assert all_sents == [
        ["halt", "traveler"],
        ["state", "your", "name"],
        ["fresh", "potions", "for", "sale"],
    ]
    guard_sents = preprocess_corpus(sample, character="guard")
    assert guard_sents == [["halt", "traveler"], ["state", "your", "name"]]

    with pytest.raises(ValueError, match="Available characters"):
        preprocess_corpus(sample, character="Wizard")



@pytest.mark.skip(reason="pending milestone 2")
def test_milestone_2():
    raise NotImplementedError()

@pytest.mark.skip(reason="pending milestone 3")
def test_milestone_3():
    raise NotImplementedError()

@pytest.mark.skip(reason="pending milestone 4")
def test_milestone_4():
    raise NotImplementedError()
