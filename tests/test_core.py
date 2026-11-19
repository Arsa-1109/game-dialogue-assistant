"""Unit and CLI tests for the Game Dialogue Assistant (Milestones 1-4)."""
import json

import pytest

from app.core import (
    END_TOKEN, UNK_TOKEN, NGramModel, detokenize, generate_dialogue,
    generate_options, list_characters, load_corpus, milestone_1,
    milestone_2, milestone_3, preprocess_corpus, save_dialogues, tokenize,
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


# Milestone 2: Build the n-gram language model with smoothing.
def test_milestone_2():
    corpus = [
        ["halt", "traveler", "state", "your", "business"],
        ["halt", "traveler", "show", "your", "papers"],
        ["stay", "alert", "near", "the", "gate"],
    ]
    model = milestone_2(sentences=corpus, n=2, smoothing=1.0)
    assert model.n == 2
    assert UNK_TOKEN in model.vocab
    assert END_TOKEN in model.vocab

    dist = model.probabilities(["halt"])
    assert pytest.approx(sum(dist.values()), rel=1e-9) == 1.0
    assert dist["traveler"] > dist["alert"]

    # Unseen word in unseen context still receives strictly positive Laplace probability
    unseen_prob = model.probability("dragon", ["unseen"])
    expected_unseen = 1.0 / (0 + 1.0 * len(model.vocab))
    assert pytest.approx(unseen_prob, rel=1e-9) == expected_unseen

    # Perplexity remains finite even on unseen sentences thanks to smoothing
    ppl = model.perplexity([["dragon", "flies", "over", "gate"]])
    assert ppl > 1.0


def test_ngram_model_validation():
    with pytest.raises(ValueError):
        NGramModel(n=0)
    with pytest.raises(ValueError):
        NGramModel(n=2, smoothing=0.0)
    with pytest.raises(ValueError):
        NGramModel(n=2, smoothing=1.0).fit([])


# Milestone 3: Implement dialogue generation logic using the model.
def test_milestone_3():
    model = milestone_2(n=3, smoothing=0.5)
    options = milestone_3(prompt="Halt traveler", length=16, options=3, model=model, seed=10)
    assert len(options) == 3
    for snippet in options:
        assert snippet.lower().startswith("halt traveler")
        assert snippet.endswith(".")
        assert len(snippet.split()) <= 16

    # Deterministic with fixed seed
    again = milestone_3(prompt="Halt traveler", length=16, options=3, model=model, seed=10)
    assert options == again


def test_generation_backoff_and_detokenize():
    corpus = preprocess_corpus(
        "Guard: Halt traveler state your business at the gate.\n"
        "Guard: Watch your step near the old bridge.\n"
    )
    model = NGramModel(n=3, smoothing=1.0).fit(corpus)
    # Prompt has an unseen word; backoff should still produce a valid continuation
    text = generate_dialogue(model, prompt="strange traveler", length=10, seed=7)
    assert text.lower().startswith("strange traveler")
    assert len(text.split()) <= 10

    assert detokenize([["i", "am", "ready"], ["i'm", "here"]]) == "I am ready. I'm here."


def test_generate_options_and_save(tmp_path):
    model = milestone_2(n=3, smoothing=0.5)
    opts = generate_options(model, prompt="Welcome friend", count=3, length=15, seed=1)
    assert len(opts) == 3

    txt_path = save_dialogues(tmp_path / "dialogues.txt", "Welcome friend", opts, {"character": "Merchant"})
    assert txt_path.exists()
    content = txt_path.read_text(encoding="utf-8")
    assert "Prompt: Welcome friend" in content
    assert "1. " in content and "3. " in content

    json_path = save_dialogues(tmp_path / "dialogues.json", "Welcome friend", opts, {"n": 3})
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["prompt"] == "Welcome friend"
    assert len(data["dialogues"]) == 3



@pytest.mark.skip(reason="pending milestone 4")
def test_milestone_4():
    raise NotImplementedError()
