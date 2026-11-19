"""Unit and CLI tests for the Game Dialogue Assistant (Milestones 1-4)."""
import json

import pytest

from app import cli
from app.core import (
    END_TOKEN,
    UNK_TOKEN,
    NGramModel,
    detokenize,
    evaluate_models,
    generate_dialogue,
    generate_options,
    list_characters,
    load_corpus,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    preprocess_corpus,
    save_dialogues,
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


# Milestone 4: Design a CLI where users can input a starting phrase and generate dialogue.
def test_milestone_4(capsys, tmp_path, monkeypatch):
    save_file = tmp_path / "out.txt"
    store_file = tmp_path / "store.json"
    monkeypatch.setenv("DATA_PATH", str(store_file))

    cli.main(
        [
            "generate",
            "--prompt",
            "Halt traveler",
            "--character",
            "Guard",
            "--length",
            "15",
            "--options",
            "2",
            "--seed",
            "42",
            "--save",
            str(save_file),
            "--store",
        ]
    )
    out = capsys.readouterr().out
    assert "Character: Guard" in out
    assert "1. Halt traveler" in out
    assert "2. Halt traveler" in out
    assert save_file.exists()

    cli.main(["history"])
    hist_out = capsys.readouterr().out
    assert "Halt traveler" in hist_out


def test_cli_interactive_prompt_and_stats(capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Welcome friend")
    cli.main(["generate", "--character", "Merchant", "--length", "14", "--options", "2", "--seed", "5"])
    out = capsys.readouterr().out
    assert "Character: Merchant" in out
    assert "1. Welcome friend" in out

    cli.main(["stats", "--character", "Mage"])
    stats_out = capsys.readouterr().out
    assert "Filter: Mage" in stats_out
    assert "Training Perplexity:" in stats_out


def test_milestone_4_api_and_cli_error():
    res = milestone_4(prompt="Good steel", character="Blacksmith", length=14, options=2, seed=9)
    assert res["character"] == "Blacksmith"
    assert len(res["dialogues"]) == 2
    assert all(line.lower().startswith("good steel") for line in res["dialogues"])

    with pytest.raises(SystemExit) as exc:
        cli.main(["generate", "--prompt", "Hello", "--character", "UnknownNPC"])
    assert exc.value.code == 1


def test_interactive_inline_and_angle_bracket_commands(capsys, monkeypatch, tmp_path):
    save_file = tmp_path / "inline_saved.txt"
    inputs = iter([
        f"How much for a room? :char <Merchant> :save {save_file}",
        ":char Guard",
        "Halt traveler",
        ":quit",
    ])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    cli.main(["interactive", "--length", "22", "--options", "2"])
    out = capsys.readouterr().out
    assert "Character set to: Merchant" in out
    assert "Prompt: 'How much for a room?' | Character: Merchant" in out
    assert "Character set to: Guard" in out
    assert "Prompt: 'Halt traveler' | Character: Guard" in out
    assert save_file.exists()
    saved_text = save_file.read_text(encoding="utf-8")
    assert "Prompt: How much for a room?" in saved_text
    assert "character=Merchant" in saved_text


def test_question_punctuation_distinct_options_and_no_dangling_stopwords():
    innkeeper_res = milestone_4(
        prompt="how much for a room?",
        character="Innkeeper",
        length=22,
        options=3,
        seed=42,
    )
    innkeeper_lines = innkeeper_res["dialogues"]
    assert len(set(innkeeper_lines)) == 3
    for line in innkeeper_lines:
        assert line.startswith("How much for a room?")
        assert len(line.split()) <= 22

    merchant_q = milestone_4(
        prompt="how much for a room?",
        character="Merchant",
        length=22,
        options=3,
        seed=42,
    )
    assert len(set(merchant_q["dialogues"])) == 3
    for line in merchant_q["dialogues"]:
        assert line.startswith("How much for a room?")
        assert "room the" not in line.lower()

    merchant_no_q = milestone_4(
        prompt="how much for a room",
        character="Merchant",
        length=22,
        options=3,
        seed=42,
    )
    for line in merchant_no_q["dialogues"]:
        assert "room the" not in line.lower()

    # Completely out-of-corpus prompt should close cleanly rather than gluing unigrams
    oov_res = milestone_4(
        prompt="where is the spaceship",
        character="Guard",
        length=20,
        options=3,
        seed=11,
    )
    for line in oov_res["dialogues"]:
        assert line.startswith("Where is the spaceship.")

    dangling = {"and.", "the.", "will.", "for.", "or.", "but.", "to.", "of.", "in."}
    for prompt_text, npc in [
        ("Halt traveler", "Guard"),
        ("Welcome friend", "Merchant"),
        ("The arcane winds", "Mage"),
        ("Good steel", "Blacksmith"),
        ("Come in from the cold", "Innkeeper"),
    ]:
        res = milestone_4(prompt=prompt_text, character=npc, length=18, options=3, seed=42)
        for line in res["dialogues"]:
            last_word = line.lower().split()[-1]
            assert last_word not in dangling, f"Dangling ending in: {line}"


def test_evaluate_models_and_cli_eval(capsys):
    report = evaluate_models(smoothing=0.05, seed=42)
    assert report["train_sentences"] > report["test_sentences"] > 0
    assert [row["order"] for row in report["results"]] == [1, 2, 3]
    unigram, bigram, trigram = report["results"]
    assert bigram["train_perplexity"] < unigram["train_perplexity"]
    assert trigram["train_perplexity"] < bigram["train_perplexity"]

    cli.main(["eval", "--smoothing", "0.05"])
    out = capsys.readouterr().out
    assert "Unigram baseline" in out
    assert "3-gram" in out
    assert "Held-out PPL" in out
