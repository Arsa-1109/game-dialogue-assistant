# Game Dialogue Assistant

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/48u6da3eaq/badge.svg)](https://ask.thili.ai/b/48u6da3eaq)

The Game Dialogue Assistant helps game designers rapidly prototype and ensure character consistency by generating NPC dialogue snippets. It leverages an n-gram language model, trained on game script corpora, to produce coherent and contextually relevant text, showcasing advanced text generation capabilities.

## What you are building

**Goal.** You will build a Game Dialogue Assistant, a tool that helps game designers generate dialogue snippets for NPCs based on a corpus of game scripts.

**Done when.** The project is done when you have a command-line interface where users can input a starting phrase and generate coherent dialogue snippets, proving the text generation capability of your n-gram model.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Collect and preprocess a corpus of game dialogues.**  
      Hint: Apply text cleaning and tokenization techniques to prepare your corpus.
- [ ] **2. Build the n-gram language model with smoothing.**  
      Hint: Implement the n-gram model and apply smoothing, as covered in the 'Build a Next-Word Predictor from Scratch' drill.
- [ ] **3. Implement dialogue generation logic using the model.**  
      Hint: Use the next-word prediction logic from the drill to generate sequences of words.
- [ ] **4. Design a CLI where users can input a starting phrase and generate dialogue.**  
      Hint: Create a command-line interface to take user input and display generated text.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to specify the desired length of the generated dialogue.
- Provide multiple generated dialogue options for a given starting phrase.
- Implement a simple way to save generated dialogues to a file.
