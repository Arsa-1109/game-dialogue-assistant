# Game Dialogue Assistant

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/48u6da3eaq/badge.svg)](https://ask.thili.ai/b/48u6da3eaq)

A lightweight, offline CLI tool for game designers to prototype character-consistent NPC dialogue from game script corpora using a statistical $n$-gram language model implemented from scratch in Python.

---

## Key Features

- **Character-Scoped Corpus Pipeline:** Parses `Speaker: Dialogue` script lines across five NPC archetypes (`Blacksmith`, `Guard`, `Innkeeper`, `Mage`, `Merchant`), normalises punctuation and casing, and tokenises contractions cleanly.
- **Smoothed $n$-Gram Language Model:** Built from scratch with configurable $n$-gram order, add-$k$ (Laplace) smoothing, `<UNK>` / `<s>` / `</s>` boundary tokens, perplexity evaluation, and interpolated order backoff ($n \to 2 \to 1$).
- **Controlled Decoding:** Supports temperature scaling, top-$k$ filtering, loop/stutter suppression, sentence-terminator preservation (`.` / `?` / `!`), and multi-option deduplication.
- **Interactive & Batch CLI:** Includes `generate`, `interactive` (with inline `:char`, `:len`, `:opts`, `:save` directives), `stats`, `history`, and `.txt` / `.json` export.

---

## Project Structure

```text
game-dialogue-assistant/
├── app/
│   ├── __init__.py
│   ├── cli.py                   # Argparse CLI & interactive prototyping loop
│   ├── core.py                  # Preprocessing, NGramModel, decoding, and export
│   ├── llm.py                   # Optional OpenAI-compatible chat client (.env)
│   ├── storage.py               # Local JSON store for generation history
│   └── corpus/
│       └── npc_dialogues.txt    # 309-sentence multi-NPC script corpus
├── tests/
│   ├── test_core.py             # Unit, model, and CLI integration tests
│   └── test_smoke.py            # Smoke tests for store, settings, and CLI
├── .github/workflows/ci.yml     # GitHub Actions pytest workflow
├── .env.example
├── pytest.ini
└── requirements.txt
```

---

## Quickstart

Requires **Python 3.12+**.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.cli hello
pytest -q
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m app.cli hello
.\.venv\Scripts\python.exe -m pytest -q
```

---

## CLI Reference

| Command | Description | Example |
| :--- | :--- | :--- |
| `hello` | Verify environment and CLI wiring | `python -m app.cli hello` |
| `stats` | Show corpus counts, characters, and training perplexity | `python -m app.cli stats --character Guard` |
| `eval` | Benchmark 2-gram & 3-gram against the unigram baseline | `python -m app.cli eval -k 0.05` |
| `generate` | Generate NPC dialogue continuations from a starting phrase | `python -m app.cli generate -p "Halt traveler" -c Guard -l 16 -o 3 --seed 42` |
| `interactive` | Launch an interactive prototyping REPL | `python -m app.cli interactive --character Merchant` |
| `history` | View previously stored generations (`--store`) | `python -m app.cli history` |

### Common `generate` Options

- `-p, --prompt` — Starting phrase for the NPC's line (or a closed sentence ending in `.`, `?`, or `!` to generate a follow-up line).
- `-c, --character` — Filter training corpus by NPC (`Blacksmith`, `Guard`, `Innkeeper`, `Mage`, `Merchant`).
- `-l, --length` — Maximum target word count per snippet (default: `18`).
- `-o, --options` — Number of distinct dialogue candidates to return (default: `3`).
- `--n` / `-k` — $n$-gram order (default: `3`) and add-$k$ smoothing constant (default: `1.0`).
- `--temperature` / `--top-k` — Sampling controls (defaults: `0.9` / `8`).
- `-s, --save` — Export generated options to `.txt` or `.json`.

---

## Sample Output

```text
$ python -m app.cli stats
Corpus: app/corpus/npc_dialogues.txt
Characters (5): Blacksmith, Guard, Innkeeper, Mage, Merchant
Filter: all
Sentences: 309 | Tokens: 3834 | Vocab: 1236
Model: 3-gram (smoothing k=1.0) | Training Perplexity: 534.99

$ python -m app.cli generate --prompt "Halt traveler" --character Guard --length 16 --options 3 --seed 42
Prompt: 'Halt traveler' | Character: Guard | Model: 3-gram (k=1.0, vocab=356)
  1. Halt traveler no weapons may remain drawn inside the city walls.
  2. Halt traveler and show your travel papers at the northern checkpoint.
  3. Halt traveler leave your saddlebags open for a room.

$ python -m app.cli generate --prompt "Welcome friend" --character Merchant --length 18 --options 3 --seed 42
Prompt: 'Welcome friend' | Character: Merchant | Model: 3-gram (k=1.0, vocab=371)
  1. Welcome friend take a look at these finely balanced blades and travel packs.
  2. Welcome friend and my scales never cheat an honest buyer.
  3. Welcome friend I will throw in a spare oil flask.
```

---

## Baseline Evaluation & Limitations

### Quantitative Evaluation vs. Unigram Baseline

Evaluated on a deterministic **80/20 train/held-out split** (`247` train / `62` test sentences; `20.0%` held-out OOV rate) via `python -m app.cli eval`:

| Model | `k = 0.05` Train PPL | `k = 0.05` Held-out PPL | `k = 1.0` Held-out PPL | Context Hit Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Unigram baseline (`n=1`)** | 362.28 | 714.46 | 459.51 | 100.0% |
| **Bigram (`n=2`)** | 50.81 | **406.35** | 710.94 | 81.5% |
| **Trigram (`n=3`)** | **50.04** | 635.58 | 896.48 | 39.9% |

**Key findings:**
- At $k = 0.05$, both **2-gram (406.35 PPL)** and **3-gram (635.58 PPL)** outperform the **Unigram baseline (714.46 PPL)**, with the 2-gram model cutting held-out perplexity by **43.1%** because **81.5%** of held-out contexts appear in training.
- At $k = 1.0$ (standard Laplace smoothing), adding $1.0 \cdot \lvert V \rvert$ to sparse bigram/trigram denominators steals too much probability mass from observed words, inflating trigram held-out perplexity (**896.48**) above unigram (**459.51**).
- Because exact 3-gram contexts match only **39.9%** of held-out positions, `NGramModel.predict_next()` uses **interpolated backoff** ($n \to 2 \to 1$) to stay grammatical without falling back prematurely to unigrams.

### Known Limitations

1. **Short Markov horizon ($n - 1 = 2$ words):** A trigram model conditions only on the previous two words, so it cannot enforce multi-sentence quest logic, speaker memory, or long-range plot constraints.
2. **Corpus sparsity & OOV sensitivity:** With 309 lines across 5 NPC archetypes (~60 lines each), prompts using out-of-domain vocabulary cannot continue mid-clause and instead close the prompt cleanly before starting a fresh in-domain NPC sentence.
3. **Line-completion model, not a conversational agent:** The predictor autocompletes lines spoken *by* the selected NPC rather than answering arbitrary player questions with semantic reasoning.
4. **Punctuation flattening:** Commas and internal quotation marks are stripped during regex tokenisation (`[a-z0-9]+(?:'[a-z0-9]+)*`), preserving only contractions and sentence-ending `.` / `?` / `!`.

---

## Testing

```bash
pytest -v
```

All 16 unit, evaluation, and CLI integration tests run offline with zero external API keys required.

## Configuration

Optional LLM endpoint settings live in `.env` (see `.env.example`), which is excluded from version control via `.gitignore`. The core $n$-gram predictor and CLI operate completely offline without `.env`.
