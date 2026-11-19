"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core import (
    DEFAULT_CORPUS_PATH,
    NGramModel,
    list_characters,
    load_corpus,
    milestone_4,
    save_dialogues,
)
from app.storage import Store


def _print_result(result: dict) -> None:
    header = (
        f"Prompt: {result['prompt']!r} | "
        f"Character: {result['character']} | "
        f"Model: {result['n']}-gram (k={result['smoothing']}, vocab={result['vocab_size']})"
    )
    print(header)
    for idx, line in enumerate(result["dialogues"], start=1):
        print(f"  {idx}. {line}")
    if result.get("saved_to"):
        print(f"Saved dialogues to: {result['saved_to']}")


def _record_in_store(result: dict, store_path: str | None = None) -> None:
    store = Store(store_path)
    history = store.get("generations", [])
    history.append(
        {
            "prompt": result["prompt"],
            "character": result["character"],
            "n": result["n"],
            "smoothing": result["smoothing"],
            "dialogues": result["dialogues"],
        }
    )
    store.put("generations", history)


def _clean_arg(raw_val: str) -> str:
    val = raw_val.strip()
    if len(val) >= 2 and val.startswith("<") and val.endswith(">"):
        val = val[1:-1].strip()
    return val


def _extract_inline_commands(raw: str) -> tuple[str, list[str]]:
    """Split a prompt and any inline `:cmd arg` directives on the same line."""
    import re

    cmd_re = re.compile(
        r"(?:^|\s)(:(?:char|len|opts|save)(?:\s+\S+)?|:(?:quit|exit|q)(?=\s|$))"
    )
    cmds: list[str] = []
    first_match = cmd_re.search(raw)
    if not first_match:
        return raw.strip(), []
    prompt_part = raw[: first_match.start()].strip()
    rest = raw[first_match.start() :].strip()
    for match in cmd_re.finditer(" " + rest):
        cmds.append(match.group(1).strip())
    return prompt_part, cmds


def _save_result_to(result: dict, target: Path) -> None:
    saved = save_dialogues(
        target,
        prompt=result["prompt"],
        dialogues=result["dialogues"],
        metadata={
            "character": result["character"],
            "n": result["n"],
            "smoothing": result["smoothing"],
        },
    )
    print(f"Saved output to: {saved}")


def _run_interactive(args: argparse.Namespace) -> None:
    character = args.character
    length = args.length
    options = args.options
    n = args.n
    smoothing = args.smoothing
    corpus_path = args.corpus

    chars = list_characters(corpus_path)
    char_lookup = {c.lower(): c for c in chars}
    print("Game Dialogue Assistant (Interactive Mode)")
    print(f"Available characters: {', '.join(chars) if chars else 'none (plain corpus)'}")
    print("Tip: n-gram predictor — continues incomplete phrases, or starts a fresh NPC sentence after . / ? / !")
    print("Commands (standalone or at the end of a line):")
    print("  :char Merchant | :char all | :len 18 | :opts 3 | :save out.txt | :quit")

    last_result: dict | None = None
    while True:
        try:
            raw = input("\nStarting phrase> ").strip()
        except EOFError:
            break
        if not raw:
            continue

        prompt_part, cmds = _extract_inline_commands(raw)
        should_quit = False
        had_error = False
        pending_saves: list[Path] = []

        for cmd in cmds:
            if cmd in {":quit", ":exit", ":q"}:
                should_quit = True
                break
            if cmd.startswith(":char"):
                parts = cmd.split(None, 1)
                if len(parts) < 2:
                    print("Usage: :char <name|all> (e.g. :char Merchant)")
                    had_error = True
                    continue
                val = _clean_arg(parts[1])
                if val.lower() == "all":
                    character = None
                    print("Character set to: all")
                elif val.lower() in char_lookup:
                    character = char_lookup[val.lower()]
                    print(f"Character set to: {character}")
                else:
                    print(f"Unknown character '{val}'. Available: {', '.join(chars) if chars else 'none'}")
                    had_error = True
            elif cmd.startswith(":len"):
                parts = cmd.split(None, 1)
                if len(parts) < 2:
                    print("Usage: :len <int>")
                    had_error = True
                    continue
                try:
                    parsed = int(_clean_arg(parts[1]))
                    if parsed < 1:
                        raise ValueError
                    length = parsed
                    print(f"Target length set to: {length}")
                except ValueError:
                    print("Invalid length. Example: :len 18")
                    had_error = True
            elif cmd.startswith(":opts"):
                parts = cmd.split(None, 1)
                if len(parts) < 2:
                    print("Usage: :opts <int>")
                    had_error = True
                    continue
                try:
                    parsed = int(_clean_arg(parts[1]))
                    if parsed < 1:
                        raise ValueError
                    options = parsed
                    print(f"Options count set to: {options}")
                except ValueError:
                    print("Invalid options count. Example: :opts 3")
                    had_error = True
            elif cmd.startswith(":save"):
                parts = cmd.split(None, 1)
                if len(parts) < 2:
                    print("Usage: :save <path>")
                    had_error = True
                    continue
                target = Path(_clean_arg(parts[1]))
                if prompt_part:
                    pending_saves.append(target)
                else:
                    if last_result is None:
                        print("Generate a dialogue first before saving.")
                        had_error = True
                        continue
                    _save_result_to(last_result, target)

        if should_quit:
            break
        if had_error or not prompt_part:
            continue

        last_result = milestone_4(
            prompt=prompt_part,
            length=length,
            options=options,
            character=character,
            n=n,
            smoothing=smoothing,
            corpus_path=corpus_path,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        _print_result(last_result)
        for target in pending_saves:
            _save_result_to(last_result, target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="game-dialogue-assistant",
        description=(
            "Game designers use this tool to generate dialogue snippets for NPCs "
            "based on a corpus of game scripts, ensuring character consistency and rapid prototyping."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hello", help="check that everything is wired up")

    gen = sub.add_parser("generate", help="generate NPC dialogue snippets from a starting phrase")
    gen.add_argument("-p", "--prompt", default=None, help="starting phrase for dialogue generation")
    gen.add_argument("-l", "--length", type=int, default=18, help="target max words per snippet (default: 18)")
    gen.add_argument("-o", "--options", type=int, default=3, help="number of dialogue options to generate (default: 3)")
    gen.add_argument("-c", "--character", default=None, help="filter corpus by NPC character (e.g. Guard, Merchant)")
    gen.add_argument("--n", type=int, default=3, help="n-gram order (default: 3)")
    gen.add_argument("-k", "--smoothing", type=float, default=1.0, help="add-k smoothing constant (default: 1.0)")
    gen.add_argument("--temperature", type=float, default=0.9, help="sampling temperature (default: 0.9)")
    gen.add_argument("--top-k", type=int, default=8, help="top-k candidates during sampling (default: 8)")
    gen.add_argument("--corpus", default=None, help=f"path to custom script corpus (default: {DEFAULT_CORPUS_PATH})")
    gen.add_argument("--seed", type=int, default=None, help="random seed for reproducible output")
    gen.add_argument("-s", "--save", default=None, help="save generated options to a .txt or .json file")
    gen.add_argument("--store", action="store_true", help="also record generation in local JSON store")

    inter = sub.add_parser("interactive", help="start an interactive prompt loop for rapid prototyping")
    inter.add_argument("-l", "--length", type=int, default=18, help="target max words per snippet (default: 18)")
    inter.add_argument("-o", "--options", type=int, default=3, help="number of dialogue options per prompt (default: 3)")
    inter.add_argument("-c", "--character", default=None, help="initial NPC character filter")
    inter.add_argument("--n", type=int, default=3, help="n-gram order (default: 3)")
    inter.add_argument("-k", "--smoothing", type=float, default=1.0, help="add-k smoothing constant (default: 1.0)")
    inter.add_argument("--temperature", type=float, default=0.9, help="sampling temperature (default: 0.9)")
    inter.add_argument("--top-k", type=int, default=8, help="top-k candidates during sampling (default: 8)")
    inter.add_argument("--corpus", default=None, help="path to custom script corpus")

    stats = sub.add_parser("stats", help="inspect corpus statistics, characters, and model perplexity")
    stats.add_argument("-c", "--character", default=None, help="filter stats to a specific NPC character")
    stats.add_argument("--n", type=int, default=3, help="n-gram order (default: 3)")
    stats.add_argument("-k", "--smoothing", type=float, default=1.0, help="add-k smoothing constant (default: 1.0)")
    stats.add_argument("--corpus", default=None, help="path to custom script corpus")

    sub.add_parser("history", help="show previously stored dialogue generations")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "hello":
            print("Game Dialogue Assistant is ready. Open README.md for your milestones.")
            return

        if args.command == "generate":
            prompt = args.prompt
            if prompt is None:
                prompt = input("Starting phrase: ").strip()
            result = milestone_4(
                prompt=prompt,
                length=args.length,
                options=args.options,
                character=args.character,
                n=args.n,
                smoothing=args.smoothing,
                corpus_path=args.corpus,
                seed=args.seed,
                save_path=args.save,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            if args.store:
                _record_in_store(result)
            _print_result(result)
            return

        if args.command == "interactive":
            _run_interactive(args)
            return

        if args.command == "stats":
            chars = list_characters(args.corpus)
            sentences = load_corpus(corpus_path=args.corpus, character=args.character)
            model = NGramModel(n=args.n, smoothing=args.smoothing).fit(sentences)
            ppl = model.perplexity(sentences)
            print(f"Corpus: {args.corpus or DEFAULT_CORPUS_PATH}")
            print(f"Characters ({len(chars)}): {', '.join(chars) if chars else 'none'}")
            print(f"Filter: {args.character or 'all'}")
            print(f"Sentences: {model.sentence_count} | Tokens: {model.token_count} | Vocab: {len(model.vocab)}")
            print(f"Model: {args.n}-gram (smoothing k={args.smoothing}) | Training Perplexity: {ppl:.2f}")
            return

        if args.command == "history":
            store = Store()
            entries = store.get("generations", [])
            if not entries:
                print("No saved generations in store.")
                return
            for idx, entry in enumerate(entries, start=1):
                print(f"[{idx}] Prompt: {entry['prompt']!r} ({entry['character']}, {entry['n']}-gram)")
                for opt_idx, line in enumerate(entry["dialogues"], start=1):
                    print(f"    {opt_idx}. {line}")
            return
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
