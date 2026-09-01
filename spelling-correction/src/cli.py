"""
cli.py
------
PART 5: Live Interactive Application

A continuous terminal loop:
  - Prompts the user to type a sentence.
  - On Enter, prints the corrected sentence with any CHANGED words
    highlighted (ANSI colour if the terminal supports it, and always
    wrapped in **asterisks** as a plain-text fallback).
  - Prints the latency (in milliseconds) of the correction.
  - Exits when the user types "exit".

Run:
    python -m src.cli
"""

import os
import re
import time

from .model_builder import build_model, save_model, load_model, MODEL_PATH_DEFAULT
from .corrector import SpellingCorrector
from .utils import ensure_brown_downloaded


GREEN = "\033[92m"
RESET = "\033[0m"

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_preserving_punctuation(sentence: str):
    """Very light tokenizer that keeps punctuation as separate tokens."""
    return TOKEN_RE.findall(sentence)


def detokenize(tokens):
    """Rejoin tokens into a sentence, avoiding a space before punctuation."""
    out = ""
    for tok in tokens:
        if out and not re.match(r"^[^\w\s]$", tok):
            out += " "
        out += tok
    return out


def highlight(word, use_color: bool):
    if use_color:
        return f"{GREEN}**{word}**{RESET}"
    return f"**{word}**"


def load_or_build_model():
    if os.path.exists(MODEL_PATH_DEFAULT):
        return load_model()
    print("No cached model found — building from the Brown corpus "
          "(this only happens once) ...")
    ensure_brown_downloaded()
    model = build_model()
    save_model(model)
    return model


def run_cli():
    model = load_or_build_model()
    corrector = SpellingCorrector(model)
    use_color = os.isatty(1) if hasattr(os, "isatty") else False

    print("\n=== Spelling Corrector — Interactive CLI ===")
    print("Type a sentence and press Enter to correct it.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            sentence = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if sentence.lower() == "exit":
            print("Goodbye!")
            break
        if not sentence:
            continue

        tokens = tokenize_preserving_punctuation(sentence)

        t0 = time.perf_counter()
        results = corrector.correct_sentence(tokens)
        latency_ms = (time.perf_counter() - t0) * 1000

        display_tokens = [
            highlight(corrected, use_color) if changed else corrected
            for (_orig, corrected, changed) in results
        ]
        corrected_sentence = detokenize(display_tokens)
        any_changed = any(changed for (_o, _c, changed) in results)

        print(corrected_sentence)
        if not any_changed:
            print("(no changes made)")
        print(f"[latency: {latency_ms:.2f} ms]\n")


if __name__ == "__main__":
    run_cli()
