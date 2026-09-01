"""
evaluate.py
-----------
PART 4: Evaluation and "Speed Demon" Benchmark

1. Test Set Generation: take 10% of Brown corpus sentences; for each,
   randomly pick one alphabetic word and introduce a single-edit mistake,
   producing:
     - a NON-WORD error version (the typo is not in the vocabulary), and
     - a REAL-WORD error version (the typo happens to also be a real,
       different vocabulary word).

2. Accuracy: run the corrector on both test sets to report accuracy
   (does the correction match the original word?).

3. Speed Demon Benchmark: build a batch of exactly 1,000 misspelled
   words, time Method A candidate generation vs Method B candidate
   generation over the identical batch, and print a conclusion.

Run directly:
    python -m src.evaluate
(builds the model first if model.pkl is not already present)
"""

import os
import random
import time

from .model_builder import build_model, save_model, load_model, MODEL_PATH_DEFAULT
from .candidate_generation import method_a_candidates, method_b_candidates
from .corrector import SpellingCorrector
from .utils import ensure_brown_downloaded, is_alpha_word, random_single_edit

RNG = random.Random(42)  

# Part 4.1: Test set generation
def generate_test_sets(sentences, vocab, sample_fraction=0.10, max_attempts=20):
    """
    Returns two lists of test cases:
      nonword_cases: list of dicts {tokens, index, original, typo}
      realword_cases: list of dicts {tokens, index, original, typo}

    `tokens` is the full (lowercased, alpha-filtered) sentence, `index` is
    the position of the manipulated word within `tokens`, `original` is
    the correct word, `typo` is the corrupted word actually inserted at
    that position.
    """
    eligible_sents = [
        [w.lower() for w in s if is_alpha_word(w)]
        for s in sentences
    ]
    eligible_sents = [s for s in eligible_sents if len(s) >= 2]

    sample_size = int(len(eligible_sents) * sample_fraction)
    sampled = RNG.sample(eligible_sents, sample_size)

    nonword_cases = []
    realword_cases = []

    for tokens in sampled:
        idx = RNG.randrange(len(tokens))
        original = tokens[idx]

        # --- non-word error version ---
        for _ in range(max_attempts):
            typo = random_single_edit(original, RNG)
            if typo not in vocab and typo != original:
                nonword_cases.append({
                    "tokens": list(tokens), "index": idx,
                    "original": original, "typo": typo,
                })
                break

        # --- real-word error version ---
        for _ in range(max_attempts):
            typo = random_single_edit(original, RNG)
            if typo in vocab and typo != original:
                realword_cases.append({
                    "tokens": list(tokens), "index": idx,
                    "original": original, "typo": typo,
                })
                break
        

    return nonword_cases, realword_cases


# Part 4.2: Accuracy
def evaluate_nonword_accuracy(corrector, cases):
    correct = 0
    for case in cases:
        pred, _ = corrector.correct_nonword(case["typo"])
        if pred == case["original"]:
            correct += 1
    return correct / len(cases) if cases else 0.0


def evaluate_realword_accuracy(corrector, cases):
    correct = 0
    for case in cases:
        tokens, idx = case["tokens"], case["index"]
        corrupted = list(tokens)
        corrupted[idx] = case["typo"]

        prev_w = corrupted[idx - 1] if idx > 0 else None
        next_w = corrupted[idx + 1] if idx + 1 < len(corrupted) else None

        pred, _ = corrector.correct_realword(prev_w, case["typo"], next_w)
        if pred == case["original"]:
            correct += 1
    return correct / len(cases) if cases else 0.0

# Entry point
def main():
    ensure_brown_downloaded()
    from nltk.corpus import brown

    if os.path.exists(MODEL_PATH_DEFAULT):
        print(f"[evaluate] Loading cached model from {MODEL_PATH_DEFAULT} ...")
        model = load_model()
    else:
        print("[evaluate] No cached model found, building from scratch ...")
        model = build_model()
        save_model(model)

    corrector = SpellingCorrector(model)
    sentences = list(brown.sents())

    print("[evaluate] Generating test sets (10% of Brown corpus sentences) ...")
    nonword_cases, realword_cases = generate_test_sets(sentences, model["vocab"])
    print(f"  Non-word test cases:  {len(nonword_cases)}")
    print(f"  Real-word test cases: {len(realword_cases)}")

    nonword_acc = evaluate_nonword_accuracy(corrector, nonword_cases)
    realword_acc = evaluate_realword_accuracy(corrector, realword_cases)

    print(f"\n[Accuracy] Non-word error correction:  {nonword_acc:.2%}")
    print(f"[Accuracy] Real-word error correction: {realword_acc:.2%}")


if __name__ == "__main__":
    main()
