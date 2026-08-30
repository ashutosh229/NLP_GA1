"""
Single entry point that runs the whole pipeline end to end:
    1. Train the classifier on en_ewt-ud-train.conllu (Parts 1 & 2)
    2. Evaluate LAS/UAS on en_ewt-ud-dev.conllu (Part 3)
    3. Parse the three example sentences from the assignment

Usage:
    python3 main.py
"""

import os

import train
import evaluate
from parser import DependencyParser

TRAIN_PATH = "UD_English-EWT/en_ewt-ud-train.conllu"
DEV_PATH = "UD_English-EWT/en_ewt-ud-dev.conllu"
MODEL_PATH = "model.pkl"

EXAMPLE_SENTENCES = [
    (["The", "cat", "sat", "on", "the", "mat", "."],
     ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN", "PUNCT"]),
    (["She", "eats", "a", "green", "salad", "."],
     ["PRON", "VERB", "DET", "ADJ", "NOUN", "PUNCT"]),
    (["I", "saw", "the", "man", "with", "a", "telescope", "."],
     ["PRON", "VERB", "DET", "NOUN", "ADP", "DET", "NOUN", "PUNCT"]),
]


def main():
    if not os.path.exists(TRAIN_PATH):
        raise SystemExit(
            f"Could not find {TRAIN_PATH}.\n"
            "Download the treebank first:\n"
            "  git clone --depth 1 https://github.com/UniversalDependencies/UD_English-EWT.git"
        )

    print("=" * 60)
    print("PART 1 & 2: Training the transition classifier")
    print("=" * 60)
    train.main()

    print()
    print("=" * 60)
    print("PART 3: Evaluating on the dev set")
    print("=" * 60)
    evaluate.evaluate(MODEL_PATH, DEV_PATH)

    print()
    print("=" * 60)
    print("Demo: parsing the example sentences")
    print("=" * 60)
    parser = DependencyParser(MODEL_PATH)
    for words, upos in EXAMPLE_SENTENCES:
        arcs = sorted(parser.parse(words, upos), key=lambda a: a[1])
        print("\nSentence:", " ".join(words))
        for head, dep, label in arcs:
            head_word = "ROOT" if head == 0 else words[head - 1]
            print(f"  {words[dep-1]:<10} <-- {label:<10} -- {head_word}")


if __name__ == "__main__":
    main()
