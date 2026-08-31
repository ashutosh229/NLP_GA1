"""
utils.py
--------
Small shared helper functions used across the project:
- Downloading / loading the NLTK Brown corpus
- Basic text cleaning
- Random single-edit "typo" generation (used for building test sets and
  for the Speed Demon benchmark batch)
"""

import random
import string

ALPHABET = string.ascii_lowercase


def ensure_brown_downloaded():
    """Make sure the Brown corpus is available locally, download if not."""
    import nltk
    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)

def clean_token(token: str):
    """
    Lowercase a token and keep it only if it is a purely alphabetic word.
    Punctuation-only tokens (Brown corpus has many, e.g. '.', ',', '``')
    are filtered out upstream by the caller.
    """
    return token.lower()

def is_alpha_word(token: str) -> bool:
    return token.isalpha()


def random_single_edit(word: str, rng: random.Random = random) -> str:
    """
    Apply ONE random edit operation (deletion, insertion, substitution,
    or transposition) to `word` and return the resulting string.

    This is used to synthesize misspellings for:
      * the evaluation test set (Part 4)
      * the 1,000-word Speed Demon benchmark batch (Part 4)
    """
    if len(word) < 2:
        # Too short to transpose; just insert or substitute.
        ops = ["insert", "substitute"]
    else:
        ops = ["delete", "insert", "substitute", "transpose"]

    op = rng.choice(ops)
    i = rng.randrange(len(word))

    if op == "delete":
        return word[:i] + word[i + 1:]

    if op == "insert":
        ch = rng.choice(ALPHABET)
        pos = rng.randrange(len(word) + 1)
        return word[:pos] + ch + word[pos:]

    if op == "substitute":
        ch = rng.choice([c for c in ALPHABET if c != word[i]] or [word[i]])
        return word[:i] + ch + word[i + 1:]

    if op == "transpose":
        j = min(i + 1, len(word) - 1)
        if i == j:
            j = max(i - 1, 0)
        w = list(word)
        w[i], w[j] = w[j], w[i]
        return "".join(w)

    return word 
