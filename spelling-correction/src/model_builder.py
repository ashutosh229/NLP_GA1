"""
model_builder.py
-----------------
PART 1: Corpus and Model Preparation

Builds, from the NLTK Brown corpus:
  1. A vocabulary of unique words + a unigram frequency distribution.
  2. A bigram probability model (with add-one / Laplace smoothing) used
     later for real-word error correction.
  3. The Symmetric-Delete ("SymSpell") preprocessing dictionary needed by
     candidate generation Method B.

Run this file directly to build and cache the model:
    python -m src.model_builder
It writes `model.pkl` to the project root.
"""

import pickle
import time
from collections import Counter, defaultdict

from .utils import ensure_brown_downloaded, is_alpha_word


MODEL_PATH_DEFAULT = "model.pkl"


def build_vocab_and_unigrams(sentences):
    """
    Part 1.1: Vocabulary and Frequencies.
    Returns (vocab:set, unigram_counts:Counter, total_tokens:int)
    """
    unigram_counts = Counter()
    for sent in sentences:
        for tok in sent:
            tok = tok.lower()
            if is_alpha_word(tok):
                unigram_counts[tok] += 1

    vocab = set(unigram_counts.keys())
    total_tokens = sum(unigram_counts.values())
    return vocab, unigram_counts, total_tokens


def build_bigram_model(sentences, vocab):
    """
    Part 1.2: Language Model (bigram).

    Returns:
      bigram_counts: Counter[(w1, w2)] -> count
      context_counts: Counter[w1] -> total count of w1 as the first word
                      of a bigram (denominator for P(w2|w1))
    """
    bigram_counts = Counter()
    context_counts = Counter()

    for sent in sentences:
        words = [w.lower() for w in sent if is_alpha_word(w)]
        for w1, w2 in zip(words, words[1:]):
            if w1 in vocab and w2 in vocab:
                bigram_counts[(w1, w2)] += 1
                context_counts[w1] += 1

    return bigram_counts, context_counts


def bigram_prob(w1, w2, bigram_counts, context_counts, vocab_size):
    """
    P(w2 | w1) with add-one (Laplace) smoothing:
        (count(w1,w2) + 1) / (count(w1) + V)
    """
    return (bigram_counts.get((w1, w2), 0) + 1) / (context_counts.get(w1, 0) + vocab_size)


def build_deletes_dict(vocab):
    """
    Part 2 preprocessing (Method B - Symmetric Delete):
    Map every possible one-character deletion of every vocabulary word back
    to the set of original word(s) it came from.

    e.g. 'ello' -> {'hello'}, 'hllo' -> {'hello'}, ...

    Returns: dict[str, set[str]]
    """
    deletes_dict = defaultdict(set)
    for word in vocab:
        for i in range(len(word)):
            deletion = word[:i] + word[i + 1:]
            deletes_dict[deletion].add(word)
    return dict(deletes_dict)


def build_model(sentences=None, verbose=True):
    """
    Build the full model dictionary described at the top of this file.
    If `sentences` is None, uses the full Brown corpus.
    """
    ensure_brown_downloaded()
    from nltk.corpus import brown

    if sentences is None:
        sentences = list(brown.sents())

    t0 = time.time()
    vocab, unigram_counts, total_tokens = build_vocab_and_unigrams(sentences)
    if verbose:
        print(f"[model_builder] vocab size = {len(vocab)} "
              f"({time.time() - t0:.2f}s)")

    t0 = time.time()
    bigram_counts, context_counts = build_bigram_model(sentences, vocab)
    if verbose:
        print(f"[model_builder] bigram pairs = {len(bigram_counts)} "
              f"({time.time() - t0:.2f}s)")

    t0 = time.time()
    deletes_dict = build_deletes_dict(vocab)
    if verbose:
        print(f"[model_builder] deletes_dict entries = {len(deletes_dict)} "
              f"({time.time() - t0:.2f}s)")

    model = {
        "vocab": vocab,
        "unigram_counts": unigram_counts,
        "total_tokens": total_tokens,
        "bigram_counts": bigram_counts,
        "context_counts": context_counts,
        "deletes_dict": deletes_dict,
    }
    return model


def save_model(model, path=MODEL_PATH_DEFAULT):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path=MODEL_PATH_DEFAULT):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    m = build_model()
    save_model(m)
    print(f"[model_builder] Saved model to {MODEL_PATH_DEFAULT}")
