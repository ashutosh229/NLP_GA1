"""
language_model.py
-----------------
Part 3: Shared Smoothed N-Gram Language Model
- Add-k smoothed Bigram and Trigram models trained on the Brown Corpus
- Provides sentence log-probability and perplexity
- Provides sliding window perplexity for real-time [GRAMMAR-ALERT]
- Reusable across sub-systems
"""

import math
from collections import Counter
from typing import List, Tuple, Dict, Set, Optional


BOS = "<s>"
EOS = "</s>"


class AddKSmoothedLM:
    """
    Bigram and Trigram Language Model with Add-k (Lidstone) smoothing.
    P(w_i | w_{i-1}) = (C(w_{i-1}, w_i) + k) / (C(w_{i-1}) + k * |V|)
    P(w_i | w_{i-2}, w_{i-1}) = (C(w_{i-2}, w_{i-1}, w_i) + k) / (C(w_{i-2}, w_{i-1}) + k * |V|)
    """

    def __init__(self, k: float = 0.05):
        """
        k = 0.05 is chosen because Brown corpus vocabulary is large (~50k types).
        Standard Laplace (k=1.0) assigns too much probability mass to unseen transitions,
        artificially inflating perplexity. Smaller k (0.01 - 0.05) preserves sharp discriminations
        between grammatical sequences and ungrammatical combinations.
        """
        self.k = k
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.context1_counts = Counter()  # C(w1)
        self.context2_counts = Counter()  # C(w1, w2)
        self.vocab: Set[str] = set()
        self.vocab_size: int = 0
        self.total_tokens: int = 0

    def train(self, sentences: List[List[str]]):
        """
        Train bigram and trigram counts on a list of tokenized sentences.
        """
        for sent in sentences:
            if not sent:
                continue
            words = [w.lower() for w in sent]
            self.vocab.update(words)
            self.total_tokens += len(words)

            padded_sent = [BOS, BOS] + words + [EOS]

            for w in words:
                self.unigram_counts[w] += 1

            for w1, w2 in zip(padded_sent[1:], padded_sent[2:]):
                self.bigram_counts[(w1, w2)] += 1
                self.context1_counts[w1] += 1

            for w1, w2, w3 in zip(padded_sent, padded_sent[1:], padded_sent[2:]):
                self.trigram_counts[(w1, w2, w3)] += 1
                self.context2_counts[(w1, w2)] += 1

        self.vocab.add(BOS)
        self.vocab.add(EOS)
        self.vocab_size = len(self.vocab)

    def bigram_prob(self, w1: str, w2: str) -> float:
        w1_l = w1.lower()
        w2_l = w2.lower()
        count = self.bigram_counts.get((w1_l, w2_l), 0)
        context = self.context1_counts.get(w1_l, 0)
        return (count + self.k) / (context + self.k * self.vocab_size)

    def trigram_prob(self, w1: str, w2: str, w3: str) -> float:
        w1_l = w1.lower()
        w2_l = w2.lower()
        w3_l = w3.lower()
        count = self.trigram_counts.get((w1_l, w2_l, w3_l), 0)
        context = self.context2_counts.get((w1_l, w2_l), 0)
        return (count + self.k) / (context + self.k * self.vocab_size)

    def sentence_log_prob_bigram(self, words: List[str]) -> float:
        if not words:
            return 0.0
        padded = [BOS] + [w.lower() for w in words] + [EOS]
        log_prob = 0.0
        for w1, w2 in zip(padded, padded[1:]):
            p = self.bigram_prob(w1, w2)
            log_prob += math.log2(p)
        return log_prob

    def sentence_log_prob_trigram(self, words: List[str]) -> float:
        if not words:
            return 0.0
        padded = [BOS, BOS] + [w.lower() for w in words] + [EOS]
        log_prob = 0.0
        for w1, w2, w3 in zip(padded, padded[1:], padded[2:]):
            p = self.trigram_prob(w1, w2, w3)
            log_prob += math.log2(p)
        return log_prob

    def sentence_perplexity_bigram(self, words: List[str]) -> float:
        if not words:
            return float('inf')
        padded_len = len(words) + 1  # includes EOS transition
        log_prob = self.sentence_log_prob_bigram(words)
        return 2.0 ** (-log_prob / max(1, padded_len))

    def sentence_perplexity_trigram(self, words: List[str]) -> float:
        if not words:
            return float('inf')
        padded_len = len(words) + 1  # includes EOS transition
        log_prob = self.sentence_log_prob_trigram(words)
        return 2.0 ** (-log_prob / max(1, padded_len))

    def window_perplexity(self, window_words: List[str], use_trigram: bool = True) -> float:
        """
        Calculates the perplexity of a local window of N words.
        Used for live [GRAMMAR-ALERT] triggers.
        """
        if not window_words:
            return 0.0
        if len(window_words) < 2:
            return 1.0

        words = [w.lower() for w in window_words]
        log_prob = 0.0
        n_transitions = 0

        if use_trigram and len(words) >= 3:
            for w1, w2, w3 in zip(words, words[1:], words[2:]):
                p = self.trigram_prob(w1, w2, w3)
                log_prob += math.log2(p)
                n_transitions += 1
        else:
            for w1, w2 in zip(words, words[1:]):
                p = self.bigram_prob(w1, w2)
                log_prob += math.log2(p)
                n_transitions += 1

        if n_transitions == 0:
            return 1.0

        return 2.0 ** (-log_prob / n_transitions)
