"""
corrector.py
------------
PART 3: Spelling Correction Logic

- Non-word error correction: word not in vocabulary -> generate candidates
  with BOTH Method A and Method B, pick the highest-unigram-frequency
  candidate.

- Real-word error correction: word IS in vocabulary but may be wrong given
  its context (e.g. "I ate an apply") -> generate ED1 candidates for the
  word, compare the bigram probability of the ORIGINAL phrase against each
  CANDIDATE phrase, and suggest a correction if a candidate is
  significantly more probable in context.
"""

import math

from .candidate_generation import method_a_candidates, method_b_candidates
from .model_builder import bigram_prob


class SpellingCorrector:
    def __init__(self, model):
        self.vocab = model["vocab"]
        self.unigram_counts = model["unigram_counts"]
        self.bigram_counts = model["bigram_counts"]
        self.context_counts = model["context_counts"]
        self.deletes_dict = model["deletes_dict"]
        self.vocab_size = len(self.vocab)

    # Candidate generation 
    def candidates_method_a(self, word):
        return method_a_candidates(word, self.vocab)

    def candidates_method_b(self, word):
        return method_b_candidates(word, self.vocab, self.deletes_dict)

    def all_candidates(self, word):
        """Union of both methods' candidate sets."""
        return self.candidates_method_a(word) | self.candidates_method_b(word)

    # Non-word error correction  (Part 3.1) 
    def correct_nonword(self, word):
        """
        `word` is assumed NOT to be in the vocabulary.
        Returns (best_correction, candidate_set). If no candidates were
        found at edit distance 1, returns (word, empty_set) unchanged.
        """
        candidates = self.all_candidates(word)
        if not candidates:
            return word, candidates

        best = max(candidates, key=lambda w: self.unigram_counts.get(w, 0))
        return best, candidates

    # Real-word error correction (Part 3.2)
    def _phrase_log_prob(self, prev_word, word, next_word):
        """
        log P(word | prev_word) + log P(next_word | word)
        Missing neighbours (start/end of sentence) are simply skipped.
        """
        score = 0.0
        if prev_word is not None:
            score += math.log(bigram_prob(prev_word, word,self.bigram_counts,self.context_counts,self.vocab_size))
        if next_word is not None:
            score += math.log(bigram_prob(word, next_word,self.bigram_counts,self.context_counts,self.vocab_size))
        return score

    def correct_realword(self, prev_word, word, next_word, log_margin=1.5):
        """
        `word` IS assumed to already be in the vocabulary. Compares the
        bigram log-probability of the sentence with `word` in place versus
        each ED1 candidate that is ALSO in the vocabulary. If some
        candidate beats the original by more than `log_margin` nats, it is
        suggested as a correction.

        Returns (suggested_word, was_changed: bool)
        """
        candidates = self.all_candidates(word)
        candidates = {c for c in candidates if c in self.vocab}
        if not candidates:
            return word, False

        original_score = self._phrase_log_prob(prev_word, word, next_word)

        best_word, best_score = word, original_score
        for cand in candidates:
            score = self._phrase_log_prob(prev_word, cand, next_word)
            if score > best_score:
                best_word, best_score = cand, score

        changed = best_word != word and (best_score - original_score) > log_margin
        return (best_word, True) if changed else (word, False)

    # Whole-sentence wrapper (used by CLI and evaluation)
    def correct_sentence(self, tokens):
        """
        tokens: list[str] (already tokenized, case as typed).
        Returns list of (original_token, corrected_token, was_changed).
        Non-alphabetic tokens (punctuation) are passed through unchanged.
        """
        results = []
        lowered = [t.lower() for t in tokens]

        for i, tok in enumerate(tokens):
            low = lowered[i]
            if not low.isalpha():
                results.append((tok, tok, False))
                continue

            if low not in self.vocab:
                corrected, _ = self.correct_nonword(low)
                changed = corrected != low
            else:
                prev_w = lowered[i - 1] if i > 0 and lowered[i - 1].isalpha() else None
                next_w = lowered[i + 1] if i + 1 < len(lowered) and lowered[i + 1].isalpha() else None
                corrected, changed = self.correct_realword(prev_w, low, next_w)

            if not changed:
                corrected = tok
            elif tok[0].isupper():
                corrected = corrected.capitalize()

            results.append((tok, corrected, changed))

        return results
