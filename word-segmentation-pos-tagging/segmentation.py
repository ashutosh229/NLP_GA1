"""
segmentation.py
---------------
Word Segmentation Engine:
1. Trigram Language Model with Linear Interpolation Smoothing, Character N-gram LM for OOV words, and Affix Modeling
2. Dynamic Programming (Viterbi) Word Segmenter with Beam Search
3. Baseline: Greedy Longest-Match Segmenter
"""

import math
from functools import lru_cache
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set, Optional


BOS = "<s>"
EOS = "</s>"


class TrigramLanguageModel:
    """
    Trigram word-level language model with linear interpolation smoothing:
    P(w3 | w1, w2) = l3 * P_ML(w3 | w1, w2) + l2 * P_ML(w3 | w2) + l1 * P_ML(w3) + l0 * P_OOV(w3)
    """

    def __init__(self, l3: float = 0.70, l2: float = 0.20, l1: float = 0.09, l0: float = 0.01):
        self.l3 = l3
        self.l2 = l2
        self.l1 = l1
        self.l0 = l0

        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.context2_counts = Counter()  # Count of (w1, w2)
        self.context1_counts = Counter()  # Count of (w1)
        self.vocab: Set[str] = set()
        self.total_tokens: int = 0
        self.max_word_len: int = 25
        self.unigram_probs: Dict[str, float] = {}

        # Character-level n-gram statistics for principled OOV scoring
        self.char_trigrams = Counter()
        self.char_bigrams = Counter()
        self.char_unigrams = Counter()
        self.total_chars = 0
        self._oov_cache: Dict[str, float] = {}

        # Common productive prefixes and suffixes
        self.common_affixes = {
            'prefixes': ('des', 'in', 'im', 're', 'pre', 'sub', 'anti', 'un', 'dis', 'mis', 'over', 'under'),
            'suffixes': ('ado', 'ada', 'ados', 'adas', 'ido', 'ida', 'idos', 'idas', 'mente', 'ción', 'cion',
                         'sión', 'sion', 'dad', 'edad', 'idad', 'oso', 'osa', 'osos', 'osas', 'able', 'ible',
                         'ando', 'iendo', 'aron', 'ieron', 'aba', 'abas', 'abais', 'aban',
                         'ing', 'ed', 'tion', 'sion', 'ment', 'ness', 'ful', 'less', 'ly', 'ies', 'est', 'er')
        }

    def train(self, sentences: List[List[str]]):
        """Train language model on a list of word lists."""
        for sent in sentences:
            if not sent:
                continue
            padded_sent = [BOS, BOS] + sent + [EOS]
            for w in sent:
                self.unigram_counts[w] += 1
                self.vocab.add(w)
                if len(w) > self.max_word_len:
                    self.max_word_len = min(len(w), 30)

                # Character statistics
                pw = '^' + w + '$'
                for c in pw:
                    self.char_unigrams[c] += 1
                    self.total_chars += 1
                for c1, c2 in zip(pw, pw[1:]):
                    self.char_bigrams[(c1, c2)] += 1
                for c1, c2, c3 in zip(pw, pw[1:], pw[2:]):
                    self.char_trigrams[(c1, c2, c3)] += 1

            for w1, w2 in zip(padded_sent, padded_sent[1:]):
                self.bigram_counts[(w1, w2)] += 1
                self.context1_counts[w1] += 1

            for w1, w2, w3 in zip(padded_sent, padded_sent[1:], padded_sent[2:]):
                self.trigram_counts[(w1, w2, w3)] += 1
                self.context2_counts[(w1, w2)] += 1

        self.total_tokens = sum(self.unigram_counts.values()) or 1
        self.unigram_probs = {w: c / self.total_tokens for w, c in self.unigram_counts.items()}
        self._oov_cache.clear()

    def char_lm_log_prob(self, word: str) -> float:
        """Score word character sequence under character trigram LM."""
        pw = '^' + word + '$'
        num_chars = len(self.char_unigrams) + 1
        log_prob = 0.0
        for c1, c2, c3 in zip(pw, pw[1:], pw[2:]):
            c_count = self.char_trigrams.get((c1, c2, c3), 0)
            b_count = self.char_bigrams.get((c1, c2), 0)
            p = (c_count + 0.05) / (b_count + 0.05 * num_chars) if b_count > 0 else 0.05 / (0.05 * num_chars)
            log_prob += math.log2(p)
        return log_prob

    def oov_prob(self, word: str) -> float:
        """Penalty probability for unseen / OOV words with internal cache."""
        if word in self._oov_cache:
            return self._oov_cache[word]

        char_score = self.char_lm_log_prob(word)
        norm_score = char_score / (len(word) + 1)

        # Affix bonus
        affix_bonus = 0.0
        for pref in self.common_affixes['prefixes']:
            if word.startswith(pref) and len(word) > len(pref) + 2:
                affix_bonus += 1.0
                break
        for suff in self.common_affixes['suffixes']:
            if word.endswith(suff) and len(word) > len(suff) + 2:
                affix_bonus += 1.5
                break

        scaled_log_p = max(-25.0, (norm_score * 3.5) + affix_bonus - 12.0)
        prob = 2.0 ** scaled_log_p
        self._oov_cache[word] = prob
        return prob

    def word_prob(self, w1: str, w2: str, w3: str) -> float:
        """Compute smoothed P(w3 | w1, w2)."""
        p3 = 0.0
        c2 = self.context2_counts.get((w1, w2), 0)
        if c2 > 0:
            p3 = self.trigram_counts.get((w1, w2, w3), 0) / c2

        p2 = 0.0
        c1 = self.context1_counts.get(w2, 0)
        if c1 > 0:
            p2 = self.bigram_counts.get((w2, w3), 0) / c1

        p1 = self.unigram_probs.get(w3, 0.0)
        p0 = self.oov_prob(w3) if w3 not in self.vocab else 1e-9

        prob = (self.l3 * p3) + (self.l2 * p2) + (self.l1 * p1) + (self.l0 * p0)
        return max(prob, 1e-15)

    def log_prob(self, w1: str, w2: str, w3: str) -> float:
        """Return log2 P(w3 | w1, w2)."""
        return math.log2(self.word_prob(w1, w2, w3))


class TrigramDPSegmenter:
    """
    Dynamic programming word segmentation using Viterbi search over Trigram Language Model.
    """

    def __init__(self, lm: TrigramLanguageModel, max_word_len: int = 25, beam_size: int = 40):
        self.lm = lm
        self.max_word_len = max_word_len
        self.beam_size = beam_size

    def segment(self, text: str) -> List[str]:
        """
        Segment an unspaced string into words using dynamic programming.
        """
        text = text.strip().lower()
        n = len(text)
        if n == 0:
            return []

        # dp[j] -> dict: (w_prev, w_curr) -> (score, prev_pos, w_prev2)
        dp: List[Dict[Tuple[str, str], Tuple[float, Optional[int], Optional[str]]]] = [{} for _ in range(n + 1)]
        dp[0][(BOS, BOS)] = (0.0, None, None)

        for j in range(1, n + 1):
            min_i = max(0, j - self.max_word_len)
            candidates_at_j = {}

            for i in range(min_i, j):
                if not dp[i]:
                    continue

                candidate_word = text[i:j]
                is_known = candidate_word in self.lm.vocab

                # Allow single characters always; multi-char unknown words only if within reasonable length
                if not is_known and len(candidate_word) > 10:
                    continue

                for (w_prev2, w_prev1), (prev_score, _, _) in dp[i].items():
                    log_p = self.lm.log_prob(w_prev2, w_prev1, candidate_word)
                    
                    # Word boundary penalty: favors cohesive longer real words over gratuitous fragments
                    if not is_known:
                        log_p -= 2.0 * len(candidate_word)

                    total_score = prev_score + log_p
                    state = (w_prev1, candidate_word)

                    if state not in candidates_at_j or total_score > candidates_at_j[state][0]:
                        candidates_at_j[state] = (total_score, i, w_prev2)

            # Beam pruning
            if candidates_at_j:
                if len(candidates_at_j) > self.beam_size:
                    sorted_candidates = sorted(candidates_at_j.items(), key=lambda item: item[1][0], reverse=True)
                    dp[j] = dict(sorted_candidates[:self.beam_size])
                else:
                    dp[j] = candidates_at_j
            else:
                # Fallback: advance by 1 char from best at j-1
                if dp[j - 1]:
                    best_prev_state, (best_score, _, _) = max(dp[j - 1].items(), key=lambda item: item[1][0])
                    char_word = text[j - 1:j]
                    log_p = self.lm.log_prob(best_prev_state[0], best_prev_state[1], char_word) - 6.0
                    dp[j][(best_prev_state[1], char_word)] = (best_score + log_p, j - 1, best_prev_state[0])

        # Find best final state including EOS transition
        best_final_score = -float('inf')
        best_final_state = None

        if not dp[n]:
            return list(text)

        for (w_prev, w_curr), (score, prev_pos, w_prev2) in dp[n].items():
            eos_score = score + self.lm.log_prob(w_prev, w_curr, EOS)
            if eos_score > best_final_score:
                best_final_score = eos_score
                best_final_state = (w_prev, w_curr)

        if best_final_state is None:
            best_final_state = max(dp[n].keys(), key=lambda st: dp[n][st][0])

        # Backtrack
        words = []
        curr_pos = n
        curr_state = best_final_state

        while curr_pos > 0 and curr_state is not None:
            w_prev, w_curr = curr_state
            if curr_state not in dp[curr_pos]:
                break
            score, prev_pos, w_prev2 = dp[curr_pos][curr_state]
            words.append(w_curr)
            curr_pos = prev_pos
            if prev_pos == 0:
                break
            curr_state = (w_prev2, w_prev)

        words.reverse()
        return words


class GreedyLongestMatchSegmenter:
    """
    Baseline Word Segmenter: Greedy Longest-Match.
    Always matches the longest prefix present in the vocabulary.
    """

    def __init__(self, vocab: Set[str], max_word_len: int = 25):
        self.vocab = set(vocab)
        self.max_word_len = max_word_len

    def segment(self, text: str) -> List[str]:
        """
        Segment an unspaced string greedily.
        """
        text = text.strip().lower()
        n = len(text)
        i = 0
        words = []

        while i < n:
            matched = False
            max_len = min(self.max_word_len, n - i)
            for l in range(max_len, 0, -1):
                sub = text[i:i + l]
                if sub in self.vocab:
                    words.append(sub)
                    i += l
                    matched = True
                    break

            if not matched:
                words.append(text[i:i + 1])
                i += 1

        return words
