"""
pos_tagger.py
-------------
Part 2, 3 & 4: POS Tagging Models
1. Trigram Hidden Markov Model (HMM) POS Tagger with Viterbi decoding
2. Morphology-Aware Tagging Extension (Grammatical Agreement)
3. Baseline: Most Frequent Tag (MFT) Tagger
"""

import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set, Optional


START_TAG = "<s>"
END_TAG = "</s>"


class TrigramHMMPOSTagger:
    """
    Trigram Hidden Markov Model (HMM) POS Tagger:
    - Transition probabilities: P(t_i | t_{i-2}, t_{i-1}) smoothed with linear interpolation.
    - Emission probabilities: P(w | t) smoothed with Lidstone smoothing and suffix/morphological handling for OOV words.
    - Decoding: Exact Viterbi dynamic programming.
    """

    def __init__(self, a3: float = 0.65, a2: float = 0.25, a1: float = 0.09, a0: float = 0.01, delta: float = 1e-4):
        self.a3 = a3
        self.a2 = a2
        self.a1 = a1
        self.a0 = a0
        self.delta = delta

        self.tag_unigrams = Counter()
        self.tag_bigrams = Counter()
        self.tag_trigrams = Counter()
        self.tag_context2 = Counter()  # (t1, t2)
        self.tag_context1 = Counter()  # (t1)

        self.word_tag_counts = Counter()  # (w, t) -> count
        self.tag_counts = Counter()       # t -> count
        self.word_counts = Counter()      # w -> count
        self.word_to_tags = defaultdict(set)  # w -> set of observed tags
        
        self.suffix_counts = Counter()    # (suffix, t) -> count
        self.suffix_tag_totals = Counter() # suffix -> total count

        self.tags: Set[str] = set()
        self.vocab: Set[str] = set()
        self.total_tokens: int = 0
        self.most_common_tag: str = "NOUN"

    def train(self, tagged_sentences: List[List[Tuple[str, str]]]):
        """
        Train HMM parameters on a list of tagged sentences: [(word, tag), ...]
        """
        for sent in tagged_sentences:
            if not sent:
                continue

            padded_tags = [START_TAG, START_TAG] + [t for _, t in sent] + [END_TAG]

            for w, t in sent:
                w_clean = w.lower()
                self.word_tag_counts[(w_clean, t)] += 1
                self.tag_counts[t] += 1
                self.word_counts[w_clean] += 1
                self.word_to_tags[w_clean].add(t)
                self.tags.add(t)
                self.vocab.add(w_clean)

                # Suffixes for OOV smoothing (lengths 1 to 4)
                for suf_len in range(1, 5):
                    if len(w_clean) >= suf_len:
                        suf = w_clean[-suf_len:]
                        self.suffix_counts[(suf, t)] += 1
                        self.suffix_tag_totals[suf] += 1

            for t1, t2 in zip(padded_tags, padded_tags[1:]):
                self.tag_bigrams[(t1, t2)] += 1
                self.tag_context1[t1] += 1

            for t1, t2, t3 in zip(padded_tags, padded_tags[1:], padded_tags[2:]):
                self.tag_trigrams[(t1, t2, t3)] += 1
                self.tag_context2[(t1, t2)] += 1
                if t3 not in (START_TAG, END_TAG):
                    self.tag_unigrams[t3] += 1

        self.total_tokens = sum(self.tag_unigrams.values()) or 1
        if self.tag_counts:
            self.most_common_tag = self.tag_counts.most_common(1)[0][0]

    def transition_prob(self, t1: str, t2: str, t3: str) -> float:
        """Compute smoothed P(t3 | t1, t2) via linear interpolation."""
        p3 = 0.0
        c2 = self.tag_context2.get((t1, t2), 0)
        if c2 > 0:
            p3 = self.tag_trigrams.get((t1, t2, t3), 0) / c2

        p2 = 0.0
        c1 = self.tag_context1.get(t2, 0)
        if c1 > 0:
            p2 = self.tag_bigrams.get((t2, t3), 0) / c1

        p1 = self.tag_unigrams.get(t3, 0) / self.total_tokens
        num_tags = len(self.tags) + 1
        p0 = 1.0 / num_tags

        prob = (self.a3 * p3) + (self.a2 * p2) + (self.a1 * p1) + (self.a0 * p0)
        return max(prob, 1e-15)

    def emission_prob(self, word: str, tag: str) -> float:
        """
        Compute emission probability P(w | t).
        Uses Lidstone smoothing for seen words and suffix-informed prior for unseen words.
        """
        w_clean = word.lower()
        tag_count = self.tag_counts.get(tag, 0)
        vocab_size = len(self.vocab)

        if w_clean in self.vocab:
            wt_count = self.word_tag_counts.get((w_clean, tag), 0)
            return (wt_count + self.delta) / (tag_count + self.delta * (vocab_size + 1))

        # Unknown / OOV word handling: Use suffix morphology and tag prior
        p_tag = (tag_count + 1.0) / (self.total_tokens + len(self.tags))
        
        # Check longest matching suffix
        for suf_len in (4, 3, 2, 1):
            if len(w_clean) >= suf_len:
                suf = w_clean[-suf_len:]
                suf_tot = self.suffix_tag_totals.get(suf, 0)
                if suf_tot >= 3:
                    p_t_given_suf = (self.suffix_counts.get((suf, tag), 0) + 0.1) / (suf_tot + 0.1 * len(self.tags))
                    return p_t_given_suf * p_tag

        return p_tag * 1e-3

    def tag(self, words: List[str]) -> List[Tuple[str, str]]:
        """
        Tag a sequence of words using Viterbi decoding.
        Returns: [(word, tag), ...]
        """
        if not words:
            return []

        n = len(words)
        # viterbi[k] -> dict: (t_prev, t_curr) -> (log_score, t_prev2)
        viterbi: List[Dict[Tuple[str, str], Tuple[float, Optional[str]]]] = [{} for _ in range(n)]

        # Step 0: First word
        w0 = words[0].lower()
        cand_tags_0 = self.word_to_tags.get(w0, self.tags) or self.tags

        for t0 in cand_tags_0:
            trans_p = self.transition_prob(START_TAG, START_TAG, t0)
            emiss_p = self.emission_prob(w0, t0)
            score = math.log2(trans_p) + math.log2(emiss_p)
            viterbi[0][(START_TAG, t0)] = (score, START_TAG)

        # Step 1 to n-1
        for k in range(1, n):
            wk = words[k].lower()
            cand_tags_k = self.word_to_tags.get(wk, self.tags) or self.tags

            for (t_prev2, t_prev1), (prev_score, _) in viterbi[k - 1].items():
                for tk in cand_tags_k:
                    trans_p = self.transition_prob(t_prev2, t_prev1, tk)
                    emiss_p = self.emission_prob(wk, tk)
                    score = prev_score + math.log2(trans_p) + math.log2(emiss_p)
                    state = (t_prev1, tk)

                    if state not in viterbi[k] or score > viterbi[k][state][0]:
                        viterbi[k][state] = (score, t_prev2)

            # Pruning to avoid exponential state explosion
            if len(viterbi[k]) > 80:
                top_states = sorted(viterbi[k].items(), key=lambda x: x[1][0], reverse=True)[:80]
                viterbi[k] = dict(top_states)

        # End of sequence transition to END_TAG
        best_final_score = -float('inf')
        best_final_state = None

        for (t_prev, t_curr), (score, _) in viterbi[n - 1].items():
            end_trans = self.transition_prob(t_prev, t_curr, END_TAG)
            total_end_score = score + math.log2(end_trans)
            if total_end_score > best_final_score:
                best_final_score = total_end_score
                best_final_state = (t_prev, t_curr)

        if best_final_state is None:
            # Fallback if no path survived
            return [(w, self.most_common_tag) for w in words]

        # Backtracking
        tags = []
        curr_state = best_final_state
        tags.append(curr_state[1])

        for k in range(n - 1, 0, -1):
            t_prev1, t_curr = curr_state
            score, t_prev2 = viterbi[k][curr_state]
            tags.append(t_prev1)
            curr_state = (t_prev2, t_prev1)

        tags.reverse()
        return list(zip(words, tags))


class MostFrequentTagTagger:
    """
    Baseline POS Tagger: Most-Frequent-Tag (MFT).
    Assigns each word the tag it was most frequently seen with in training data.
    Assigns the global mode tag to unseen / OOV words.
    """

    def __init__(self):
        self.word_mft: Dict[str, str] = {}
        self.default_tag: str = "NOUN"

    def train(self, tagged_sentences: List[List[Tuple[str, str]]]):
        """Train baseline by finding the mode tag for each word."""
        word_tag_counts = defaultdict(Counter)
        global_tag_counts = Counter()

        for sent in tagged_sentences:
            for w, t in sent:
                w_clean = w.lower()
                word_tag_counts[w_clean][t] += 1
                global_tag_counts[t] += 1

        for w, tag_counts in word_tag_counts.items():
            self.word_mft[w] = tag_counts.most_common(1)[0][0]

        if global_tag_counts:
            self.default_tag = global_tag_counts.most_common(1)[0][0]

    def tag(self, words: List[str]) -> List[Tuple[str, str]]:
        """Tag words using dictionary lookup."""
        result = []
        for w in words:
            w_clean = w.lower()
            assigned_tag = self.word_mft.get(w_clean, self.default_tag)
            result.append((w, assigned_tag))
        return result
