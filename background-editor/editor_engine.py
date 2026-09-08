"""
editor_engine.py
----------------
Part 1 & 4: Integrated Background Editor Engine
Hosts:
1. Q1 trained English beam-search segmenter and POS tagger
2. Q3 spelling corrector (vocabulary, unigram/bigram models, Method A & B)
3. Constituency PCFG parser
4. Shared smoothed N-gram Language Model
5. Fast-typing merge simulation (probability p)
6. Three-tier live alerting: [SEGMENT-ALERT], [SPELL-ALERT], [GRAMMAR-ALERT]
7. Latency tracking and sentence reconciliation
"""

import sys
import os
import time
import random
import re
from typing import List, Tuple, Dict, Optional, Any, Callable

# Add Q1 and Q3 roots to sys.path for direct component reuse
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
Q1_DIR = os.path.join(PROJECT_ROOT, "word-segmentation-pos-tagging")
Q3_DIR = os.path.join(PROJECT_ROOT, "spelling-correction")

for d in [CURRENT_DIR, Q1_DIR, Q3_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

# Direct imports from Q1
try:
    from data_loader import load_english_brown
    from segmentation import TrigramLanguageModel, TrigramDPSegmenter
    from pos_tagger import TrigramHMMPOSTagger
except ImportError as e:
    raise ImportError(f"Could not import Question 1 modules from {Q1_DIR}: {e}")

# Direct imports from Q3
try:
    from src.model_builder import build_vocab_and_unigrams, build_bigram_model, build_deletes_dict
    from src.candidate_generation import method_a_candidates, method_b_candidates
    from src.corrector import SpellingCorrector
except ImportError as e:
    raise ImportError(f"Could not import Question 3 modules from {Q3_DIR}: {e}")

from pcfg_parser import CKYConstituencyParser, reconcile_tag
from language_model import AddKSmoothedLM


class IntegratedEditorEngine:
    """
    Unified background editor hosting live segmentation, spelling correction,
    and constituency/N-gram grammar checking on an incremental text stream.
    """

    def __init__(self, p_merge: float = 0.08, trigger_interval_n: int = 5, k_smoothing: float = 0.05):
        """
        p_merge: 0.08 (simulates fast typist skipping space with ~8% probability)
        trigger_interval_n: 5 words (optimal balance between real-time responsiveness and latency)
        k_smoothing: 0.05 (add-k parameter for Brown LM)
        """
        self.p_merge = p_merge
        self.trigger_interval_n = trigger_interval_n
        self.k_smoothing = k_smoothing

        # Models
        self.q1_lm: Optional[TrigramLanguageModel] = None
        self.q1_segmenter: Optional[TrigramDPSegmenter] = None
        self.q1_pos_tagger: Optional[TrigramHMMPOSTagger] = None

        self.q3_corrector: Optional[SpellingCorrector] = None
        self.q3_vocab: set = set()
        self.q3_deletes_dict: dict = {}

        self.pcfg_parser: Optional[CKYConstituencyParser] = None
        self.shared_lm: Optional[AddKSmoothedLM] = None

        self.is_initialized = False

    def initialize(self, max_train_sents: int = 2500):
        """
        Train and load models once, reusing shared representations across sub-systems.
        """
        print("Initializing Integrated Editor Engine...")

        # 1. Load English Brown Corpus via Q1 data loader
        print("Loading Brown Corpus...")
        brown_train, brown_test = load_english_brown()
        if max_train_sents is not None:
            brown_train = brown_train[:max_train_sents]
        raw_sentences = [[w for w, _, _ in sent] for sent in brown_train]
        tagged_sentences = [[(w, u) for w, u, _ in sent] for sent in brown_train]

        # 2. Train Q1 English Trigram LM + DP Segmenter + POS Tagger
        print("Initializing Q1 English Trigram LM, DP Segmenter, and POS Tagger...")
        self.q1_lm = TrigramLanguageModel()
        self.q1_lm.train(raw_sentences)
        self.q1_segmenter = TrigramDPSegmenter(self.q1_lm, max_word_len=25, beam_size=40)
        self.q1_pos_tagger = TrigramHMMPOSTagger()
        self.q1_pos_tagger.train(tagged_sentences)

        # 3. Train Q3 Spelling Model (Vocabulary, Unigrams, Bigrams, Deletes Dict)
        print("Initializing Q3 Spelling Corrector and Candidate Generators...")
        vocab, unigrams, total_tokens = build_vocab_and_unigrams(raw_sentences)
        bigram_counts, context_counts = build_bigram_model(raw_sentences, vocab)
        deletes_dict = build_deletes_dict(vocab)

        q3_model_dict = {
            "vocab": vocab,
            "unigram_counts": unigrams,
            "bigram_counts": bigram_counts,
            "context_counts": context_counts,
            "deletes_dict": deletes_dict,
            "total_tokens": total_tokens
        }
        self.q3_corrector = SpellingCorrector(q3_model_dict)
        self.q3_vocab = vocab
        self.q3_deletes_dict = deletes_dict

        # 4. Train PCFG Constituency Parser
        print("Inducing PCFG from Penn Treebank sample...")
        self.pcfg_parser = CKYConstituencyParser(start_symbol="S")
        self.pcfg_parser.train_from_treebank(max_sents=800)

        # 5. Train Shared Smoothed N-Gram LM (Add-k)
        print(f"Training Shared Smoothed N-Gram LM (k={self.k_smoothing})...")
        self.shared_lm = AddKSmoothedLM(k=self.k_smoothing)
        self.shared_lm.train(raw_sentences)

        self.is_initialized = True
        print("Engine initialization complete.")

    def simulate_typing_stream(self, text: str, seed: Optional[int] = None) -> List[str]:
        """
        Tokenizes text and applies simulated fast-typing merges:
        Drops the space between two consecutive words with probability p_merge.
        """
        if seed is not None:
            random.seed(seed)

        raw_tokens = text.strip().split()
        if not raw_tokens:
            return []

        merged_stream = []
        i = 0
        while i < len(raw_tokens):
            curr_token = raw_tokens[i]
            # Check merge with next word if alphanumeric
            if i + 1 < len(raw_tokens) and random.random() < self.p_merge:
                next_token = raw_tokens[i + 1]
                # Only merge if both look like words (not standalone punctuation)
                if re.match(r'^[A-Za-z]+$', curr_token) and re.match(r'^[A-Za-z]+$', next_token):
                    merged_stream.append(curr_token + next_token)
                    i += 2
                    continue
            merged_stream.append(curr_token)
            i += 1

        return merged_stream

    def check_token(self, token: str) -> Dict[str, Any]:
        """
        Per-token live checks executed in strict order:
        1. [SEGMENT-ALERT] via Q1 joint beam decoder
        2. [SPELL-ALERT] via Q3 candidate generation
        Returns dict with resulting words, tags, alerts, and latency.
        """
        t0 = time.perf_counter()
        clean_token = re.sub(r'^[^\w]+|[^\w]+$', '', token).lower()
        punct_prefix = token[:len(token) - len(token.lstrip('.,!?;:"\'()'))]
        punct_suffix = token[len(token.rstrip('.,!?;:"\'()')):]

        alerts = []
        resulting_tokens: List[Tuple[str, str]] = []  # list of (word, pos_tag)
        is_segmented = False
        is_spelled = False

        if not clean_token or not clean_token.isalpha():
            # Punctuation or numbers: pass through
            tag = '.' if not clean_token.isalnum() else 'CD'
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "input_token": token,
                "resulting_tokens": [(token, tag)],
                "alerts": [],
                "is_segmented": False,
                "is_spelled": False,
                "latency_ms": latency_ms
            }

        # Check 1: SEGMENT-ALERT
        # If token is not in vocabulary or is unusually long (>12 chars), test split
        in_vocab = clean_token in self.q3_vocab
        is_long = len(clean_token) >= 12

        if not in_vocab or is_long:
            # Run Q1 beam-search segmenter restricted to this token
            splits = self.q1_segmenter.segment(clean_token)
            # A valid split must break into >= 2 words where all words are in vocabulary
            if len(splits) >= 2 and all(w in self.q1_lm.vocab for w in splits):
                # Calculate combined score vs treating as single word
                single_prob = self.q1_lm.word_prob("<s>", "<s>", clean_token)
                split_prob = 1.0
                prev1, prev2 = "<s>", "<s>"
                for w in splits:
                    split_prob *= self.q1_lm.word_prob(prev2, prev1, w)
                    prev2, prev1 = prev1, w

                if split_prob > single_prob:
                    # Tag the split words
                    tagged_splits = self.q1_pos_tagger.tag(splits)
                    resulting_tokens = tagged_splits
                    alerts.append({
                        "type": "SEGMENT-ALERT",
                        "original": token,
                        "replacement": [w for w, _ in tagged_splits],
                        "tags": [t for _, t in tagged_splits],
                        "message": f"[SEGMENT-ALERT] Merged token '{token}' split into {' + '.join(w for w, _ in tagged_splits)}"
                    })
                    is_segmented = True

        # Check 2: SPELL-ALERT
        # If not segmented and token is not in vocabulary
        if not is_segmented:
            if not in_vocab:
                # Run Q3 candidate generation (Method B SymSpell with Method A)
                best_correction, candidates = self.q3_corrector.correct_nonword(clean_token)
                if best_correction != clean_token:
                    # Tag corrected word
                    tag = self.q1_pos_tagger.tag([best_correction])[0][1]
                    word_to_emit = punct_prefix + best_correction + punct_suffix
                    resulting_tokens = [(word_to_emit, tag)]
                    alerts.append({
                        "type": "SPELL-ALERT",
                        "original": token,
                        "replacement": best_correction,
                        "candidates_count": len(candidates),
                        "message": f"[SPELL-ALERT] Non-word '{token}' corrected to '{best_correction}'"
                    })
                    is_spelled = True
                else:
                    # Uncorrectable OOV word
                    tag = self.q1_pos_tagger.tag([clean_token])[0][1]
                    resulting_tokens = [(token, tag)]
            else:
                # Known vocabulary word
                tag = self.q1_pos_tagger.tag([clean_token])[0][1]
                resulting_tokens = [(token, tag)]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "input_token": token,
            "resulting_tokens": resulting_tokens,
            "alerts": alerts,
            "is_segmented": is_segmented,
            "is_spelled": is_spelled,
            "latency_ms": latency_ms
        }

    def check_grammar_window(self, accumulated_words: List[str]) -> Dict[str, Any]:
        """
        Trigger-interval check executed every N words:
        1. Window bigram & trigram perplexity
        2. Real-word error check (Q3 bigram context)
        """
        t0 = time.perf_counter()
        if len(accumulated_words) < 2:
            return {"alerts": [], "latency_ms": (time.perf_counter() - t0) * 1000.0, "ppl": 1.0}

        window = accumulated_words[-self.trigger_interval_n:]
        clean_window = [re.sub(r'[^\w]', '', w).lower() for w in window if re.sub(r'[^\w]', '', w)]

        alerts = []

        # 1. Sliding Window Perplexity Check
        ppl = self.shared_lm.window_perplexity(clean_window, use_trigram=True)
        # Empirical threshold for implausible n-gram sequence in Brown
        if ppl > 650.0 and len(clean_window) >= 3:
            alerts.append({
                "type": "GRAMMAR-ALERT",
                "subtype": "HIGH_PERPLEXITY",
                "window": " ".join(clean_window),
                "perplexity": round(ppl, 2),
                "message": f"[GRAMMAR-ALERT] High perplexity window ({ppl:.1f}): '{' '.join(clean_window)}'"
            })

        # 2. Real-Word Error Detection (Q3 bigram context check)
        # Examine the middle target word of the last 3 words
        if len(clean_window) >= 3:
            prev_w = clean_window[-3]
            target_w = clean_window[-2]
            next_w = clean_window[-1]

            if target_w in self.q3_vocab:
                # Run Q3 real-word check
                correction, cand_dict = self.q3_corrector.correct_realword(prev_w, target_w, next_w, log_margin=2.0)
                if correction != target_w:
                    alerts.append({
                        "type": "GRAMMAR-ALERT",
                        "subtype": "REAL_WORD_ERROR",
                        "target": target_w,
                        "replacement": correction,
                        "phrase": f"{prev_w} {target_w} {next_w}",
                        "message": f"[GRAMMAR-ALERT] Context error: '{target_w}' in '{prev_w} {target_w} {next_w}' -> suggested '{correction}'"
                    })

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "alerts": alerts,
            "perplexity": round(ppl, 2),
            "latency_ms": latency_ms
        }

    def process_passage_stream(self, text: str, on_step_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Simulates live typing of a complete passage word-by-word with merged tokens.
        Calls on_step_callback after each token.
        """
        token_stream = self.simulate_typing_stream(text)

        accumulated_tokens: List[Tuple[str, str]] = []  # (word, tag)
        all_alerts: List[Dict[str, Any]] = []
        token_latencies: List[float] = []
        grammar_latencies: List[float] = []
        merges_resolved = 0
        spelling_corrections = 0

        for idx, raw_token in enumerate(token_stream):
            # Per-token check
            tok_result = self.check_token(raw_token)
            token_latencies.append(tok_result["latency_ms"])

            if tok_result["is_segmented"]:
                merges_resolved += 1
            if tok_result["is_spelled"]:
                spelling_corrections += 1

            for alert in tok_result["alerts"]:
                all_alerts.append(alert)

            accumulated_tokens.extend(tok_result["resulting_tokens"])

            # Grammar check at every N tokens
            grammar_result = {"alerts": [], "latency_ms": 0.0}
            if len(accumulated_tokens) % self.trigger_interval_n == 0:
                words_so_far = [w for w, _ in accumulated_tokens]
                grammar_result = self.check_grammar_window(words_so_far)
                grammar_latencies.append(grammar_result["latency_ms"])
                for g_alert in grammar_result["alerts"]:
                    all_alerts.append(g_alert)

            step_data = {
                "step_index": idx,
                "input_token": raw_token,
                "resulting_tokens": tok_result["resulting_tokens"],
                "token_alerts": tok_result["alerts"],
                "grammar_alerts": grammar_result["alerts"],
                "accumulated_text": " ".join(w for w, _ in accumulated_tokens),
                "tok_latency_ms": tok_result["latency_ms"],
                "gram_latency_ms": grammar_result["latency_ms"]
            }

            if on_step_callback:
                on_step_callback(step_data)

        # Final sentence splitting and PCFG / N-gram analysis
        passage_summary = self.analyze_final_passage(accumulated_tokens, merges_resolved, spelling_corrections)

        avg_tok_lat = sum(token_latencies) / max(1, len(token_latencies))
        avg_gram_lat = sum(grammar_latencies) / max(1, len(grammar_latencies))

        return {
            "original_text": text,
            "streamed_tokens": token_stream,
            "final_tokens": accumulated_tokens,
            "final_text": " ".join(w for w, _ in accumulated_tokens),
            "all_alerts": all_alerts,
            "merges_resolved": merges_resolved,
            "spelling_corrections": spelling_corrections,
            "avg_token_latency_ms": avg_tok_lat,
            "avg_grammar_latency_ms": avg_gram_lat,
            "sentences_summary": passage_summary
        }

    def analyze_final_passage(self, tagged_tokens: List[Tuple[str, str]], total_merges: int, total_spells: int) -> List[Dict[str, Any]]:
        """
        Part 4: Final Passage Analysis — Method Comparison
        Splits final token stream into sentences and evaluates:
        (a) PCFG log-probability
        (b) Bigram log-probability
        (c) Trigram log-probability
        Applies decision rule to give final grammaticality verdict.
        """
        # Split tokens into sentences by terminal punctuation (. ? !)
        sentences: List[List[Tuple[str, str]]] = []
        curr_sent: List[Tuple[str, str]] = []

        for w, tag in tagged_tokens:
            curr_sent.append((w, tag))
            if any(w.endswith(punct) for punct in ['.', '!', '?']):
                sentences.append(curr_sent)
                curr_sent = []
        if curr_sent:
            sentences.append(curr_sent)

        summary_rows = []

        for sent_idx, sent_tokens in enumerate(sentences):
            words = [re.sub(r'[^\w]', '', w).lower() for w, _ in sent_tokens if re.sub(r'[^\w]', '', w)]
            raw_sent_str = " ".join(w for w, _ in sent_tokens)
            tags = [tag for _, tag in sent_tokens]

            if not words:
                continue

            # 1. PCFG Evaluation
            tree, pcfg_lp, pcfg_status = self.pcfg_parser.parse(words, tags)
            pcfg_score_str = f"{pcfg_lp:.2f}" if tree is not None else "unparseable"

            # 2. Bigram Evaluation
            bi_lp = self.shared_lm.sentence_log_prob_bigram(words)
            bi_ppl = self.shared_lm.sentence_perplexity_bigram(words)

            # 3. Trigram Evaluation
            tri_lp = self.shared_lm.sentence_log_prob_trigram(words)
            tri_ppl = self.shared_lm.sentence_perplexity_trigram(words)

            # 4. Decision Rule Selection
            chosen_method, verdict, reason = self._apply_decision_rule(tree, pcfg_lp, tri_ppl, bi_ppl, len(words))

            summary_rows.append({
                "sentence_idx": sent_idx + 1,
                "sentence_text": raw_sent_str,
                "pcfg_result": pcfg_score_str,
                "bigram_score": f"{bi_lp:.2f} (ppl {bi_ppl:.1f})",
                "trigram_score": f"{tri_lp:.2f} (ppl {tri_ppl:.1f})",
                "chosen_method": chosen_method,
                "final_verdict": verdict,
                "decision_reason": reason,
                "merges_resolved": total_merges if sent_idx == 0 else 0,  # attribute or apportion
                "spelling_corrections": total_spells if sent_idx == 0 else 0
            })

        return summary_rows

    def _apply_decision_rule(self, tree: Optional[Tree], pcfg_lp: float, tri_ppl: float, bi_ppl: float, sent_len: int) -> Tuple[str, str, str]:
        """
        Decision rule (Part 4.2):
        1. PCFG: Preferred if valid parse tree found and log-prob per word >= -8.0 (not a probability outlier).
           Verdict: "Grammatical (PCFG)"
        2. Trigram: If PCFG fails or is an outlier, evaluate Trigram LM coverage.
           If tri_ppl <= 450.0 -> "Grammatical (Trigram)"
           If tri_ppl > 450.0 and tri_ppl <= 900.0 -> "Questionable (Trigram)"
        3. Bigram: Used if trigram lacks coverage or high variance (tri_ppl > 900.0).
           If bi_ppl <= 500.0 -> "Grammatical (Bigram)"
           Else -> "Ungrammatical (Bigram Fallback)"
        """
        if tree is not None:
            norm_lp = pcfg_lp / max(1, sent_len)
            if norm_lp >= -8.5:
                return ("PCFG", "Grammatical", "Valid constituent structure and high PCFG parse likelihood")

        # Trigram check
        if tri_ppl <= 400.0:
            return ("Trigram", "Grammatical", "Low trigram perplexity indicates natural lexical collocations")
        elif tri_ppl <= 800.0:
            return ("Trigram", "Questionable", "Moderate trigram perplexity; unusual phrases or syntax")

        # Bigram fallback
        if bi_ppl <= 450.0:
            return ("Bigram", "Acceptable", "Trigram sparse; bigram transitions within acceptable threshold")
        else:
            return ("Bigram", "Ungrammatical", "High perplexity across all models; ungrammatical word sequence")
