"""
evaluate.py
-----------
Comprehensive Evaluation & Error-Source Breakdown Suite:
1. Word Segmentation Metrics: Precision, Recall, F1, Exact Sentence Match
2. POS Tagging Accuracy: Gold-segmented vs Full End-to-End Pipeline
3. Confusion Matrix Analysis (Top Confusions Table)
4. Error-Source Breakdown: Segmentation-induced Errors vs. Genuine Tagging Errors
5. Baseline vs Proposed Comparison
"""

import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Optional

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def calculate_spans(words: List[str]) -> List[Tuple[int, int]]:
    """Compute character (start, end) spans for a list of words."""
    spans = []
    curr = 0
    for w in words:
        start = curr
        end = curr + len(w)
        spans.append((start, end))
        curr = end
    return spans


def evaluate_segmentation(gold_sentences: List[List[str]], pred_sentences: List[List[str]]) -> Dict[str, float]:
    """
    Evaluate word segmentation at the span and sentence levels.
    """
    total_gold_spans = 0
    total_pred_spans = 0
    correct_spans = 0
    exact_sentence_matches = 0
    num_sentences = len(gold_sentences)

    for gold_words, pred_words in zip(gold_sentences, pred_sentences):
        gold_spans = set(calculate_spans(gold_words))
        pred_spans = set(calculate_spans(pred_words))

        total_gold_spans += len(gold_spans)
        total_pred_spans += len(pred_spans)
        matched = len(gold_spans & pred_spans)
        correct_spans += matched

        if gold_words == pred_words:
            exact_sentence_matches += 1

    precision = correct_spans / total_pred_spans if total_pred_spans > 0 else 0.0
    recall = correct_spans / total_gold_spans if total_gold_spans > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    sentence_acc = exact_sentence_matches / num_sentences if num_sentences > 0 else 0.0

    return {
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1": f1 * 100.0,
        "sentence_acc": sentence_acc * 100.0,
        "total_gold_words": total_gold_spans,
        "total_pred_words": total_pred_spans,
        "correct_words": correct_spans
    }


def evaluate_pos_tagger_gold(gold_tagged_sents: List[List[Tuple[str, str]]], tagger) -> Dict[str, Any]:
    """
    Evaluate POS tagger accuracy on gold-segmented words.
    """
    total_tokens = 0
    correct_tokens = 0
    confusion_pairs = Counter()  # (gold_tag, pred_tag) -> count
    tag_set = set()

    for sent in gold_tagged_sents:
        words = [w for w, _ in sent]
        gold_tags = [t for _, t in sent]
        pred_tuples = tagger.tag(words)
        pred_tags = [t for _, t in pred_tuples]

        for w, g_tag, p_tag in zip(words, gold_tags, pred_tags):
            total_tokens += 1
            tag_set.add(g_tag)
            tag_set.add(p_tag)
            if g_tag == p_tag:
                correct_tokens += 1
            else:
                confusion_pairs[(g_tag, p_tag)] += 1

    accuracy = (correct_tokens / total_tokens * 100.0) if total_tokens > 0 else 0.0
    return {
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "correct_tokens": correct_tokens,
        "confusion_pairs": confusion_pairs,
        "tags": sorted(list(tag_set))
    }


def evaluate_end_to_end_and_error_breakdown(
    test_sentences: List[List[Tuple[str, str, str]]],
    pred_segmented_words: List[List[str]],
    tagger,
    use_morphology: bool = False
) -> Dict[str, Any]:
    """
    Part 5: End-to-End Evaluation with Error-Source Breakdown.
    Separates all tagging errors into:
    1. Errors caused by Segmentation Mistakes (incorrect token boundaries)
    2. Genuine POS Tagging Errors (correct word boundary, but wrong tag assigned)
    """
    total_gold_tokens = 0
    total_pred_tokens = 0
    correct_spans_and_tags = 0

    seg_induced_errors = 0
    genuine_tagging_errors = 0
    unmatched_gold_tokens = 0

    confusion_matrix = Counter()  # (gold_tag, pred_tag) -> count

    for sent, pred_words in zip(test_sentences, pred_segmented_words):
        gold_words = [w for w, _, _ in sent]
        gold_tags = [m if use_morphology else u for _, u, m in sent]
        gold_spans = calculate_spans(gold_words)
        gold_span_to_tag = {span: tag for span, tag in zip(gold_spans, gold_tags)}
        total_gold_tokens += len(gold_words)

        pred_tuples = tagger.tag(pred_words)
        pred_tags = [t for _, t in pred_tuples]
        pred_spans = calculate_spans(pred_words)
        total_pred_tokens += len(pred_words)

        matched_gold_spans = set()

        for (p_start, p_end), p_tag in zip(pred_spans, pred_tags):
            p_span = (p_start, p_end)
            if p_span in gold_span_to_tag:
                g_tag = gold_span_to_tag[p_span]
                matched_gold_spans.add(p_span)
                if p_tag == g_tag:
                    correct_spans_and_tags += 1
                else:
                    # Case A: Genuine POS Tagging Error
                    genuine_tagging_errors += 1
                    confusion_matrix[(g_tag, p_tag)] += 1
            else:
                # Case B: Error caused by Segmentation Mistake
                seg_induced_errors += 1

        for g_span in gold_spans:
            if g_span not in matched_gold_spans:
                unmatched_gold_tokens += 1

    total_pipeline_errors = seg_induced_errors + genuine_tagging_errors
    seg_error_pct = (seg_induced_errors / total_pipeline_errors * 100.0) if total_pipeline_errors > 0 else 0.0
    genuine_error_pct = (genuine_tagging_errors / total_pipeline_errors * 100.0) if total_pipeline_errors > 0 else 0.0
    pipeline_accuracy = (correct_spans_and_tags / total_gold_tokens * 100.0) if total_gold_tokens > 0 else 0.0

    return {
        "pipeline_accuracy": pipeline_accuracy,
        "total_gold_tokens": total_gold_tokens,
        "total_pred_tokens": total_pred_tokens,
        "correct_tokens": correct_spans_and_tags,
        "total_pipeline_errors": total_pipeline_errors,
        "seg_induced_errors": seg_induced_errors,
        "seg_error_pct": seg_error_pct,
        "genuine_tagging_errors": genuine_tagging_errors,
        "genuine_error_pct": genuine_error_pct,
        "confusion_matrix": confusion_matrix
    }


def format_confusion_table(confusion_pairs: Counter, top_k: int = 10) -> str:
    """Format top confused tag pairs into a markdown table."""
    top_items = confusion_pairs.most_common(top_k)
    headers = ["Actual (Gold) Tag", "Predicted Tag", "Error Count", "Share of Errors (%)"]
    total_confusions = sum(confusion_pairs.values()) or 1

    rows = []
    for (actual, pred), count in top_items:
        share = (count / total_confusions) * 100.0
        rows.append([actual, pred, count, f"{share:.2f}%"])

    if HAS_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="github")
    
    out = "| " + " | ".join(headers) + " |\n"
    out += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for r in rows:
        out += f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |\n"
    return out
