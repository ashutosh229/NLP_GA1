"""
evaluator.py
------------
Part 4 & 5.2: Passage Evaluation & Speed Demon Benchmark
- Generates 1,000 corrupted words (single-edit spelling errors and merged words)
- Runs isolated benchmarking comparing:
  (a) Full per-token live-check pipeline (segmentation + spelling)
  (b) Grammar-trigger check in isolation
- Calculates total and average per-word latencies
- Formats comparison tables and logs benchmark conclusions
"""

import time
import random
import string
from typing import List, Dict, Tuple, Any
from tabulate import tabulate


def generate_speed_demon_batch(vocab: List[str], count: int = 1000, seed: int = 42) -> List[str]:
    """
    Constructs a reproducible batch of exactly 1,000 corrupted simulated words:
    - 40% non-word single-edit spelling errors (deletions, substitutions, insertions)
    - 40% merged-word tokens (two consecutive words without space)
    - 20% clean words (to simulate mixed typist stream)
    """
    random.seed(seed)
    valid_words = [w for w in vocab if len(w) >= 3 and w.isalpha()]
    alphabet = string.ascii_lowercase

    batch = []
    for i in range(count):
        mode = random.random()

        if mode < 0.40:
            # Single-edit spelling corruption (non-word)
            w = random.choice(valid_words)
            edit_type = random.choice(['del', 'ins', 'sub', 'trans'])
            if edit_type == 'del' and len(w) > 2:
                idx = random.randrange(len(w))
                corrupted = w[:idx] + w[idx+1:]
            elif edit_type == 'ins':
                idx = random.randrange(len(w) + 1)
                corrupted = w[:idx] + random.choice(alphabet) + w[idx:]
            elif edit_type == 'sub':
                idx = random.randrange(len(w))
                corrupted = w[:idx] + random.choice(alphabet) + w[idx+1:]
            else:  # trans
                if len(w) >= 2:
                    idx = random.randrange(len(w) - 1)
                    corrupted = w[:idx] + w[idx+1] + w[idx] + w[idx+2:]
                else:
                    corrupted = w + 'x'
            batch.append(corrupted)

        elif mode < 0.80:
            # Merged words (simulated spacebar drop)
            w1 = random.choice(valid_words[:500])
            w2 = random.choice(valid_words[:500])
            batch.append(w1 + w2)

        else:
            # Clean word
            batch.append(random.choice(valid_words))

    return batch[:count]


def run_speed_demon_benchmark(engine, batch_size: int = 1000) -> Dict[str, Any]:
    """
    Executes the Speed Demon Benchmark:
    1. Pass 1,000 words through full per-token pipeline (segmentation + spelling).
    2. Pass 1,000 words through grammar check in isolation.
    3. Record and compare latencies.
    """
    vocab_list = list(engine.q3_vocab)
    batch = generate_speed_demon_batch(vocab_list, count=batch_size, seed=42)

    # 1. Full Live-Check Pipeline (Segmentation + Spelling)
    t0_full = time.perf_counter()
    full_latencies = []
    segmented_count = 0
    spelled_count = 0

    for word in batch:
        t_w0 = time.perf_counter()
        res = engine.check_token(word)
        full_latencies.append((time.perf_counter() - t_w0) * 1000.0)
        if res["is_segmented"]:
            segmented_count += 1
        if res["is_spelled"]:
            spelled_count += 1

    t1_full = time.perf_counter()
    total_full_ms = (t1_full - t0_full) * 1000.0
    avg_full_ms = total_full_ms / batch_size

    # 2. Grammar Check in Isolation
    # Feeds the batch as a moving window to the grammar checker
    t0_gram = time.perf_counter()
    gram_latencies = []
    window = []

    for word in batch:
        window.append(word)
        t_g0 = time.perf_counter()
        if len(window) >= engine.trigger_interval_n:
            _ = engine.check_grammar_window(window[-engine.trigger_interval_n:])
            window = []
        gram_latencies.append((time.perf_counter() - t_g0) * 1000.0)

    t1_gram = time.perf_counter()
    total_gram_ms = (t1_gram - t0_gram) * 1000.0
    avg_gram_ms = total_gram_ms / batch_size

    latency_added_ms = max(0.0, avg_full_ms - avg_gram_ms)
    ratio = (avg_full_ms / max(1e-4, avg_gram_ms))

    results = {
        "batch_size": batch_size,
        "segmented_count": segmented_count,
        "spelled_count": spelled_count,
        "total_full_ms": total_full_ms,
        "avg_full_ms": avg_full_ms,
        "total_gram_ms": total_gram_ms,
        "avg_gram_ms": avg_gram_ms,
        "latency_added_ms": latency_added_ms,
        "latency_ratio": ratio
    }

    return results


def format_summary_table(summary_rows: List[Dict[str, Any]]) -> str:
    """
    Formats the Part 4 per-sentence comparison table using tabulate.
    """
    headers = [
        "Sent #", "Sentence Text", "PCFG Result", "Bigram Score",
        "Trigram Score", "Method", "Verdict", "Merges", "Spells"
    ]
    table_data = []
    for r in summary_rows:
        text_disp = r["sentence_text"]
        if len(text_disp) > 40:
            text_disp = text_disp[:37] + "..."
        table_data.append([
            r["sentence_idx"],
            text_disp,
            r["pcfg_result"],
            r["bigram_score"],
            r["trigram_score"],
            r["chosen_method"],
            r["final_verdict"],
            r["merges_resolved"],
            r["spelling_corrections"]
        ])
    return tabulate(table_data, headers=headers, tablefmt="grid")
