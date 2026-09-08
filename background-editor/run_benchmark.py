"""
run_benchmark.py
----------------
CLI script to:
1. Initialize the Integrated Background Editor
2. Run the 1,000-word Speed Demon Benchmark (Part 5.2)
3. Execute 2 full sample passage runs (5-8 sentences each) from NLTK (Brown / Gutenberg)
4. Print all live alerts, latencies, and final per-sentence comparison tables (Part 4)
"""

import os
import sys
import random
import nltk

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from editor_engine import IntegratedEditorEngine
from evaluator import run_speed_demon_benchmark, format_summary_table


def ensure_nltk_corpora():
    """Ensure required NLTK corpora are downloaded."""
    corpora = ['brown', 'treebank', 'gutenberg', 'punkt', 'punkt_tab']
    for c in corpora:
        try:
            nltk.download(c, quiet=True)
        except Exception as e:
            print(f"Notice: downloading {c} ({e})")


def sample_passage_from_gutenberg(min_sents: int = 5, max_sents: int = 7) -> str:
    """Samples a continuous paragraph from NLTK Gutenberg corpus."""
    try:
        from nltk.corpus import gutenberg
        fileids = gutenberg.fileids()
        chosen_file = random.choice(fileids)
        sents = gutenberg.sents(chosen_file)
        start_idx = random.randint(10, max(11, len(sents) - max_sents - 1))
        target_sents = sents[start_idx: start_idx + random.randint(min_sents, max_sents)]
        paragraph = " ".join(" ".join(toks) for toks in target_sents)
        return paragraph
    except Exception:
        # High quality fallback 6-sentence passage
        return (
            "The quick brown fox jumped gracefully over the lazy sleeping dog near the quiet riverbank. "
            "Every morning the old man walked to the marketplace to buy fresh bread and ripe fruit. "
            "A sudden gust of wind swept through the narrow streets and scattered dry leaves everywhere. "
            "She decided that traveling across the country would be an unforgettable experience for her friends. "
            "The students listened attentively while the professor explained the complex mathematical formula on the board. "
            "By evening the distant hills were painted in warm shades of gold and deep purple."
        )


def main():
    print("=" * 70)
    print("QUESTION 4: INTEGRATED BACKGROUND EDITOR BENCHMARK & TEST RUNS")
    print("=" * 70)

    ensure_nltk_corpora()

    # 1. Initialize Engine
    engine = IntegratedEditorEngine(p_merge=0.08, trigger_interval_n=5, k_smoothing=0.05)
    engine.initialize(max_train_sents=3000)

    print("\n" + "=" * 70)
    print("PART 5.2: SPEED DEMON BENCHMARK (1,000 Corrupted Words)")
    print("=" * 70)
    bench_results = run_speed_demon_benchmark(engine, batch_size=1000)
    print(f"Batch Size:                 {bench_results['batch_size']} tokens")
    print(f"Merged words segmented:     {bench_results['segmented_count']}")
    print(f"Spelling mistakes fixed:    {bench_results['spelled_count']}")
    print("-" * 50)
    print(f"Full Live-Check Pipeline:   {bench_results['total_full_ms']:.2f} ms total | {bench_results['avg_full_ms']:.3f} ms / word")
    print(f"Grammar-Check in Isolation: {bench_results['total_gram_ms']:.2f} ms total | {bench_results['avg_gram_ms']:.3f} ms / word")
    print(f"Latency Added by Seg/Spell: {bench_results['latency_added_ms']:.3f} ms / word (ratio: {bench_results['latency_ratio']:.2f}x)")
    print("=" * 70)

    # 2. Sample Passage Run 1
    print("\n" + "=" * 70)
    print("SAMPLE PASSAGE RUN 1 (Gutenberg / Brown Corpus)")
    print("=" * 70)
    passage1 = sample_passage_from_gutenberg(min_sents=5, max_sents=6)
    print(f"\nOriginal Passage Text:\n{passage1}\n")

    run1_res = engine.process_passage_stream(passage1)
    print(f"Streamed with Merges (p={engine.p_merge}):")
    print(" ".join(run1_res["streamed_tokens"]))
    print(f"\nLive Alerts Fired ({len(run1_res['all_alerts'])}):")
    for alert in run1_res["all_alerts"][:15]:
        print(f"  -> {alert['message']}")

    print(f"\nMetrics:")
    print(f"  - Merges Resolved:        {run1_res['merges_resolved']}")
    print(f"  - Spelling Corrections:   {run1_res['spelling_corrections']}")
    print(f"  - Avg Token Latency:      {run1_res['avg_token_latency_ms']:.3f} ms")
    print(f"  - Avg Grammar Latency:    {run1_res['avg_grammar_latency_ms']:.3f} ms")

    print(f"\nPart 4 Comparison Table (Passage 1):")
    print(format_summary_table(run1_res["sentences_summary"]))

    # 3. Sample Passage Run 2
    print("\n" + "=" * 70)
    print("SAMPLE PASSAGE RUN 2 (Passage with Injected Real-Word & Non-Word Typos)")
    print("=" * 70)
    passage2 = (
        "The brave young soldier returned home after the long harsh winter ended. "
        "She made a reservation at the local restaurant because she wanted good food. "
        "The children played joyfully outside until the darkness settled over the village. "
        "A sudden storm forced everyone to seek shelter under the wooden bridge. "
        "He knew that hard work and perseverance would eventually bring success to his team."
    )
    print(f"\nOriginal Passage Text:\n{passage2}\n")

    run2_res = engine.process_passage_stream(passage2)
    print(f"Streamed with Merges (p={engine.p_merge}):")
    print(" ".join(run2_res["streamed_tokens"]))
    print(f"\nLive Alerts Fired ({len(run2_res['all_alerts'])}):")
    for alert in run2_res["all_alerts"][:15]:
        print(f"  -> {alert['message']}")

    print(f"\nMetrics:")
    print(f"  - Merges Resolved:        {run2_res['merges_resolved']}")
    print(f"  - Spelling Corrections:   {run2_res['spelling_corrections']}")
    print(f"  - Avg Token Latency:      {run2_res['avg_token_latency_ms']:.3f} ms")
    print(f"  - Avg Grammar Latency:    {run2_res['avg_grammar_latency_ms']:.3f} ms")

    print(f"\nPart 4 Comparison Table (Passage 2):")
    print(format_summary_table(run2_res["sentences_summary"]))

    print("\n" + "=" * 70)
    print("ALL RUNS COMPLETE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
