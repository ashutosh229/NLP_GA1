"""
main.py
-------
Main execution script for Question 1: Word Segmentation and POS Tagging.
Performs:
1. Data Loading (English Brown 80/20 & Spanish UD-GSD)
2. Model Training (Trigram LM + DP Segmenter, Trigram HMM POS Tagger, Morphology-aware Tagger, Baselines)
3. Quantitative Evaluation (Segmentation metrics, POS accuracy, Confusion Matrices, Error-Source Breakdown)
4. Sample Test String Inference (PDF test strings + custom sentences)
5. Generation of Comparative Analysis Results
"""

import sys
import os
import time
import argparse
from typing import List, Tuple, Dict, Any

# Ensure current script directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_english_brown, load_spanish_ud, sentence_to_unspaced
from segmentation import TrigramLanguageModel, TrigramDPSegmenter, GreedyLongestMatchSegmenter
from pos_tagger import TrigramHMMPOSTagger, MostFrequentTagTagger
from pipeline import NLPJointPipeline
from evaluate import (
    evaluate_segmentation,
    evaluate_pos_tagger_gold,
    evaluate_end_to_end_and_error_breakdown,
    format_confusion_table
)

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


SAMPLE_TEST_STRINGS = {
    "Spanish": [
        ("mispadrespuedenviajar", "mis padres pueden viajar"),
        ("elcielodespejadoesazul", "el cielo despejado es azul"),
        ("lacasarojaesgrande", "la casa roja es grande"),
        ("losestudiantesleennuevoslibros", "los estudiantes leen nuevos libros"),
        ("unagranfiestaesperada", "una gran fiesta esperada"),
    ],
    "English": [
        ("thequickbrownfoxjumpsoverthelazydog", "the quick brown fox jumps over the lazy dog"),
        ("naturalprocessingmodelsarefast", "natural processing models are fast"),
        ("shelovesreadinginterestingbooks", "she loves reading interesting books"),
        ("theweatherwastodayverynice", "the weather was today very nice"),
    ]
}


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def run_language_experiment(lang_name: str, train_sents: List, test_sents: List, eval_limit: int = 500) -> Dict[str, Any]:
    """Train pipeline and run comprehensive evaluations on test set."""
    print_section(f"TRAINING & EVALUATION: {lang_name.upper()}")
    print(f"[*] Training Sentences: {len(train_sents):,}")
    print(f"[*] Evaluation Sentences: {min(len(test_sents), eval_limit):,} (out of {len(test_sents):,})")

    eval_test_sents = test_sents[:eval_limit]

    t0 = time.time()
    pipeline = NLPJointPipeline(language=lang_name)
    pipeline.train(train_sents)
    train_time = time.time() - t0
    print(f"[+] Training completed in {train_time:.2f}s")
    print(f"    - Vocabulary Size: {len(pipeline.lm.vocab):,} words")
    print(f"    - Standard Tags: {len(pipeline.hmm_standard.tags)} tags")
    print(f"    - Morphology-Aware Tags: {len(pipeline.hmm_morph.tags)} tags")

    # 1. Evaluate Word Segmentation
    print("\n--- 1. Word Segmentation Evaluation ---")
    gold_words_test = [[w for w, _, _ in s] for s in eval_test_sents]
    unspaced_test = ["".join(w) for w in gold_words_test]

    # Proposed Trigram DP
    t_seg_prop = time.time()
    pred_words_dp = [pipeline.dp_segmenter.segment(u) for u in unspaced_test]
    dp_seg_time = time.time() - t_seg_prop
    dp_seg_results = evaluate_segmentation(gold_words_test, pred_words_dp)

    # Baseline Greedy Longest Match
    t_seg_base = time.time()
    pred_words_greedy = [pipeline.greedy_segmenter.segment(u) for u in unspaced_test]
    greedy_seg_time = time.time() - t_seg_base
    greedy_seg_results = evaluate_segmentation(gold_words_test, pred_words_greedy)

    seg_table = [
        ["Proposed: Trigram LM + DP", f"{dp_seg_results['precision']:.2f}%", f"{dp_seg_results['recall']:.2f}%", f"{dp_seg_results['f1']:.2f}%", f"{dp_seg_results['sentence_acc']:.2f}%", f"{dp_seg_time:.2f}s"],
        ["Baseline: Greedy Longest-Match", f"{greedy_seg_results['precision']:.2f}%", f"{greedy_seg_results['recall']:.2f}%", f"{greedy_seg_results['f1']:.2f}%", f"{greedy_seg_results['sentence_acc']:.2f}%", f"{greedy_seg_time:.2f}s"]
    ]
    headers_seg = ["Model", "Precision", "Recall", "Word F1", "Exact Sent Acc", "Time"]
    print(tabulate(seg_table, headers=headers_seg, tablefmt="github") if HAS_TABULATE else seg_table)

    # 2. Evaluate Gold-Segmented POS Tagging (Standard & Morphology)
    print("\n--- 2. POS Tagging Accuracy (Gold Segmented Words) ---")
    gold_standard_sents = [[(w, u) for w, u, _ in s] for s in eval_test_sents]
    gold_morph_sents = [[(w, m) for w, _, m in s] for s in eval_test_sents]

    hmm_std_res = evaluate_pos_tagger_gold(gold_standard_sents, pipeline.hmm_standard)
    base_std_res = evaluate_pos_tagger_gold(gold_standard_sents, pipeline.baseline_tagger_standard)

    hmm_morph_res = evaluate_pos_tagger_gold(gold_morph_sents, pipeline.hmm_morph)
    base_morph_res = evaluate_pos_tagger_gold(gold_morph_sents, pipeline.baseline_tagger_morph)

    pos_table = [
        ["Standard POS: Trigram HMM", f"{hmm_std_res['accuracy']:.2f}%", f"{hmm_std_res['correct_tokens']:,}/{hmm_std_res['total_tokens']:,}"],
        ["Standard POS: Baseline MFT", f"{base_std_res['accuracy']:.2f}%", f"{base_std_res['correct_tokens']:,}/{base_std_res['total_tokens']:,}"],
        ["Morph-Aware: Trigram HMM", f"{hmm_morph_res['accuracy']:.2f}%", f"{hmm_morph_res['correct_tokens']:,}/{hmm_morph_res['total_tokens']:,}"],
        ["Morph-Aware: Baseline MFT", f"{base_morph_res['accuracy']:.2f}%", f"{base_morph_res['correct_tokens']:,}/{base_morph_res['total_tokens']:,}"]
    ]
    headers_pos = ["Model / Tagset", "Tagging Accuracy", "Correct / Total Tokens"]
    print(tabulate(pos_table, headers=headers_pos, tablefmt="github") if HAS_TABULATE else pos_table)

    # 3. End-to-End Pipeline & Error-Source Breakdown (Part 5)
    print("\n--- 3. End-to-End Pipeline & Error-Source Breakdown ---")
    e2e_prop_std = evaluate_end_to_end_and_error_breakdown(eval_test_sents, pred_words_dp, pipeline.hmm_standard, use_morphology=False)
    e2e_base_std = evaluate_end_to_end_and_error_breakdown(eval_test_sents, pred_words_greedy, pipeline.baseline_tagger_standard, use_morphology=False)

    e2e_prop_morph = evaluate_end_to_end_and_error_breakdown(eval_test_sents, pred_words_dp, pipeline.hmm_morph, use_morphology=True)
    e2e_base_morph = evaluate_end_to_end_and_error_breakdown(eval_test_sents, pred_words_greedy, pipeline.baseline_tagger_morph, use_morphology=True)

    e2e_table = [
        ["Proposed Pipeline (Standard)", f"{e2e_prop_std['pipeline_accuracy']:.2f}%", f"{e2e_prop_std['total_pipeline_errors']:,}", f"{e2e_prop_std['seg_induced_errors']:,} ({e2e_prop_std['seg_error_pct']:.1f}%)", f"{e2e_prop_std['genuine_tagging_errors']:,} ({e2e_prop_std['genuine_error_pct']:.1f}%)"],
        ["Baseline Pipeline (Standard)", f"{e2e_base_std['pipeline_accuracy']:.2f}%", f"{e2e_base_std['total_pipeline_errors']:,}", f"{e2e_base_std['seg_induced_errors']:,} ({e2e_base_std['seg_error_pct']:.1f}%)", f"{e2e_base_std['genuine_tagging_errors']:,} ({e2e_base_std['genuine_error_pct']:.1f}%)"],
        ["Proposed Pipeline (Morph-Aware)", f"{e2e_prop_morph['pipeline_accuracy']:.2f}%", f"{e2e_prop_morph['total_pipeline_errors']:,}", f"{e2e_prop_morph['seg_induced_errors']:,} ({e2e_prop_morph['seg_error_pct']:.1f}%)", f"{e2e_prop_morph['genuine_tagging_errors']:,} ({e2e_prop_morph['genuine_error_pct']:.1f}%)"],
        ["Baseline Pipeline (Morph-Aware)", f"{e2e_base_morph['pipeline_accuracy']:.2f}%", f"{e2e_base_morph['total_pipeline_errors']:,}", f"{e2e_base_morph['seg_induced_errors']:,} ({e2e_base_morph['seg_error_pct']:.1f}%)", f"{e2e_base_morph['genuine_tagging_errors']:,} ({e2e_base_morph['genuine_error_pct']:.1f}%)"]
    ]
    headers_e2e = ["Pipeline Configuration", "Pipeline Acc", "Total Errors", "Seg-Induced Errors (% of errs)", "Genuine Tagging Errors (% of errs)"]
    print(tabulate(e2e_table, headers=headers_e2e, tablefmt="github") if HAS_TABULATE else e2e_table)

    # 4. Confusion Matrix Analysis (Top 8 Confusions)
    print("\n--- 4. Top Confused Tag Pairs (Standard POS) ---")
    conf_table_str = format_confusion_table(hmm_std_res["confusion_pairs"], top_k=8)
    print(conf_table_str)

    # 5. Run Demonstration on Sample Test Strings
    print("\n--- 5. Demonstration on Assignment & Sample Test Strings ---")
    sample_list = SAMPLE_TEST_STRINGS.get(lang_name, [])
    for unspaced, gold_ref in sample_list:
        pred_std = pipeline.process_proposed(unspaced, morphology_aware=False)
        pred_morph = pipeline.process_proposed(unspaced, morphology_aware=True)
        pred_base = pipeline.process_baseline(unspaced, morphology_aware=False)

        print(f"\n[Input String]: '{unspaced}'")
        print(f" [Reference] : {gold_ref}")
        print(f" [Trigram+DP Standard] : {pred_std}")
        print(f" [Trigram+DP Morph-Tag]: {pred_morph}")
        print(f" [Baseline Greedy+MFT] : {pred_base}")

    return {
        "pipeline": pipeline,
        "seg_dp": dp_seg_results,
        "seg_greedy": greedy_seg_results,
        "pos_hmm_std": hmm_std_res,
        "pos_base_std": base_std_res,
        "pos_hmm_morph": hmm_morph_res,
        "pos_base_morph": base_morph_res,
        "e2e_prop_std": e2e_prop_std,
        "e2e_base_std": e2e_base_std,
        "e2e_prop_morph": e2e_prop_morph,
        "e2e_base_morph": e2e_base_morph,
    }


def main():
    parser = argparse.ArgumentParser(description="NLP Group Assignment 1 - Question 1: Word Segmentation & POS Tagging")
    parser.add_argument("--eval-limit", type=int, default=500, help="Number of test sentences to evaluate (default: 500)")
    parser.add_argument("--interactive", action="store_true", help="Interactive terminal mode")
    parser.add_argument("--lang", type=str, choices=["english", "spanish", "both"], default="both", help="Language to run")
    args = parser.parse_args()

    print("================================================================================")
    print("  NLP Assignment 1 - Question 1: Word Segmentation & POS Tagging")
    print("================================================================================")

    results = {}

    if args.lang in ("english", "both"):
        print("\n[*] Loading English Brown Corpus...")
        en_train, en_test = load_english_brown(split_ratio=0.8, seed=42)
        results["English"] = run_language_experiment("English", en_train, en_test, eval_limit=args.eval_limit)

    if args.lang in ("spanish", "both"):
        print("\n[*] Loading Spanish UD-GSD Corpus...")
        es_train, es_dev, es_test = load_spanish_ud()
        results["Spanish"] = run_language_experiment("Spanish", es_train, es_test, eval_limit=args.eval_limit)

    if args.interactive:
        print_section("INTERACTIVE MODE")
        while True:
            try:
                text = input("\nEnter unspaced text (or 'q' to quit): ").strip()
                if not text or text.lower() == 'q':
                    break
                lang_choice = input("Select language (en/es) [default: en]: ").strip().lower() or "en"
                target_lang = "English" if lang_choice.startswith("en") else "Spanish"
                if target_lang not in results:
                    print(f"Language {target_lang} model not loaded.")
                    continue
                pipe = results[target_lang]["pipeline"]
                out_std = pipe.process_proposed(text, morphology_aware=False)
                out_morph = pipe.process_proposed(text, morphology_aware=True)
                print(f"Standard POS: {out_std}")
                print(f"Morph-Aware : {out_morph}")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
