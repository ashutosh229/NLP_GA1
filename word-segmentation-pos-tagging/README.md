# NLP Group Assignment 1 — Question 1: Word Segmentation & POS Tagging

An end-to-end NLP system for unspaced sentence segmentation and Part-of-Speech (POS) tagging with standard and morphology-aware tagsets across **English** (Brown Corpus) and **Spanish** (Universal Dependencies `UD_Spanish-GSD`).

---

## 🌟 Features & Architecture

1. **Word Segmentation Model**:
   - **Trigram Language Model**: Linear interpolation smoothing across trigram, bigram, unigram maximum-likelihood estimates, plus character-level subword LM for out-of-vocabulary (OOV) tokens.
   - **Dynamic Programming (Viterbi)**: Decodes the globally optimal sequence of word boundaries from continuous unspaced text using beam search.
   - **Baseline Segmenter**: Greedy longest-match dictionary scan.

2. **Part-of-Speech (POS) Tagging Model**:
   - **Trigram Hidden Markov Model (HMM)**: Second-order tag transitions $P(t_i \mid t_{i-2}, t_{i-1})$ with linear interpolation smoothing and Lidstone/suffix-smoothed word emission probabilities $P(w_i \mid t_i)$.
   - **Morphology-Aware Extension**: Fine-grained morphological tags (e.g., `NOUN-Masc-Plur`, `ADJ-Fem-Sing`, `DET-Fem-Sing`, `VERB-Sing3`) modeling grammatical agreement constraints across adjacent tokens.
   - **Baseline Tagger**: Most-Frequent-Tag (MFT) baseline.

3. **Evaluation Suite**:
   - Word segmentation Precision, Recall, F1, and Exact Sentence Match.
   - Gold-segmented POS tagging accuracy vs. Full pipeline accuracy.
   - Detailed Confusion Matrices and top confused grammatical category pairs.
   - **Error-Source Breakdown**: Exact separation of errors caused by segmentation mistakes vs. genuine POS tagging errors.

---

## 📁 Repository Structure

```
word-segmentation-pos-tagging/
├── data/
│   └── UD_Spanish-GSD/             # Universal Dependencies Spanish Treebank
│       ├── es_gsd-ud-train.conllu
│       ├── es_gsd-ud-dev.conllu
│       └── es_gsd-ud-test.conllu
├── data_loader.py                  # Brown (80/20) & UD-GSD dataset parsers
├── segmentation.py                 # Trigram LM + Viterbi DP + Greedy baseline
├── pos_tagger.py                   # Trigram HMM + Morphology tags + MFT baseline
├── pipeline.py                     # Unified end-to-end joint inference pipeline
├── evaluate.py                     # Comprehensive evaluation & error breakdown
├── main.py                         # CLI entrypoint for training, evaluation, & demos
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation & execution guide
└── REPORT.md                       # Full comparative analysis report
```

---

## 🚀 Quickstart & Usage

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Full Evaluation & Benchmark

Train all models on English (Brown Corpus) and Spanish (`UD_Spanish-GSD`) and evaluate against the baselines:

```bash
python main.py --eval-limit 400
```

### 3. Evaluate a Single Language

```bash
python main.py --lang english --eval-limit 200
python main.py --lang spanish --eval-limit 200
```

### 4. Interactive Mode

Test custom unspaced strings in real-time:

```bash
python main.py --interactive
```

---

## 📊 Summary of Benchmark Results

### Word Segmentation Performance

| Language | Model | Precision | Recall | Word F1 | Sentence Exact Match |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **English** | **Proposed Trigram LM + DP** | **96.16%** | **96.73%** | **96.45%** | **69.25%** |
| | Baseline Greedy Longest-Match | 63.24% | 75.54% | 68.85% | 16.00% |
| **Spanish** | **Proposed Trigram LM + DP** | **89.96%** | **90.86%** | **90.40%** | **30.00%** |
| | Baseline Greedy Longest-Match | 45.43% | 58.83% | 51.27% | 4.75% |

### End-to-End Pipeline & Error Breakdown

| Language | Pipeline Mode | Pipeline Accuracy | Total Errors | Segmentation-Induced Errors | Genuine Tagging Errors |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **English** | **Proposed (Standard)** | **94.29%** | 477 | 292 (61.2%) | 185 (38.8%) |
| | Baseline (Standard) | 71.81% | 3,601 | 3,319 (92.2%) | 282 (7.8%) |
| **Spanish** | **Proposed (Standard)** | **86.75%** | 1,457 | 1,037 (71.2%) | 420 (28.8%) |
| | Baseline (Standard) | 54.67% | 7,651 | 7,226 (94.4%) | 425 (5.6%) |

See [REPORT.md](REPORT.md) for the complete comparative analysis report.
