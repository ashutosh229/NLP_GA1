# Question 4: Integrated Background Editor

Live Word Segmentation, Spelling Correction, and Constituency-Based Grammar Checking

This project implements an integrated background-running editor for live text streams, uniting:
1. **Word Segmentation & Feature-Based POS Tagger** (Reused from Question 1)
2. **Spelling Corrector & Candidate Generators** (Reused from Question 3)
3. **PCFG Constituency Parser** with CKY/Viterbi dynamic programming (Trained on Penn Treebank sample)
4. **Shared Add-$k$ Smoothed N-Gram Language Model** (Trained on Brown Corpus)
5. **Fast-Typing Merge Simulation** (with spacebar drop probability $p = 0.08$)
6. **Three-Tier Real-Time Alerts** (`[SEGMENT-ALERT]`, `[SPELL-ALERT]`, `[GRAMMAR-ALERT]`)
7. **Interactive Streamlit Web Application** & **Speed Demon Benchmark**

---

## Directory Structure

```text
background-editor/
├── requirements.txt       # Python dependencies (streamlit, nltk, scikit-learn, etc.)
├── pcfg_parser.py         # PCFG induction, CKY parser, and tagset reconciliation
├── language_model.py      # Shared add-k smoothed bigram and trigram LM
├── editor_engine.py       # Live stream editor engine integrating Q1 & Q3
├── evaluator.py           # Speed Demon benchmark (1,000 words) & passage comparison table
├── app.py                 # Interactive Streamlit Web Application
├── run_benchmark.py       # CLI runner for Speed Demon benchmark & sample passage runs
├── REPORT.md              # Comparative analysis report with parameter justifications & runs
└── README.md              # This documentation
```

---

## Installation & Setup

1. Activate the dedicated virtual environment from `NLP_GA1`:
   ```powershell
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -r background-editor/requirements.txt
   ```

---

## Running the Components

### 1. Launch the Streamlit Live Web Application
```bash
streamlit run background-editor/app.py
```
- **Live Typing Mode:** Type or paste text incrementally into the live editor. Observe live segmentation, spelling, and grammar alert badges as tokens arrive.
- **Simulated Typing Stream:** Streams sample paragraphs from Gutenberg, Brown, or custom texts with live animation and spacebar drop simulation.
- **Speed Demon Benchmark:** Run the 1,000-word latency test interactively.

### 2. Run the Speed Demon Benchmark & Sample Runs via CLI
```bash
python background-editor/run_benchmark.py
```
This runs:
- The 1,000-word Speed Demon latency benchmark comparing full live-check vs. isolated grammar check.
- Two complete sample passage runs with all live alerts, metrics, and final sentence comparison tables.

---

## Key Design Parameters & Justifications

- **Merge Probability ($p = 0.08$):** Mimics natural typing cadences where ~8% of words accidentally merge due to a missed spacebar stroke, providing realistic input for Question 1's beam-search segmenter.
- **Trigger Interval ($N = 5$ words):** Triggers sliding-window perplexity checks every 5 words, ensuring real-time responsiveness without incurring excessive n-gram re-evaluation overhead.
- **Smoothing Parameter ($k = 0.05$):** Avoids over-flattening the probability distribution in large Brown corpus vocabularies while granting sufficient probability mass to rare transitions.
- **Tagset Reconciliation:** Uses deterministic `BROWN_TO_PTB` dictionary mapping to harmonize Brown/Universal POS tags produced by Q1 into Penn Treebank tags for PCFG parsing.
