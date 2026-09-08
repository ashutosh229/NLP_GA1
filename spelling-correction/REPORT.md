# Efficient Spelling Corrector — Report

## 1. Overview

This project implements an end-to-end spelling correction system for both
non-word and real-word errors in English text. The system is built on the
NLTK Brown corpus and is designed to correct mistakes such as
`sentnce -> sentence` and `I ate an apply -> I ate an apple` within an edit
distance of 1.

The implementation is structured as a small pipeline of modules:

| File | Part | Responsibility |
|---|---|---|
| `build_model.py` | — | One-shot script that builds and caches the model (`model.pkl`) |
| `src/model_builder.py` | 1 | Builds vocabulary, unigram counts, bigram counts, and the SymSpell-style deletes dictionary |
| `src/candidate_generation.py` | 2 | Implements both brute-force edit-distance-1 generation and the symmetric-delete method |
| `src/corrector.py` | 3 | Applies non-word correction and real-word correction using n-gram context |
| `src/evaluate.py` | 4 | Generates benchmark test sets and reports accuracy plus speed comparisons |
| `src/cli.py` | 5 | Runs the interactive terminal spelling-correction app |
| `src/utils.py` | — | Download helper, typo generation, and support functions |

## 2. Design choices

**Corpus and language model.** The project uses the Brown corpus to build a
vocabulary and frequency model. From this, it derives:

- a unigram frequency distribution,
- a bigram language model with add-one (Laplace) smoothing,
- a deletion dictionary used by the fast candidate-generation strategy.

This gives the corrector a data-driven way to prefer common words over rare
alternatives and to score candidate phrases in context.

**Candidate generation.** The system implements two candidate-generation
strategies for edit-distance-1 words:

1. Method A: brute-force generation over possible edits using character
   deletions, transpositions, substitutions, and insertions.
2. Method B: symmetric-delete spelling correction, which precomputes one-character
   deletions for all vocabulary words once and then checks only deletions of
   the query word at lookup time.

Method B is faster because it moves the
expensive alphabet-based generation step out of the per-query hot path.

**Correction logic.**

- For non-word errors, the system ranks all edit-distance-1 candidates by
  their unigram frequency and picks the highest-frequency word.
- For real-word errors, it compares the probability of the original phrase
  with the probability of a candidate phrase in context from the bigram model.
  It only accepts a correction when the candidate is sufficiently more likely
  than the original, controlled by a log-probability margin.

This avoids over-correcting noisy or ambiguous contexts while still allowing
real-word replacements when evidence is strong.

**Evaluation strategy.** The evaluation script auto-generates test cases by
sampling 10% of the Brown corpus sentences and injecting one single-edit typo
per sentence. It produces separate benchmark sets for non-word and real-word
errors, then reports both correction accuracy and method runtime.

**Interactive deployment.** The project includes a live terminal CLI that takes
user input, corrects sentences, highlights changed words, and measures query
latency. This makes the project practical beyond offline evaluation.

## 3. Data

- Corpus: NLTK Brown corpus
- Model cache: `model.pkl`
- Test generation: 10% of Brown corpus sentences, one typo per sentence
- Benchmark sizes reported in the README:
  - Non-word test cases: 5,582
  - Real-word test cases: 3,713

The project automatically downloads the Brown corpus on first use and caches the
trained model for future runs.

## 4. Results

The README includes the following measured output from evaluation:

| Metric | Score |
|---|---|
| Non-word correction accuracy | 81.03% |
| Real-word correction accuracy | 68.87% |
| Speed benchmark batch size | 1,000 words |
| Method A runtime | 0.0915s |
| Method B runtime | 0.0091s |
| Speedup factor | ~10.1x |

Method B is substantially faster because Method A has
to generate and test many candidate strings for every query, while Method B
only computes the query word's one-character deletions and looks them up in a
prebuilt hash map.

## 5. Observation

The main technical contribution is the combination of:

- efficient candidate enumeration,
- corpus-driven frequency ranking,
- contextual real-word correction via a bigram model,
- evaluation and benchmarking driven by automatically generated noisy samples.

Important implementation details:

- Smoothing is applied using add-one (Laplace) smoothing so unseen bigrams do
  not lead to zero-probability errors.
- Real-word corrections require a substantial probability gain over the
  original phrase; this threshold reduces over-correction.
- Method B is still guaranteed to match Method A's edit-distance-1 behavior
  because there is a final verification step to reject false positives caused by
  deletion collisions.

The design makes a sensible trade-off between accuracy and speed: the model is
precomputed once; then many queries benefit from low-latency lookup-based
generation.

## 6. Reproducing

The project can be set up as follows:

```bash
cd spelling_corrector
python -m venv .venv
.venv/Scripts/Activate
pip install -r requirements.txt
```

Then build the model explicitly or let the project build it on first run:

```bash
python build_model.py
```

To evaluate the model and benchmark the candidate-generation methods:

```bash
python -m src.evaluate
```

To run the live interactive CLI:

```bash
python -m src.cli
```

Example CLI interaction from the README:

```text
=== Spelling Corrector — Interactive CLI ===
Type a sentence and press Enter to correct it.
Type 'exit' to quit.

> I hav a good feeling about this.
I **had** a good feeling about this.
[latency: 1.44 ms]

> This is a test sentnce.
This is a test **sentence**.
[latency: 1.83 ms]

> I would like to sea the world.
I would like to **see** the world.
[latency: 1.71 ms]

> exit
Goodbye!
```

## 7. Conclusion

The project demonstrates a practical and educational spelling-correction
pipeline that combines traditional NLP modeling with efficient algorithmic
optimization. It balances accuracy and runtime while exposing a clear path from
model construction to evaluation to interactive use.
