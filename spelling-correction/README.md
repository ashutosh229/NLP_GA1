# Efficient Spelling Corrector — Build, Benchmark, Deploy

A spelling corrector that fixes both **non-word errors** (e.g. `sentnce` →
`sentence`) and **real-word errors** (e.g. `I ate an apply` → `I ate an apple`)
within an edit distance of 1, built on the NLTK Brown corpus. It implements
and benchmarks two different candidate-generation strategies, evaluates
accuracy on an auto-generated test set, and ships as a live interactive
terminal application.

## Description

This project answers the assignment "Building, Benchmarking, and Deploying
an Efficient Spelling Corrector" end-to-end:

- **Part 1 — Corpus & Model Preparation**: builds a vocabulary + unigram
  frequency distribution and a bigram language model from the Brown corpus.
- **Part 2 — Candidate Generation**: implements two ways of finding
  edit-distance-1 candidates for a word:
  - **Method A**: standard brute-force edit-distance-1 generation
    (deletions, transpositions, substitutions, insertions over the full
    alphabet).
  - **Method B**: Symmetric Delete Spelling Correction (SymSpell-style) 
    precomputes every one-character deletion of every vocabulary word once,
    then only computes the deletions of the *query* word at lookup time.
- **Part 3 — Correction Logic**: non-word correction picks the
  highest-frequency candidate; real-word correction uses the bigram model
  to compare the probability of the original phrase against candidate
  phrases in context.
- **Part 4 — Evaluation & "Speed Demon" Benchmark**: auto-generates a
  test set (10% of Brown sentences, one injected typo per sentence),
  reports accuracy for both error types, and times Method A vs. Method B
  over an identical 1,000-word batch.
- **Part 5 — Live Interactive Application**: a continuous terminal CLI
  that corrects sentences you type, highlights changed words, and reports
  latency.

## Directory Structure

```
spelling_corrector/
├── README.md                    <- this file
├── requirements.txt              <- Python dependencies (nltk)
├── build_model.py                <- one-shot script: builds & caches model.pkl
├── model.pkl                     <- (generated) cached vocab/bigram/SymSpell model
└── src/
    ├── __init__.py
    ├── utils.py                  <- corpus download helper, typo generator
    ├── model_builder.py           <- Part 1: vocab, unigram & bigram models,
    │                                 + SymSpell "deletes dictionary" preprocessing
    ├── candidate_generation.py    <- Part 2: Method A and Method B
    ├── corrector.py                <- Part 3: non-word & real-word correction logic
    ├── evaluate.py                 <- Part 4: test-set generation, accuracy,
    │                                 Speed Demon benchmark
    └── cli.py                      <- Part 5: live interactive terminal app
```

## Setup

Requires Python 3.8+.

```bash
cd spelling_corrector
python -m venv .venv
.venv/Scripts/Activate
pip install -r requirements.txt
```

The first time you run *any* entry point (`build_model.py`, `src.evaluate`,
or `src.cli`), it will automatically download the NLTK **Brown corpus**
(one-time, ~a few MB) and cache the built model as `model.pkl` in the
project root so subsequent runs are instant. You can also trigger this
explicitly up front:

```bash
python build_model.py
```

Typical build output:
```
[model_builder] vocab size = 40234 (0.66s)
[model_builder] bigram pairs = 388815 (2.23s)
[model_builder] deletes_dict entries = 280032 (0.82s)
Done. Model cached at ./model.pkl
```

## How to Use — Per Part

### Part 1 & 2 — Build the model / inspect candidate generation
The model (vocabulary, unigram counts, bigram counts, and the Method-B
"deletes dictionary") is built by `src/model_builder.py` and cached to
`model.pkl`. You normally don't call this directly — `build_model.py`,
`evaluate.py`, and `cli.py` all load/build it automatically. To inspect
candidate generation directly:

```python
from src.model_builder import load_model
from src.corrector import SpellingCorrector

model = load_model()
c = SpellingCorrector(model)

print(c.candidates_method_a("helo"))   # brute-force ED1 ∩ vocab Output -> {'hilo', 'hero', 'halo', 'helm', 'held', 'hel', 'hell', 'hello', 'help'}
print(c.candidates_method_b("helo"))   # symmetric-delete ED1 ∩ vocab Output - > {'hilo', 'hero', 'halo', 'helm', 'hel', 'held', 'hell', 'hello', 'help'}
```

### Part 3 — Correction logic

```python
c.correct_nonword("sentnce")
#Output -> ('sentence', {'sentence'})

c.correct_realword(prev_word="to", word="sea", next_word="the")
#Output -> ('see', True)   # bigram-in-context beats the original
```

### Part 4 — Evaluation & Speed Demon Benchmark

```bash
python -m src.evaluate
```

This will:
1. Load (or build) the model.
2. Sample 10% of Brown corpus sentences and inject one single-edit typo
   per sentence, producing a non-word test set and a real-word test set.
3. Report accuracy for both.
4. Build a 1,000-word misspelling batch and time Method A vs. Method B,
   printing a conclusion explaining the speed gap.

My output :
```
[evaluate] Loading cached model from model.pkl ...
[evaluate] Generating test sets (10% of Brown corpus sentences) ...
  Non-word test cases:  5582
  Real-word test cases: 3713

[Accuracy] Non-word error correction:  81.03%
[Accuracy] Real-word error correction: 68.87%

[evaluate] Running Speed Demon Benchmark (n=1000) ...

[Speed Demon Benchmark] batch size = 1000
  Method A (standard edit-distance-1 generation): 0.0915s (0.0915 ms/word)
  Method B (symmetric delete):                    0.0091s (0.0091 ms/word)
  Method B is ~10.1x faster than Method A.

Conclusion: Method A must, for every word of length L, build and
hash roughly 54*L + 25 candidate STRINGS (deletions, transpositions,
26 substitutions per position, 26 insertions per position+1), then
test EACH one for vocabulary membership. The cost scales with the
size of the alphabet (26) and is paid fresh for every single query.
Method B never touches the alphabet at query time: it only computes
the L one-character deletions of the query word and performs L
dictionary lookups against a hash map that was already built once
up front (build_deletes_dict, paid a single time during model
preparation, not per query). Because L << 54*L + 25, and because
the expensive alphabet-driven string generation has been moved
entirely out of the per-query hot path, Method B achieves its
large constant-factor speedup — it trades a one-time O(V*avg_len)
preprocessing cost for an O(L) lookup cost per query, versus
Method A's O(26*L) generation cost paid every single time.
```

**Why Method B is faster:** Method A must, for every word of length *L*,
generate and hash roughly `54*L + 25` candidate strings (all deletions,
transpositions, and 26 substitutions/insertions per position) and test
each one for vocabulary membership — this alphabet-driven cost is paid
fresh on *every single query*. Method B only computes the *L*
one-character deletions of the query and performs *L* dictionary lookups
against a hash map that was already built **once**, up front, during model
preparation (`build_deletes_dict`). Because the expensive alphabet loop
has been moved entirely out of the per-query path, Method B trades a
one-time `O(V * avg_word_len)` preprocessing cost for an `O(L)` per-query
lookup cost, instead of Method A's `O(26*L)` per-query generation cost.

### Part 5 — Live interactive CLI

```bash
python -m src.cli
```

```
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

Changed words are wrapped in `**asterisks**` (and additionally shown in
green if your terminal supports ANSI colour) so they're easy to spot at a
glance. Type `exit` at any time to quit.

## Notes 

- **Smoothing**: the bigram model uses add-one (Laplace) smoothing so
  unseen bigrams never produce a zero probability, avoiding `log(0)`
  errors during real-word scoring.
- **Real-word correction threshold**: a candidate must beat the original
  phrase's log-probability by a margin (`log_margin`, default `1.5` nats)
  before it's suggested and this avoids over-correcting on noisy bigram
  estimates from a single mid-sized corpus.
- **Method B correctness**: the raw symmetric-delete lookup (matching a
  deletion of the query against a deletion of a vocabulary word) can, in
  rare cases, match pairs that are genuinely edit-distance 2 apart (e.g.
  `at` and `to` both reduce to `t`). `candidate_generation.is_edit_distance_1`
  performs a cheap final verification pass so Method B's output always
  matches Method A's edit-distance-1 guarantee exactly.
