# Transition-Based Dependency Parser — Report

## 1. Overview

This project implements a greedy, arc-standard transition-based dependency
parser trained on the Universal Dependencies English-EWT treebank, exactly
as specified in the assignment. The pipeline is split across small,
single-purpose modules:

| File | Part | Responsibility |
|---|---|---|
| `conllu_io.py` | 1.1 | Parses `.conllu` files into `Sentence` objects (forms, UPOS, gold heads/labels) |
| `transition_system.py` | — | `Configuration` class + SHIFT / LEFT-ARC / RIGHT-ARC transitions |
| `oracle.py` | 1.2 | Static oracle: replays the gold tree through arc-standard to emit `(configuration, transition, label)` training instances |
| `features.py` | 2.1 | Extracts the 4 required POS-tag features from a configuration |
| `train.py` | 2.2 | Builds the training set and fits a scikit-learn classifier |
| `parser.py` | 3.1 | Greedy parsing loop driven by the trained classifier |
| `evaluate.py` | 3.2 | Computes UAS/LAS on the dev set |
| `main.py` | — | Runs the whole pipeline end to end |

## 2. Design choices

**Transition system.** Implemented exactly as specified: `stack` starts as
`[ROOT]` (id 0), `buffer` holds all sentence tokens in order. LEFT-ARC pops
`stack[-2]` (making `stack[-1]` its head) and is disallowed when
`stack[-2]` is ROOT, since the root token can never be a dependent.
RIGHT-ARC pops `stack[-1]` (making `stack[-2]` its head).

**Oracle.** A classic *static* oracle (Nivre, 2004): at each configuration,
LEFT-ARC fires if the gold head of `stack[-2]` is `stack[-1]`; RIGHT-ARC
fires if the gold head of `stack[-1]` is `stack[-2]` **and** `stack[-1]`
has already collected all of its own gold children (otherwise those
children would be orphaned once `stack[-1]` is popped); otherwise SHIFT.
This oracle can only reproduce **projective** trees. Since arc-standard is
inherently limited to projective parses, sentences whose gold tree is
non-projective are detected (the oracle gets stuck with no legal,
gold-consistent move) and skipped when building the training set. On
`en_ewt-ud-train.conllu` this affects 287 of 12,544 sentences (~2.3%),
consistent with the known non-projectivity rate of English-EWT.

**Multiword tokens / empty nodes.** CoNLL-U ranges like `8-9` (contractions
such as "don't") and empty nodes like `8.1` (elided material) are skipped
during reading, keeping only tokens with plain integer IDs — these are the
tokens that actually receive a syntactic head in the tree.

**Classifier.** The oracle emits a `(transition, label)` pair per step. We
collapse this into a single string class (e.g. `LEFT-ARC:det`,
`RIGHT-ARC:nsubj`, or plain `SHIFT`), so one multi-class classifier jointly
predicts the transition type and, where relevant, the dependency label.
We use scikit-learn's `LogisticRegression` (`lbfgs` solver) over
one-hot-encoded features from `DictVectorizer`. With only 4 categorical
POS-tag features (~73 one-hot dimensions after vectorization) this trains
in under 2 minutes on the full training set and 82 output classes.

**Parsing loop / legality masking.** At each step the classifier ranks all
classes by predicted probability (`predict_proba`); we walk down the
ranking and take the first-ranked transition that is *legal* in the
current configuration (e.g. LEFT-ARC/RIGHT-ARC need ≥2 items on the
stack; LEFT-ARC additionally needs `stack[-2] != ROOT`). This guarantees
the parser always terminates in a well-formed tree, even though the raw
classifier occasionally prefers an illegal move.

## 3. Data

- Training: `en_ewt-ud-train.conllu` — 12,544 sentences, 12,257 used after
  skipping non-projective ones, yielding **392,242** training instances.
- Evaluation: `en_ewt-ud-dev.conllu` — 2,001 sentences / 25,148 tokens.
- POS tags used for both training and parsing are the gold UPOS tags from
  the treebank (the assignment's Part 3 spec defines the parser's input as
  "a sentence — a list of words **and their POS tags**", i.e. POS tagging
  itself is out of scope for this assignment).

## 4. Results

| Metric | Score |
|---|---|
| Training-set transition accuracy | 80.6% |
| **Dev-set LAS** | **56.84%** |
| Dev-set UAS | 66.75% |
| Parsing speed | ~3,400 tokens/sec |

## 5. Discussion

A LAS in the mid-50s is expected and reasonable given the deliberately
minimal feature set (only 4 POS tags — no lexical/word-form features, no
distance features, no features of already-built arcs or their labels).
For comparison, classic arc-standard parsers with richer feature
templates (word forms, lemmas, leftmost/rightmost children, stack-buffer
distance, etc.) typically reach 85–90+ LAS on this treebank. The gap here
comes almost entirely from feature poverty rather than a flawed transition
system or oracle — the oracle reconstructs 100% of projective gold trees
exactly (verified by asserting the oracle's own arc set matches the gold
tree for every sentence used in training), so all evaluation error is
attributable to the classifier's limited view of each configuration.

**Error patterns observed:** the parser is noticeably better at UAS than
LAS (a ~10-point gap), meaning it more often gets the *attachment*
approximately right but the *label* wrong — unsurprising since POS tags
alone are a weak signal for finer-grained relations (e.g. distinguishing
`obj` vs `obl`, or `nsubj` vs `csubj`). It also struggles most with
long-distance attachments (e.g. PP-attachment ambiguity, as in the
"telescope" example) and with coordination, both classic hard cases for
greedy, feature-light parsers with no beam search.

**Straightforward extensions** (not implemented, to stay within the
assignment's specified feature set) that would meaningfully raise LAS:
adding word-form (not just POS) features for `s0`/`b0`; features of the
already-built leftmost/rightmost children of `s0`/`s1`; and switching from
greedy 1-best decoding to beam search.

## 6. Reproducing

```bash
git clone --depth 1 https://github.com/UniversalDependencies/UD_English-EWT.git
pip install scikit-learn numpy
python3 main.py        # trains, evaluates, and demos the 3 example sentences
```

Individual stages can also be run separately: `python3 train.py`,
`python3 evaluate.py`, `python3 parser.py`.
