"""
candidate_generation.py
------------------------
PART 2: Candidate Generation Methods

Method A: Standard Edit Distance 1 Generation
    Generates EVERY string reachable from `word` via a single deletion,
    transposition, replacement (substitution) or insertion. This is the
    classic Norvig-style edit-distance-1 generator. It produces a large
    set of strings (most of which are not real words); the caller is
    expected to filter against the vocabulary.

Method B: Symmetric Delete Spelling Correction (SymSpell, ED=1)
    Much faster: only ever generates the (len(word)) one-character
    deletions of the input, then looks them up in a precomputed
    deletes-dictionary (built once from the vocabulary in
    model_builder.build_deletes_dict). No alphabet loop, no candidate
    string ever gets "generated and discarded" the way Method A does.
"""

import string

ALPHABET = string.ascii_lowercase


# Method A: Standard edit distance 1
def edit_distance_1(word):
    """
    Returns the SET of all strings at edit distance exactly 1 from `word`
    (deletions, transpositions, substitutions, insertions).

    This mirrors the well-known Norvig spelling-corrector formulation.
    """
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in ALPHABET]
    inserts = [L + c + R for L, R in splits for c in ALPHABET]

    return set(deletes + transposes + replaces + inserts)


def method_a_candidates(word, vocab):
    """
    Full Method A pipeline for use by the corrector: generate all ED1
    strings, then keep only the ones that are real vocabulary words.

    Note: `edit_distance_1` can regenerate `word` itself (e.g. substituting
    a letter with the SAME letter, or inserting/deleting+reinserting in a
    way that reproduces the original string) — that's not an actual
    correction, so it's explicitly excluded here.
    """
    return {w for w in edit_distance_1(word) if w in vocab and w != word}


# Method B: Symmetric Delete Spelling Correction
def word_deletes(word):
    """All one-character deletions of `word` (the query-side of SymSpell)."""
    return {word[:i] + word[i + 1:] for i in range(len(word))}


def is_edit_distance_1(a, b):
    """
    Cheap O(len) verification that two strings are within TRUE edit
    distance 1 (deletion, insertion, substitution, or adjacent
    transposition). This is needed because the naive symmetric-delete
    lookup (matching on a shared one-deletion form from EACH side) can
    also match pairs that are actually edit distance 2 apart (e.g.
    'at' and 'to' both reduce to 't' by deleting one character each,
    but are themselves 2 substitutions apart). We use this as a final
    verification/pruning step so Method B's returned candidates are
    always genuinely at edit distance <= 1, matching Method A exactly.
    """
    if a == b:
        return False
    len_a, len_b = len(a), len(b)
    if abs(len_a - len_b) > 1:
        return False

    if len_a == len_b:
        # substitution: exactly one differing position
        diffs = [i for i in range(len_a) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        # adjacent transposition: exactly two adjacent differing positions
        # that are swaps of each other
        if len(diffs) == 2 and diffs[1] == diffs[0] + 1:
            i, j = diffs
            return a[i] == b[j] and a[j] == b[i]
        return False

    # insertion/deletion: shorter must be the longer with exactly one
    # character removed
    longer, shorter = (a, b) if len_a > len_b else (b, a)
    i = j = 0
    skipped = False
    while i < len(longer) and j < len(shorter):
        if longer[i] == shorter[j]:
            i += 1
            j += 1
        elif not skipped:
            skipped = True
            i += 1
        else:
            return False
    return True


def method_b_candidates(word, vocab, deletes_dict):
    """
    Full Method B pipeline.

    deletes_dict: dict mapping a one-character deletion of a vocab word ->
                  set of vocab word(s) it came from (built once, up front,
                  in model_builder.build_deletes_dict).

    generate all one-character deletions of the
    misspelled word, then look those up in the preprocessed dictionary.
    We additionally check two boundary cases so the method correctly
    covers the full edit-distance-1 space (insertion & deletion errors
    at the query's own length), which is:

      1. word itself in vocab -> already correct, no candidates needed.
      2. A deletion of `word` is itself a real vocab word directly
         (handles: query = vocab_word + 1 inserted character).
      3. `word` itself is a key in deletes_dict (handles: query =
         vocab_word - 1 deleted character, i.e. the query IS someone
         else's deletion).
      4. A deletion of `word` is a key in deletes_dict (handles:
         substitution / transposition errors, where deleting the same
         relative position from both strings converges to one form).
    """
    candidates = set()

    deletions = word_deletes(word)

    for d in deletions:
        if d in vocab:                      # case 2
            candidates.add(d)
        if d in deletes_dict:               # case 4
            candidates |= deletes_dict[d]

    if word in deletes_dict:                # case 3
        candidates |= deletes_dict[word]

    candidates.discard(word)

    candidates = {c for c in candidates if is_edit_distance_1(word, c)}
    return candidates
