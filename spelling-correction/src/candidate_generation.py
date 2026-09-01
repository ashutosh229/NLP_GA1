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




