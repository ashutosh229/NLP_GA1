"""
Part 2.1: Feature extraction.

For a given parser configuration (+ access to the sentence's POS tags),
extract the four simple features required by the assignment:
    * POS tag of the word on top of the stack        (s0_pos)
    * POS tag of the second word on the stack          (s1_pos)
    * POS tag of the first word in the buffer          (b0_pos)
    * POS tag of the second word in the buffer         (b1_pos)

Each feature is returned in a dict of {feature_name: value} so it plugs
directly into scikit-learn's DictVectorizer. Missing elements (e.g. stack
has only ROOT, or buffer is empty) are represented with a special
"<NULL>" value rather than omitted, so every configuration produces a
feature vector of the same shape.
"""

from typing import Dict

from transition_system import Configuration, ROOT_ID

NULL = "<NULL>"
ROOT_POS = "<ROOT>"


def _pos_of(token_id: int, pos_by_id: Dict[int, str]) -> str:
    if token_id == ROOT_ID:
        return ROOT_POS
    return pos_by_id.get(token_id, NULL)


def extract_features(config: Configuration, pos_by_id: Dict[int, str]) -> Dict[str, str]:
    """
    config:    current Configuration (stack / buffer / arcs)
    pos_by_id: dict mapping token id -> UPOS tag for the sentence being parsed
    """
    stack = config.stack
    buffer = config.buffer

    s0_pos = _pos_of(stack[-1], pos_by_id) if len(stack) >= 1 else NULL
    s1_pos = _pos_of(stack[-2], pos_by_id) if len(stack) >= 2 else NULL
    b0_pos = _pos_of(buffer[0], pos_by_id) if len(buffer) >= 1 else NULL
    b1_pos = _pos_of(buffer[1], pos_by_id) if len(buffer) >= 2 else NULL

    return {
        "s0_pos": s0_pos,
        "s1_pos": s1_pos,
        "b0_pos": b0_pos,
        "b1_pos": b1_pos,
    }


if __name__ == "__main__":
    from conllu_io import read_conllu
    from oracle import run_oracle

    sents = read_conllu("UD_English-EWT/en_ewt-ud-dev.conllu")
    s = sents[0]
    pos_by_id = {s.ids[i]: s.upos[i] for i in range(len(s))}
    for (cfg, trans, label) in run_oracle(s):
        feats = extract_features(cfg, pos_by_id)
        print(feats, "->", trans, label)
