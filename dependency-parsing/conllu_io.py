"""
Part 1.1: CoNLL-U reader.

Reads a .conllu file and returns a list of sentences. Each sentence is a
dict with parallel lists:
    ids:    1-based token ids (int)
    forms:  surface word forms
    upos:   universal POS tags
    heads:  gold head id for each token (0 = root)
    deprels: gold dependency relation label for each token

Multiword tokens (e.g. "8-9") and empty nodes (e.g. "8.1") are skipped, as
is standard practice, since they don't get their own syntactic arc.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Sentence:
    ids: List[int] = field(default_factory=list)
    forms: List[str] = field(default_factory=list)
    upos: List[str] = field(default_factory=list)
    heads: List[int] = field(default_factory=list)
    deprels: List[str] = field(default_factory=list)

    def __len__(self):
        return len(self.ids)


def _is_normal_token_id(tok_id: str) -> bool:
    """True for plain integer ids; False for '8-9' (MWT) or '8.1' (empty node)."""
    return tok_id.isdigit()


def read_conllu(path: str) -> List[Sentence]:
    sentences: List[Sentence] = []
    current = Sentence()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line.strip() == "":
                if len(current) > 0:
                    sentences.append(current)
                current = Sentence()
                continue

            cols = line.split("\t")
            if len(cols) < 8:
                continue
            tok_id, form, lemma, upos, xpos, feats, head, deprel = cols[:8]

            if not _is_normal_token_id(tok_id):
                continue  # skip multiword tokens / empty nodes

            current.ids.append(int(tok_id))
            current.forms.append(form)
            current.upos.append(upos)
            # some heads are "_" for orphan/empty-node artifacts; guard anyway
            current.heads.append(int(head) if head.isdigit() else -1)
            current.deprels.append(deprel)

    if len(current) > 0:
        sentences.append(current)

    return sentences


if __name__ == "__main__":
    sents = read_conllu("UD_English-EWT/en_ewt-ud-dev.conllu")
    print(f"Read {len(sents)} sentences")
    s = sents[0]
    print(s.forms)
    print(s.upos)
    print(s.heads)
    print(s.deprels)
