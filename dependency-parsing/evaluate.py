"""
Part 3.2: Evaluation -- Labeled Attachment Score (LAS) on the dev set.

LAS = percentage of (non-root-only) tokens for which the parser predicted
BOTH the correct head AND the correct dependency label, out of all tokens
in the evaluation set.

We also report UAS (Unlabeled Attachment Score, head only) since it's a
useful diagnostic, though only LAS is required by the assignment.

The parser is fed the *gold* UPOS tags for each sentence, since the
assignment's Part 3 spec defines the parser's input as "a sentence (a list
of words and their POS tags)" -- i.e. POS tagging itself is out of scope.
"""

import time

from conllu_io import read_conllu
from parser import DependencyParser


def evaluate(model_path: str, dev_path: str):
    parser = DependencyParser(model_path)
    sentences = read_conllu(dev_path)

    total_tokens = 0
    correct_uas = 0
    correct_las = 0

    t0 = time.time()
    for sent in sentences:
        n = len(sent)
        if n == 0:
            continue

        pred_arcs = parser.parse(sent.forms, sent.upos)
        pred_head = {}
        pred_label = {}
        for head, dep, label in pred_arcs:
            pred_head[dep] = head
            pred_label[dep] = label

        for i in range(n):
            tok_id = sent.ids[i]
            gold_head = sent.heads[i]
            gold_label = sent.deprels[i]

            total_tokens += 1
            if pred_head.get(tok_id) == gold_head:
                correct_uas += 1
                if pred_label.get(tok_id) == gold_label:
                    correct_las += 1

    elapsed = time.time() - t0
    uas = correct_uas / total_tokens * 100
    las = correct_las / total_tokens * 100

    print(f"Sentences evaluated: {len(sentences)}")
    print(f"Tokens evaluated:    {total_tokens}")
    print(f"Parsing time:        {elapsed:.1f}s ({total_tokens/elapsed:.0f} tok/s)")
    print(f"UAS: {uas:.2f}%")
    print(f"LAS: {las:.2f}%")
    return uas, las


if __name__ == "__main__":
    evaluate("model.pkl", "UD_English-EWT/en_ewt-ud-dev.conllu")
