"""
Part 3.1: The parser itself.

Given a sentence (list of word forms + POS tags) and a trained classifier,
greedily runs the arc-standard transition system: at each step, extract
features from the current configuration, ask the classifier for the most
probable transition, mask out any transition that is not legal in the
current configuration (falling back to the next-best legal choice), apply
it, and repeat until the configuration is terminal.

Returns the predicted set of (head, dependent, label) arcs.
"""

import pickle
from typing import List, Tuple

import numpy as np

from features import extract_features
from train import decode_label
from transition_system import (
    Configuration,
    initial_configuration,
    apply_transition,
    legal_transitions,
    SHIFT,
    LEFT_ARC,
    RIGHT_ARC,
)


class DependencyParser:
    def __init__(self, model_path: str = "model.pkl"):
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        self.vectorizer = saved["vectorizer"]
        self.classifier = saved["classifier"]
        self.classes = self.classifier.classes_

    def _predict_ranked(self, feats: dict) -> List[str]:
        """Return class labels (encoded transition[:label]) ranked best-first."""
        X = self.vectorizer.transform([feats])
        probs = self.classifier.predict_proba(X)[0]
        order = np.argsort(-probs)
        return [self.classes[i] for i in order]

    def parse(self, words: List[str], upos: List[str]) -> List[Tuple[int, int, str]]:
        """
        words: list of surface forms, 1 per token
        upos:  list of UPOS tags, same length as words
        Returns list of (head_id, dependent_id, label), 1-indexed ids, 0=ROOT.
        """
        n = len(words)
        pos_by_id = {i + 1: upos[i] for i in range(n)}
        config = initial_configuration(n)

        steps = 0
        max_steps = 4 * n + 5
        while not config.is_terminal() and steps < max_steps:
            steps += 1
            feats = extract_features(config, pos_by_id)
            ranked_classes = self._predict_ranked(feats)
            legal = set(legal_transitions(config))

            chosen_transition, chosen_label = None, None
            for class_label in ranked_classes:
                transition, label = decode_label(class_label)
                if transition in legal:
                    chosen_transition, chosen_label = transition, label
                    break

            if chosen_transition is None:
                # Should not happen (SHIFT/RIGHT-ARC always legal unless
                # both buffer and multi-element stack are exhausted), but
                # fall back safely just in case.
                if SHIFT in legal:
                    chosen_transition, chosen_label = SHIFT, None
                elif RIGHT_ARC in legal:
                    chosen_transition, chosen_label = RIGHT_ARC, "dep"
                else:
                    chosen_transition, chosen_label = LEFT_ARC, "dep"

            apply_transition(config, chosen_transition, chosen_label)

        # Safety net: if we hit max_steps without finishing (shouldn't
        # normally happen), attach any un-headed tokens to root so the
        # output is still well-formed.
        headed = {dep for (_h, dep, _l) in config.arcs}
        for tok_id in range(1, n + 1):
            if tok_id not in headed:
                config.arcs.append((0, tok_id, "dep"))

        return config.arcs


if __name__ == "__main__":
    parser = DependencyParser("model.pkl")

    examples = [
        (
            ["The", "cat", "sat", "on", "the", "mat", "."],
            ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN", "PUNCT"],
        ),
        (
            ["She", "eats", "a", "green", "salad", "."],
            ["PRON", "VERB", "DET", "ADJ", "NOUN", "PUNCT"],
        ),
        (
            ["I", "saw", "the", "man", "with", "a", "telescope", "."],
            ["PRON", "VERB", "DET", "NOUN", "ADP", "DET", "NOUN", "PUNCT"],
        ),
    ]

    for words, upos in examples:
        arcs = parser.parse(words, upos)
        arcs_sorted = sorted(arcs, key=lambda a: a[1])
        print("\nSentence:", " ".join(words))
        for head, dep, label in arcs_sorted:
            head_word = "ROOT" if head == 0 else words[head - 1]
            print(f"  {words[dep-1]:<10} <-- {label:<10} -- {head_word}")
