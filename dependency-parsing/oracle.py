"""
Part 1.2: Oracle simulator.

Given a gold-parsed sentence, replays the arc-standard transition system,
using the gold heads/labels to pick the *correct* transition at each step
(a static oracle, Nivre 2004). This produces the (configuration,
transition, label) training instances used in Part 2.

Note: the arc-standard system (as given) can only produce *projective*
trees. English-EWT contains a small number of non-projective sentences;
for those the oracle will reach a configuration where no gold-consistent
move is available. We detect this (the oracle gets "stuck": SHIFT is
attempted on an empty buffer, or the reconstructed arcs don't match the
gold tree) and simply skip that sentence when building training data,
which is the standard approach for training a projective parser.
"""

from typing import List, Optional, Tuple

from conllu_io import Sentence
from transition_system import (
    Configuration,
    initial_configuration,
    apply_transition,
    SHIFT,
    LEFT_ARC,
    RIGHT_ARC,
    ROOT_ID,
)


class OracleFailure(Exception):
    """Raised when the sentence's gold tree is not reachable by arc-standard
    (i.e. it is non-projective), so no oracle sequence exists."""


def _oracle_move(config: Configuration, gold_head, gold_children_remaining) -> Tuple[str, Optional[str], object]:
    """Decide the next gold transition for the current configuration.

    gold_head: dict token_id -> gold head id
    gold_children_remaining: dict token_id -> set of gold-children ids not
        yet attached (i.e. not yet popped as a dependent). Used to decide
        RIGHT-ARC only fires once a word has collected all its children.
    Returns (transition_name, label_or_None).
    """
    stack = config.stack
    buffer = config.buffer

    if len(stack) >= 2:
        s1 = stack[-1]
        s2 = stack[-2]

        if s2 != ROOT_ID and gold_head.get(s2) == s1:
            return LEFT_ARC

        if gold_head.get(s1) == s2 and len(gold_children_remaining.get(s1, ())) == 0:
            return RIGHT_ARC

    if len(buffer) > 0:
        return SHIFT

    raise OracleFailure("No legal oracle move (sentence is likely non-projective)")


def run_oracle(sentence: Sentence) -> List[Tuple[Configuration, str, Optional[str]]]:
    """Replay arc-standard parsing of `sentence` using the gold tree.

    Returns a list of (config_snapshot, transition, label) triples, one per
    step, where config_snapshot is a *copy* of the configuration BEFORE the
    transition was applied (this is what the feature extractor will see at
    parse time).

    Raises OracleFailure if the gold tree is not projective (cannot be
    produced by the arc-standard system).
    """
    n = len(sentence)
    gold_head = {sentence.ids[i]: sentence.heads[i] for i in range(n)}
    gold_label = {sentence.ids[i]: sentence.deprels[i] for i in range(n)}

    # children (in gold tree) of each token id, used to know when a word's
    # subtree is "complete" so it can be RIGHT-ARC'd off the stack.
    gold_children_remaining = {}
    for i in range(n):
        tok_id = sentence.ids[i]
        head = sentence.heads[i]
        gold_children_remaining.setdefault(head, set()).add(tok_id)

    config = initial_configuration(n)
    instances = []

    max_steps = 4 * n + 2  # arc-standard always finishes in exactly 2n steps
    steps = 0
    while not config.is_terminal():
        steps += 1
        if steps > max_steps:
            raise OracleFailure("Oracle did not terminate (unexpected)")

        transition = _oracle_move(config, gold_head, gold_children_remaining)

        if transition == LEFT_ARC:
            dependent = config.stack[-2]
            label = gold_label[dependent]
        elif transition == RIGHT_ARC:
            dependent = config.stack[-1]
            label = gold_label[dependent]
        else:
            label = None

        # snapshot BEFORE applying (deep-ish copy of the mutable lists)
        snapshot = Configuration(
            stack=list(config.stack),
            buffer=list(config.buffer),
            arcs=list(config.arcs),
        )
        instances.append((snapshot, transition, label))

        apply_transition(config, transition, label)

        if transition in (LEFT_ARC, RIGHT_ARC):
            dependent = instances[-1][0].stack[-2] if transition == LEFT_ARC else instances[-1][0].stack[-1]
            gold_children_remaining.setdefault(gold_head[dependent], set()).discard(dependent)

    # Sanity check: the arcs produced must exactly equal the gold tree.
    predicted_head = {dep: head for (head, dep, _lbl) in config.arcs}
    for i in range(n):
        tok_id = sentence.ids[i]
        if predicted_head.get(tok_id) != sentence.heads[i]:
            raise OracleFailure(
                f"Oracle result does not match gold tree for token {tok_id} "
                f"(got head {predicted_head.get(tok_id)}, expected {sentence.heads[i]})"
            )

    return instances


if __name__ == "__main__":
    from conllu_io import read_conllu

    sents = read_conllu("UD_English-EWT/en_ewt-ud-dev.conllu")
    ok, fail = 0, 0
    total_instances = 0
    for s in sents[:200]:
        try:
            inst = run_oracle(s)
            ok += 1
            total_instances += len(inst)
        except OracleFailure:
            fail += 1
    print(f"ok={ok} fail(non-projective)={fail} total_instances={total_instances}")

    # show one worked example
    s = sents[0]
    inst = run_oracle(s)
    print("\nSentence:", " ".join(s.forms))
    for (cfg, trans, label) in inst:
        stack_words = [s.forms[s.ids.index(i)] if i != 0 else "ROOT" for i in cfg.stack]
        buf_words = [s.forms[s.ids.index(i)] for i in cfg.buffer[:3]]
        print(f"  stack={stack_words}  buffer={buf_words}...  ->  {trans} {label or ''}")
