"""
Arc-standard transition system.

Configuration:
    stack:  list of token ids, stack[-1] is the top.  stack[0] is always 0
            (the artificial ROOT token).
    buffer: list of remaining token ids in sentence order, buffer[0] is
            the next word to be shifted.
    arcs:   list of (head_id, dependent_id, label) built so far.

Transitions (exactly as specified in the assignment):
    SHIFT              : move buffer[0] onto the stack.
    LEFT-ARC(label)     : stack[-1] becomes head of stack[-2]; pop stack[-2].
    RIGHT-ARC(label)    : stack[-2] becomes head of stack[-1]; pop stack[-1].

LEFT-ARC / RIGHT-ARC both require at least two items on the stack.  In
addition LEFT-ARC is not legal when stack[-2] is the ROOT (id 0), since the
root may never be assigned a head.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

ROOT_ID = 0

SHIFT = "SHIFT"
LEFT_ARC = "LEFT-ARC"
RIGHT_ARC = "RIGHT-ARC"


@dataclass
class Configuration:
    stack: List[int] = field(default_factory=lambda: [ROOT_ID])
    buffer: List[int] = field(default_factory=list)
    arcs: List[Tuple[int, int, str]] = field(default_factory=list)

    def is_terminal(self) -> bool:
        return len(self.buffer) == 0 and len(self.stack) == 1

    def can_left_arc(self) -> bool:
        return len(self.stack) >= 2 and self.stack[-2] != ROOT_ID

    def can_right_arc(self) -> bool:
        return len(self.stack) >= 2

    def can_shift(self) -> bool:
        return len(self.buffer) > 0


def initial_configuration(n_tokens: int) -> Configuration:
    """n_tokens = number of real words (ids 1..n_tokens)."""
    return Configuration(stack=[ROOT_ID], buffer=list(range(1, n_tokens + 1)), arcs=[])


def apply_transition(config: Configuration, transition: str, label: str = None) -> None:
    """Mutates `config` in place by applying the given transition."""
    if transition == SHIFT:
        if not config.can_shift():
            raise ValueError("SHIFT not legal: buffer is empty")
        config.stack.append(config.buffer.pop(0))

    elif transition == LEFT_ARC:
        if not config.can_left_arc():
            raise ValueError("LEFT-ARC not legal in this configuration")
        head = config.stack[-1]
        dependent = config.stack[-2]
        config.arcs.append((head, dependent, label))
        config.stack.pop(-2)

    elif transition == RIGHT_ARC:
        if not config.can_right_arc():
            raise ValueError("RIGHT-ARC not legal in this configuration")
        head = config.stack[-2]
        dependent = config.stack[-1]
        config.arcs.append((head, dependent, label))
        config.stack.pop(-1)

    else:
        raise ValueError(f"Unknown transition: {transition}")


def legal_transitions(config: Configuration) -> List[str]:
    legal = []
    if config.can_left_arc():
        legal.append(LEFT_ARC)
    if config.can_right_arc():
        legal.append(RIGHT_ARC)
    if config.can_shift():
        legal.append(SHIFT)
    return legal
