"""
pcfg_parser.py
--------------
Part 2: PCFG Constituency Parser and Tagset Reconciliation
- Induces a PCFG from the Penn Treebank sample (nltk.corpus.treebank)
- Converts rules to Chomsky Normal Form (CNF)
- Implements a Viterbi / CKY most-probable-parse algorithm
- Gracefully handles unparseable sentences without crashing
- Reconciles Question 1 Brown POS tags with Penn Treebank tags
"""

import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional, Any, Set
import nltk
from nltk import Tree, Nonterminal, PCFG, ProbabilisticProduction


# Comprehensive Brown-to-PTB tag reconciliation mapping
BROWN_TO_PTB = {
    'AT': 'DT', 'DT': 'DT', 'DTI': 'DT', 'DTS': 'DT', 'DTX': 'DT', 'AP': 'JJ', 'ABL': 'PDT', 'ABN': 'PDT',
    'NN': 'NN', 'NNS': 'NNS', 'NP': 'NNP', 'NPS': 'NNPS', 'NR': 'NNP', 'NRS': 'NNPS', 'NC': 'NN',
    'JJ': 'JJ', 'JJR': 'JJR', 'JJS': 'JJS', 'JJT': 'JJS',
    'RB': 'RB', 'RBR': 'RBR', 'RBT': 'RBS', 'RN': 'RB', 'RP': 'RP', 'QL': 'RB', 'QLP': 'RB', 'WRB': 'WRB',
    'VB': 'VB', 'VBD': 'VBD', 'VBG': 'VBG', 'VBN': 'VBN', 'VBZ': 'VBZ', 'VBP': 'VBP',
    'BE': 'VB', 'BED': 'VBD', 'BEDZ': 'VBD', 'BEG': 'VBG', 'BEM': 'VBP', 'BEN': 'VBN', 'BER': 'VBP', 'BEZ': 'VBZ',
    'HV': 'VB', 'HVD': 'VBD', 'HVG': 'VBG', 'HVN': 'VBN', 'HVZ': 'VBZ', 'HVP': 'VBP',
    'DO': 'VB', 'DOD': 'VBD', 'DOZ': 'VBZ', 'MD': 'MD',
    'IN': 'IN', 'CS': 'IN', 'CC': 'CC', 'CD': 'CD', 'OD': 'JJ',
    'PN': 'PRP', 'PP$': 'PRP$', 'PP$$': 'PRP$', 'PPL': 'PRP', 'PPLS': 'PRP', 'PPO': 'PRP', 'PPS': 'PRP', 'PPSS': 'PRP',
    'WP$': 'WP$', 'WPO': 'WP', 'WPS': 'WP', 'WDT': 'WDT', 'EX': 'EX', 'TO': 'TO', 'UH': 'UH',
    '.': '.', ',': ',', '(': '(', ')': ')', '--': ':', ':': ':', '\'\'': '\'\'', '``': '``',
    # Universal POS fallbacks if Q1 returns UPOS tags
    'NOUN': 'NN', 'VERB': 'VB', 'ADJ': 'JJ', 'ADV': 'RB', 'PRON': 'PRP', 'DET': 'DT',
    'ADP': 'IN', 'NUM': 'CD', 'CONJ': 'CC', 'PRT': 'RP', 'PUNCT': '.', 'X': 'FW'
}


def reconcile_tag(tag: str) -> str:
    """
    Reconciles a POS tag from Question 1 (Brown corpus / UPOS) to Penn Treebank tagset.
    Strips morphological suffixes (e.g. 'NOUN-Masc-Sg' -> 'NOUN') and maps to PTB.
    """
    if not tag:
        return 'NN'
    clean = tag.split('-')[0].split('+')[0].split('$')[0] + ('$' if '$' in tag else '')
    clean_upper = clean.upper()

    if tag in BROWN_TO_PTB:
        return BROWN_TO_PTB[tag]
    if clean_upper in BROWN_TO_PTB:
        return BROWN_TO_PTB[clean_upper]
    return clean_upper


def binarize_tree(tree: Tree) -> Tree:
    """
    Binarize an NLTK tree into CNF (Chomsky Normal Form).
    Standard Chomsky-style left-factoring:
    A -> B C D  ==>  A -> B @A_1, @A_1 -> C D
    """
    if isinstance(tree, str):
        return tree
    if not isinstance(tree, Tree):
        return tree

    # Clean nonterminal labels (strip function tags e.g. NP-SBJ -> NP)
    raw_label = tree.label()
    clean_label = raw_label.split('-')[0].split('=')[0]
    if not clean_label:
        clean_label = 'X'

    # Recurse on children
    binarized_children = [binarize_tree(child) for child in tree]

    # Leaf or unary
    if len(binarized_children) <= 2:
        return Tree(clean_label, binarized_children)

    # N-ary rule binarization
    curr = binarized_children[-1]
    for i in range(len(binarized_children) - 2, 0, -1):
        intermediate_label = f"@{clean_label}_{i}"
        curr = Tree(intermediate_label, [binarized_children[i], curr])
    
    return Tree(clean_label, [binarized_children[0], curr])


class CKYConstituencyParser:
    """
    Viterbi / CKY Probabilistic Context-Free Grammar (PCFG) Parser.
    Learns rule probabilities from treebank trees and finds most probable parse tree.
    """

    def __init__(self, start_symbol: str = "S"):
        self.start_symbol = start_symbol
        # binary_rules: dict (B, C) -> list of (A, log_prob)
        self.binary_rules = defaultdict(list)
        # unary_rules: dict B -> list of (A, log_prob)
        self.unary_rules = defaultdict(list)
        # lexical_rules: dict word.lower() -> list of (A, log_prob)
        self.lexical_rules = defaultdict(list)
        # pos_rules: dict PTB_tag -> list of (A, log_prob)
        self.pos_rules = defaultdict(list)
        self.vocab: Set[str] = set()
        self.nonterminals: Set[str] = set()

    def train_from_treebank(self, max_sents: int = 1000):
        """
        Induces PCFG from Penn Treebank sample in NLTK.
        """
        try:
            from nltk.corpus import treebank
            trees = treebank.parsed_sents()[:max_sents]
        except Exception:
            nltk.download('treebank', quiet=True)
            from nltk.corpus import treebank
            trees = treebank.parsed_sents()[:max_sents]

        lhs_counts = Counter()
        binary_counts = Counter()
        unary_counts = Counter()
        lexical_counts = Counter()
        pos_counts = Counter()

        for raw_tree in trees:
            if not raw_tree or len(raw_tree) == 0:
                continue
            cnf_tree = binarize_tree(raw_tree)
            self._collect_productions(cnf_tree, lhs_counts, binary_counts, unary_counts, lexical_counts, pos_counts)

        # Calculate log probabilities
        for (A, B, C), count in binary_counts.items():
            prob = count / lhs_counts[A]
            self.binary_rules[(B, C)].append((A, math.log(max(prob, 1e-12))))
            self.nonterminals.add(A)

        for (A, B), count in unary_counts.items():
            prob = count / lhs_counts[A]
            self.unary_rules[B].append((A, math.log(max(prob, 1e-12))))
            self.nonterminals.add(A)

        for (A, w), count in lexical_counts.items():
            prob = count / lhs_counts[A]
            self.lexical_rules[w].append((A, math.log(max(prob, 1e-12))))
            self.vocab.add(w)

        for (A, tag), count in pos_counts.items():
            prob = count / lhs_counts[A]
            self.pos_rules[tag].append((A, math.log(max(prob, 1e-12))))

    def _collect_productions(self, tree: Tree, lhs_counts, binary_counts, unary_counts, lexical_counts, pos_counts):
        if isinstance(tree, str):
            return

        A = tree.label()
        lhs_counts[A] += 1
        children = [c for c in tree]

        if len(children) == 1 and isinstance(children[0], str):
            # Lexical pre-terminal
            word = children[0].lower()
            lexical_counts[(A, word)] += 1
            pos_counts[(A, A)] += 1
        elif len(children) == 1 and isinstance(children[0], Tree):
            # Unary non-terminal
            B = children[0].label()
            unary_counts[(A, B)] += 1
            self._collect_productions(children[0], lhs_counts, binary_counts, unary_counts, lexical_counts, pos_counts)
        elif len(children) == 2:
            # Binary non-terminal
            B = children[0].label() if isinstance(children[0], Tree) else children[0]
            C = children[1].label() if isinstance(children[1], Tree) else children[1]
            binary_counts[(A, B, C)] += 1
            if isinstance(children[0], Tree):
                self._collect_productions(children[0], lhs_counts, binary_counts, unary_counts, lexical_counts, pos_counts)
            if isinstance(children[1], Tree):
                self._collect_productions(children[1], lhs_counts, binary_counts, unary_counts, lexical_counts, pos_counts)

    def parse(self, words: List[str], tags: Optional[List[str]] = None) -> Tuple[Optional[Tree], float, str]:
        """
        Viterbi CKY parse for a list of words and optional reconciled POS tags.
        Returns:
            (best_tree, log_probability, status_message)
        """
        n = len(words)
        if n == 0:
            return (None, -float('inf'), "unparseable: empty input")

        reconciled_tags = [reconcile_tag(t) for t in tags] if tags else [None] * n

        # Chart: chart[i][j][A] = (log_prob, backpointer)
        chart = [[defaultdict(lambda: (-float('inf'), None)) for _ in range(n + 1)] for _ in range(n + 1)]

        # 1. Fill leaf cells (lexical and POS fallback)
        for i in range(n):
            w = words[i].lower()
            tag = reconciled_tags[i]
            matched = False

            # Known word lexical rules
            if w in self.lexical_rules:
                for A, log_p in self.lexical_rules[w]:
                    if log_p > chart[i][i + 1][A][0]:
                        chart[i][i + 1][A] = (log_p, words[i])
                        matched = True

            # If unknown or fallback needed, use reconciled POS tag
            if not matched and tag:
                if tag in self.pos_rules:
                    for A, log_p in self.pos_rules[tag]:
                        # Slight smoothing penalty for unseen lexical token
                        smoothed_p = log_p - 1.5
                        if smoothed_p > chart[i][i + 1][A][0]:
                            chart[i][i + 1][A] = (smoothed_p, words[i])
                            matched = True
                else:
                    # Treat tag as direct preterminal
                    chart[i][i + 1][tag] = (-2.0, words[i])
                    matched = True

            # General unknown fallback (Noun preterminal)
            if not matched:
                chart[i][i + 1]['NN'] = (-5.0, words[i])

            # Apply unary closures on span length 1
            self._apply_unary_closure(chart[i][i + 1])

        # 2. Dynamic programming for span lengths 2..n
        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    left_cell = chart[i][k]
                    right_cell = chart[k][j]

                    for B, (lp_b, bp_b) in left_cell.items():
                        for C, (lp_c, bp_c) in right_cell.items():
                            if (B, C) in self.binary_rules:
                                for A, rule_lp in self.binary_rules[(B, C)]:
                                    cand_lp = rule_lp + lp_b + lp_c
                                    if cand_lp > chart[i][j][A][0]:
                                        chart[i][j][A] = (cand_lp, (k, B, C))

                # Apply unary closures on span [i, j]
                self._apply_unary_closure(chart[i][j])

        # 3. Retrieve root parse
        best_tree = None
        best_lp = -float('inf')

        # Prefer S / SINV / SQ as root
        preferred_roots = [self.start_symbol, "SINV", "SQ", "FRAG", "NP"]
        for root in preferred_roots:
            if root in chart[0][n]:
                lp, bp = chart[0][n][root]
                if lp > best_lp:
                    best_lp = lp
                    best_tree = self._build_tree(root, 0, n, chart)

        if best_tree is None:
            # Check any non-intermediate nonterminal in top cell
            for root, (lp, bp) in chart[0][n].items():
                if not root.startswith("@") and lp > best_lp:
                    best_lp = lp
                    best_tree = self._build_tree(root, 0, n, chart)

        if best_tree is None:
            return (None, -float('inf'), "unparseable")

        return (best_tree, best_lp, "parsed")

    def _apply_unary_closure(self, cell, max_depth: int = 2):
        for _ in range(max_depth):
            additions = []
            for B, (lp_b, bp_b) in list(cell.items()):
                if B in self.unary_rules:
                    for A, rule_lp in self.unary_rules[B]:
                        cand_lp = rule_lp + lp_b
                        if cand_lp > cell[A][0]:
                            additions.append((A, cand_lp, B))
            if not additions:
                break
            for A, cand_lp, B in additions:
                cell[A] = (cand_lp, ('UNARY', B))

    def _build_tree(self, symbol: str, i: int, j: int, chart) -> Tree:
        lp, bp = chart[i][j][symbol]

        if j - i == 1 and isinstance(bp, str):
            return Tree(symbol, [bp])

        if isinstance(bp, tuple) and bp[0] == 'UNARY':
            child_sym = bp[1]
            child_tree = self._build_tree(child_sym, i, j, chart)
            return Tree(symbol, [child_tree])

        if isinstance(bp, tuple) and len(bp) == 3:
            k, B, C = bp
            left_child = self._build_tree(B, i, k, chart)
            right_child = self._build_tree(C, k, j, chart)
            
            # De-binarize intermediate nodes (@S_1 etc)
            children = []
            if symbol.startswith("@"):
                children.extend(left_child if isinstance(left_child, list) else [left_child])
                children.extend(right_child if isinstance(right_child, list) else [right_child])
                return children
            else:
                for c in [left_child, right_child]:
                    if isinstance(c, list):
                        children.extend(c)
                    else:
                        children.append(c)
                return Tree(symbol, children)

        return Tree(symbol, ["..."])
