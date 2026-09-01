"""
data_loader.py
--------------
Data loading and preprocessing utilities for:
1. English: NLTK Brown Corpus (80% train / 20% test split)
   - Standard Universal POS tags & Penn Treebank style tags
   - Morphology-aware tags (Number, Tense, Aspect)
2. Spanish: Universal Dependencies Spanish-GSD (UD_Spanish-GSD)
   - Train, Dev, Test splits
   - Standard Universal POS (UPOS) tags
   - Morphology-aware tags (UPOS + Gender + Number + Person)
"""

import os
import re
import random
from typing import List, Tuple, Dict, Optional

# Deterministic mapping from Brown tags to Universal POS and PTB-like tags
BROWN_TO_UNIVERSAL = {
    'AT': 'DET', 'DT': 'DET', 'DTI': 'DET', 'DTS': 'DET', 'DTX': 'DET', 'AP': 'DET', 'ABL': 'DET', 'ABN': 'DET',
    'NN': 'NOUN', 'NNS': 'NOUN', 'NP': 'NOUN', 'NPS': 'NOUN', 'NR': 'NOUN', 'NRS': 'NOUN', 'NC': 'NOUN',
    'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ', 'JJT': 'ADJ',
    'RB': 'ADV', 'RBR': 'ADV', 'RBT': 'ADV', 'RN': 'ADV', 'RP': 'PRT', 'QL': 'ADV', 'QLP': 'ADV', 'WRB': 'ADV',
    'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBZ': 'VERB', 'VBP': 'VERB',
    'BE': 'VERB', 'BED': 'VERB', 'BEDZ': 'VERB', 'BEG': 'VERB', 'BEM': 'VERB', 'BEN': 'VERB', 'BER': 'VERB', 'BEZ': 'VERB',
    'HV': 'VERB', 'HVD': 'VERB', 'HVG': 'VERB', 'HVN': 'VERB', 'HVZ': 'VERB', 'HVP': 'VERB',
    'DO': 'VERB', 'DOD': 'VERB', 'DOZ': 'VERB', 'MD': 'VERB',
    'IN': 'ADP', 'CS': 'ADP',
    'CC': 'CONJ',
    'CD': 'NUM', 'OD': 'NUM',
    'PN': 'PRON', 'PP$': 'PRON', 'PP$$': 'PRON', 'PPL': 'PRON', 'PPLS': 'PRON', 'PPO': 'PRON', 'PPS': 'PRON', 'PPSS': 'PRON',
    'WP$': 'PRON', 'WPO': 'PRON', 'WPS': 'PRON', 'WDT': 'DET', 'EX': 'PRON',
    'TO': 'PRT',
    'UH': 'X',
    '.': '.', ',': '.', '(': '.', ')': '.', '--': '.', ':': '.', '\'\'': '.', '``': '.', '*': 'X',
}

BROWN_TO_PTB = {
    'AT': 'DT', 'NN': 'NN', 'NNS': 'NNS', 'NP': 'NNP', 'NPS': 'NNPS', 'NR': 'NNP',
    'JJ': 'JJ', 'JJR': 'JJR', 'JJS': 'JJS', 'JJT': 'JJS',
    'RB': 'RB', 'RBR': 'RBR', 'RBT': 'RBS', 'RP': 'RP', 'WRB': 'WRB',
    'VB': 'VB', 'VBD': 'VBD', 'VBG': 'VBG', 'VBN': 'VBN', 'VBZ': 'VBZ', 'VBP': 'VBP',
    'BE': 'VB', 'BED': 'VBD', 'BEDZ': 'VBD', 'BEG': 'VBG', 'BEM': 'VBP', 'BEN': 'VBN', 'BER': 'VBP', 'BEZ': 'VBZ',
    'HV': 'VB', 'HVD': 'VBD', 'HVG': 'VBG', 'HVN': 'VBN', 'HVZ': 'VBZ', 'HVP': 'VBP',
    'DO': 'VB', 'DOD': 'VBD', 'DOZ': 'VBZ', 'MD': 'MD',
    'IN': 'IN', 'CS': 'IN', 'CC': 'CC', 'CD': 'CD', 'OD': 'JJ',
    'PN': 'NN', 'PP$': 'PRP$', 'PPL': 'PRP', 'PPO': 'PRP', 'PPS': 'PRP', 'PPSS': 'PRP',
    'WP$': 'WP$', 'WPO': 'WP', 'WPS': 'WP', 'WDT': 'WDT', 'EX': 'EX', 'TO': 'TO', 'UH': 'UH',
    '.': '.', ',': ',', '(': '(', ')': ')', '--': ':', ':': ':', '\'\'': '\'\'', '``': '``'
}


def map_brown_tag_universal(raw_tag: str) -> str:
    """Map raw Brown tag to Universal tagset."""
    clean_tag = raw_tag.split('-')[0].split('+')[0].split('$')[0] + ('$' if '$' in raw_tag else '')
    if raw_tag in BROWN_TO_UNIVERSAL:
        return BROWN_TO_UNIVERSAL[raw_tag]
    if clean_tag in BROWN_TO_UNIVERSAL:
        return BROWN_TO_UNIVERSAL[clean_tag]
    base = re.sub(r'[^A-Za-z]', '', raw_tag).upper()
    if base in BROWN_TO_UNIVERSAL:
        return BROWN_TO_UNIVERSAL[base]
    if base.startswith('NN') or base.startswith('NP'):
        return 'NOUN'
    if base.startswith('VB') or base.startswith('BE') or base.startswith('HV') or base.startswith('DO'):
        return 'VERB'
    if base.startswith('JJ'):
        return 'ADJ'
    if base.startswith('RB'):
        return 'ADV'
    if base.startswith('PP') or base.startswith('PN') or base.startswith('WP'):
        return 'PRON'
    if base.startswith('AT') or base.startswith('DT'):
        return 'DET'
    if base.startswith('IN') or base.startswith('CS'):
        return 'ADP'
    if base.startswith('CC'):
        return 'CONJ'
    if base.startswith('CD') or base.startswith('OD'):
        return 'NUM'
    return 'X'


def map_brown_tag_ptb(raw_tag: str) -> str:
    """Map raw Brown tag to Penn Treebank style tag."""
    clean_tag = raw_tag.split('-')[0].split('+')[0]
    if raw_tag in BROWN_TO_PTB:
        return BROWN_TO_PTB[raw_tag]
    if clean_tag in BROWN_TO_PTB:
        return BROWN_TO_PTB[clean_tag]
    base = re.sub(r'[^A-Za-z]', '', clean_tag).upper()
    if base in BROWN_TO_PTB:
        return BROWN_TO_PTB[base]
    return map_brown_tag_universal(raw_tag)


def brown_tag_to_morph(word: str, raw_tag: str) -> str:
    """Extract morphology-aware tag for English from Brown tag."""
    base_upos = map_brown_tag_universal(raw_tag)
    tag_clean = raw_tag.upper()

    # Nouns
    if base_upos == 'NOUN':
        if 'NNS' in tag_clean or 'NPS' in tag_clean or 'NRS' in tag_clean:
            return 'NOUN-Plur'
        return 'NOUN-Sing'
    
    # Verbs
    if base_upos == 'VERB':
        if 'VBZ' in tag_clean or 'BEZ' in tag_clean or 'HVZ' in tag_clean or 'DOZ' in tag_clean:
            return 'VERB-Sing3'
        if 'VBD' in tag_clean or 'BED' in tag_clean or 'HVD' in tag_clean or 'DOD' in tag_clean:
            return 'VERB-Past'
        if 'VBG' in tag_clean or 'BEG' in tag_clean or 'HVG' in tag_clean:
            return 'VERB-Gerund'
        if 'VBN' in tag_clean or 'BEN' in tag_clean or 'HVN' in tag_clean:
            return 'VERB-Part'
        if 'MD' in tag_clean:
            return 'VERB-Modal'
        return 'VERB-Base'
    
    # Adjectives
    if base_upos == 'ADJ':
        if 'JJR' in tag_clean:
            return 'ADJ-Comp'
        if 'JJS' in tag_clean or 'JJT' in tag_clean:
            return 'ADJ-Sup'
        return 'ADJ-Pos'
    
    # Pronouns
    if base_upos == 'PRON':
        if 'PPS' in tag_clean:
            return 'PRON-3Sing'
        if 'PPSS' in tag_clean or 'PPLS' in tag_clean:
            return 'PRON-Plur'
        if 'PP$' in tag_clean or 'PP$$' in tag_clean:
            return 'PRON-Poss'
        return 'PRON-General'
        
    # Determiners
    if base_upos == 'DET':
        if word.lower() in ('the', 'this', 'that', 'a', 'an'):
            return 'DET-Sing'
        if word.lower() in ('these', 'those', 'all', 'many', 'few', 'both'):
            return 'DET-Plur'
        return 'DET-General'
        
    return base_upos


def load_english_brown(split_ratio: float = 0.8, seed: int = 42) -> Tuple[List[List[Tuple[str, str, str]]], List[List[Tuple[str, str, str]]]]:
    """
    Load Brown corpus and return (train_sentences, test_sentences).
    Each sentence is a list of tuples: (word, universal_pos, morph_pos).
    Words are lowercased and purely alphabetic tokens are preserved.
    """
    import nltk
    from nltk.corpus import brown

    raw_sents = brown.tagged_sents()
    cleaned_sents = []

    for sent in raw_sents:
        cleaned_sent = []
        for word, raw_tag in sent:
            w = word.strip().lower()
            if not w or not w.isalnum():
                continue
            upos = map_brown_tag_universal(raw_tag)
            morph_tag = brown_tag_to_morph(w, raw_tag)
            cleaned_sent.append((w, upos, morph_tag))
        if len(cleaned_sent) >= 3:
            cleaned_sents.append(cleaned_sent)

    rng = random.Random(seed)
    rng.shuffle(cleaned_sents)

    split_idx = int(len(cleaned_sents) * split_ratio)
    train_sents = cleaned_sents[:split_idx]
    test_sents = cleaned_sents[split_idx:]
    return train_sents, test_sents


def parse_spanish_feats(feats_str: str, upos: str) -> str:
    """
    Parse UD CoNLL-U FEATS column and build morphology-aware tag:
    e.g., 'NOUN-Fem-Sing', 'ADJ-Masc-Plur', 'VERB-Fin-Sing-3'
    """
    if not feats_str or feats_str == '_':
        return upos

    feat_dict = {}
    for item in feats_str.split('|'):
        if '=' in item:
            k, v = item.split('=', 1)
            feat_dict[k] = v

    gender = feat_dict.get('Gender', '')
    number = feat_dict.get('Number', '')
    person = feat_dict.get('Person', '')
    verb_form = feat_dict.get('VerbForm', '')

    parts = [upos]
    if gender:
        parts.append(gender)
    if number:
        parts.append(number)
    if person and upos in ('VERB', 'AUX', 'PRON'):
        parts.append(f"P{person}")
    elif verb_form and upos in ('VERB', 'AUX'):
        parts.append(verb_form)

    if len(parts) == 1:
        return upos
    return "-".join(parts)


def load_spanish_conllu(file_path: str) -> List[List[Tuple[str, str, str]]]:
    """
    Load CoNLL-U format file and return list of sentences:
    Each sentence is [(word, upos, morph_tag), ...]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CoNLL-U file not found: {file_path}")

    sentences = []
    current_sent = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sent:
                    if len(current_sent) >= 2:
                        sentences.append(current_sent)
                    current_sent = []
                continue
            if line.startswith('#'):
                continue

            fields = line.split('\t')
            if len(fields) < 6:
                continue

            tok_id, form, lemma, upos, xpos, feats = fields[:6]
            # Skip multiword token range lines (e.g. 2-3)
            if '-' in tok_id or '.' in tok_id:
                continue

            w = form.strip().lower()
            if not w or not w.isalnum():
                continue

            morph_tag = parse_spanish_feats(feats, upos)
            current_sent.append((w, upos, morph_tag))

    if current_sent and len(current_sent) >= 2:
        sentences.append(current_sent)

    return sentences


def load_spanish_ud(data_dir: Optional[str] = None) -> Tuple[List[List[Tuple[str, str, str]]], List[List[Tuple[str, str, str]]], List[List[Tuple[str, str, str]]]]:
    """
    Load Spanish UD-GSD dataset (train, dev, test).
    """
    if data_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data", "UD_Spanish-GSD")

    train_file = os.path.join(data_dir, "es_gsd-ud-train.conllu")
    dev_file = os.path.join(data_dir, "es_gsd-ud-dev.conllu")
    test_file = os.path.join(data_dir, "es_gsd-ud-test.conllu")

    train_sents = load_spanish_conllu(train_file)
    dev_sents = load_spanish_conllu(dev_file)
    test_sents = load_spanish_conllu(test_file)

    return train_sents, dev_sents, test_sents


def sentence_to_unspaced(sentence: List[Tuple[str, str, str]]) -> Tuple[str, List[str], List[str], List[str], List[Tuple[int, int]]]:
    """
    Convert a tagged sentence to an unspaced string, returning:
    - unspaced_string: 'mispadrespuedenviajar'
    - words: ['mis', 'padres', 'pueden', 'viajar']
    - upos_tags: ['DET', 'NOUN', 'VERB', 'VERB']
    - morph_tags: ['DET-Plur', 'NOUN-Masc-Plur', ...]
    - spans: [(0, 3), (3, 9), (9, 15), (15, 21)]
    """
    words = [w for w, _, _ in sentence]
    upos_tags = [u for _, u, _ in sentence]
    morph_tags = [m for _, _, m in sentence]

    unspaced_chars = []
    spans = []
    curr_pos = 0
    for w in words:
        start = curr_pos
        end = curr_pos + len(w)
        spans.append((start, end))
        unspaced_chars.append(w)
        curr_pos = end

    unspaced_string = "".join(unspaced_chars)
    return unspaced_string, words, upos_tags, morph_tags, spans
