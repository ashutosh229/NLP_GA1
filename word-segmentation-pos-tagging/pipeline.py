"""
pipeline.py
-----------
Unified End-to-End Pipeline for Word Segmentation and POS Tagging:
- Proposed Model: Trigram LM + DP (Viterbi) + Trigram HMM POS Tagger
- Morphology-Aware Extension: Trigram LM + DP + Agreement-Aware POS Tagger
- Baseline Pipeline: Greedy Longest-Match + Most-Frequent-Tag Tagger
"""

from typing import List, Tuple, Dict, Optional
try:
    from .segmentation import TrigramLanguageModel, TrigramDPSegmenter, GreedyLongestMatchSegmenter
    from .pos_tagger import TrigramHMMPOSTagger, MostFrequentTagTagger
except (ImportError, ValueError):
    from segmentation import TrigramLanguageModel, TrigramDPSegmenter, GreedyLongestMatchSegmenter
    from pos_tagger import TrigramHMMPOSTagger, MostFrequentTagTagger


class NLPJointPipeline:
    """
    End-to-End Word Segmentation and POS Tagging Pipeline.
    Supports both Standard and Morphology-Aware POS tagging.
    """

    def __init__(self, language: str = "english"):
        self.language = language.lower()
        self.lm = TrigramLanguageModel()
        self.dp_segmenter: Optional[TrigramDPSegmenter] = None
        self.greedy_segmenter: Optional[GreedyLongestMatchSegmenter] = None

        self.hmm_standard = TrigramHMMPOSTagger()
        self.hmm_morph = TrigramHMMPOSTagger()

        self.baseline_tagger_standard = MostFrequentTagTagger()
        self.baseline_tagger_morph = MostFrequentTagTagger()

    def train(self, train_sentences: List[List[Tuple[str, str, str]]]):
        """
        Train all models on a list of sentences: [(word, upos_tag, morph_tag), ...]
        """
        raw_words_corpus = [[w for w, _, _ in sent] for sent in train_sentences]
        standard_tagged_corpus = [[(w, u) for w, u, _ in sent] for sent in train_sentences]
        morph_tagged_corpus = [[(w, m) for w, _, m in sent] for sent in train_sentences]

        # 1. Train Language Model & DP Segmenter
        self.lm.train(raw_words_corpus)
        self.dp_segmenter = TrigramDPSegmenter(self.lm)
        self.greedy_segmenter = GreedyLongestMatchSegmenter(self.lm.vocab)

        # 2. Train Standard POS Taggers (Trigram HMM & Baseline MFT)
        self.hmm_standard.train(standard_tagged_corpus)
        self.baseline_tagger_standard.train(standard_tagged_corpus)

        # 3. Train Morphology-Aware POS Taggers
        self.hmm_morph.train(morph_tagged_corpus)
        self.baseline_tagger_morph.train(morph_tagged_corpus)

    def process_proposed(self, unspaced_text: str, morphology_aware: bool = False) -> List[Tuple[str, str]]:
        """
        Process unspaced string using proposed Trigram DP + Trigram HMM model.
        """
        if self.dp_segmenter is None:
            raise RuntimeError("Pipeline must be trained before inference.")

        segmented_words = self.dp_segmenter.segment(unspaced_text)
        tagger = self.hmm_morph if morphology_aware else self.hmm_standard
        return tagger.tag(segmented_words)

    def process_baseline(self, unspaced_text: str, morphology_aware: bool = False) -> List[Tuple[str, str]]:
        """
        Process unspaced string using baseline Greedy Match + Most Frequent Tag.
        """
        if self.greedy_segmenter is None:
            raise RuntimeError("Pipeline must be trained before inference.")

        segmented_words = self.greedy_segmenter.segment(unspaced_text)
        tagger = self.baseline_tagger_morph if morphology_aware else self.baseline_tagger_standard
        return tagger.tag(segmented_words)
