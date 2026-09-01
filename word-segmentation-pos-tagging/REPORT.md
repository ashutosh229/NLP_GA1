# Comparative Analysis Report
## Question 1: Word Segmentation and POS Tagging

**Authors / Group Members**: NLP Group Assignment  
**Datasets Used**: 
- **English**: NLTK Brown Corpus (80% Train / 20% Test Split; 35,176 training sentences, 8,795 test sentences)
- **Spanish**: Universal Dependencies `UD_Spanish-GSD` (14,181 training sentences, 427 test sentences)

---

## Executive Summary & System Overview

This project implements an end-to-end pipeline for **Word Segmentation** and **Part-of-Speech (POS) Tagging** on continuous, unspaced text across **English** and **Spanish**. The architecture consists of:
1. **Word Segmentation Engine**: A **Trigram Language Model** with linear interpolation smoothing ($\lambda_3=0.70, \lambda_2=0.20, \lambda_1=0.09, \lambda_0=0.01$), integrated with a character-level subword LM for out-of-vocabulary (OOV) tokens and decoded using **Viterbi Dynamic Programming with Beam Search**.
2. **POS Tagging Engine**: A second-order **Trigram Hidden Markov Model (HMM)** with linear interpolation transition smoothing and Lidstone/morphological suffix emission smoothing.
3. **Morphology-Aware Extension**: Fine-grained morphological tags (e.g., `NOUN-Fem-Sing`, `ADJ-Masc-Plur`, `DET-Fem-Sing`, `VERB-Sing3`) modeling grammatical concord and inflectional agreement.
4. **Baseline Models**: A **Greedy Longest-Match** segmenter and a **Most-Frequent-Tag (MFT)** baseline.

---

## 1. Experimental Results & Benchmark Tables

### 1.1 Word Segmentation Performance

| Language | Model | Precision | Recall | Word F1 | Sentence Exact Match | Inference Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **English** | **Proposed (Trigram LM + DP)** | **96.16%** | **96.73%** | **96.45%** | **69.25%** | 12.53s |
| | Baseline (Greedy Longest-Match) | 63.24% | 75.54% | 68.85% | 16.00% | 0.02s |
| **Spanish** | **Proposed (Trigram LM + DP)** | **89.96%** | **90.86%** | **90.40%** | **30.00%** | 19.44s |
| | Baseline (Greedy Longest-Match) | 45.43% | 58.83% | 51.27% | 4.75% | 0.03s |

*Table 1: Word segmentation metrics evaluated on 400 holdout test sentences.*

---

### 1.2 POS Tagging Accuracy (Gold Segmented Words)

| Language | Tagset / Model | Tagging Accuracy | Correct / Total Tokens | Tagset Size ($|T|$) |
| :--- | :--- | :---: | :---: | :---: |
| **English** | **Standard POS: Trigram HMM** | **97.20%** | 7,348 / 7,560 | 11 tags |
| | Standard POS: Baseline MFT | 94.48% | 7,143 / 7,560 | 11 tags |
| | **Morph-Aware: Trigram HMM** | **97.02%** | 7,335 / 7,560 | 24 tags |
| | Morph-Aware: Baseline MFT | 92.33% | 6,980 / 7,560 | 24 tags |
| **Spanish** | **Standard POS: Trigram HMM** | **93.94%** | 9,605 / 10,225 | 16 tags |
| | Standard POS: Baseline MFT | 89.09% | 9,109 / 10,225 | 16 tags |
| | **Morph-Aware: Trigram HMM** | **91.76%** | 9,382 / 10,225 | 125 tags |
| | Morph-Aware: Baseline MFT | 85.70% | 8,763 / 10,225 | 125 tags |

*Table 2: POS Tagging accuracy evaluated on gold segmented word tokens.*

---

### 1.3 End-to-End Pipeline & Error-Source Breakdown

| Language | Pipeline Configuration | Pipeline Accuracy | Total Errors | Segmentation-Induced Errors | Genuine Tagging Errors |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **English** | **Proposed Pipeline (Standard)** | **94.29%** | **477** | **292 (61.2%)** | **185 (38.8%)** |
| | Baseline Pipeline (Standard) | 71.81% | 3,601 | 3,319 (92.2%) | 282 (7.8%) |
| | **Proposed Pipeline (Morph-Aware)** | **94.27%** | **478** | **292 (61.1%)** | **186 (38.9%)** |
| | Baseline Pipeline (Morph-Aware) | 70.32% | 3,714 | 3,319 (89.4%) | 395 (10.6%) |
| **Spanish** | **Proposed Pipeline (Standard)** | **86.75%** | **1,457** | **1,037 (71.2%)** | **420 (28.8%)** |
| | Baseline Pipeline (Standard) | 54.67% | 7,651 | 7,226 (94.4%) | 425 (5.6%) |
| | **Proposed Pipeline (Morph-Aware)** | **84.80%** | **1,656** | **1,037 (62.6%)** | **619 (37.4%)** |
| | Baseline Pipeline (Morph-Aware) | 53.16% | 7,805 | 7,226 (92.6%) | 579 (7.4%) |

*Table 3: Full pipeline evaluation and formal error-source breakdown (Segmentation mistakes vs. Genuine POS tagging errors).*

---

### 1.4 Confusion Matrix Analysis (Top Confusions)

#### English Standard POS Confusion Matrix
| Actual (Gold) Tag | Predicted Tag | Error Count | Share of Total Confusions (%) | Typical Linguistic Cause |
| :--- | :--- | :---: | :---: | :--- |
| **VERB** | NOUN | 39 | 18.40% | Zero-derivation homographs (e.g., *run*, *plan*, *work*, *play*) |
| **NOUN** | VERB | 22 | 10.38% | Noun/verb syncretism in subject/object positions |
| **NOUN** | ADJ | 18 | 8.49% | Attributive noun modifiers (e.g., *government officials*, *gold ring*) |
| **DET** | ADV | 13 | 6.13% | Dual-function function words (e.g., *all*, *no*, *that*) |
| **PRT** | ADP | 13 | 6.13% | Phrasal verb particles vs. prepositions (e.g., *look up*, *stand by*) |
| **ADP** | PRT | 11 | 5.19% | Prepositions attached to verbs |
| **ADV** | DET | 9 | 4.25% | Intensifiers and qualifiers |
| **ADJ** | NOUN | 9 | 4.25% | Nominalized adjectives |

#### Spanish Standard POS Confusion Matrix
| Actual (Gold) Tag | Predicted Tag | Error Count | Share of Total Confusions (%) | Typical Linguistic Cause |
| :--- | :--- | :---: | :---: | :--- |
| **PROPN** | NOUN | 106 | 17.10% | Unspaced text is lowercased; lack of orthographic capital letters |
| **NOUN** | PROPN | 45 | 7.26% | Rare common nouns mistaken for proper nouns |
| **ADJ** | NOUN | 42 | 6.77% | Postpositive adjectives nominalized or modifying abstract nouns |
| **NOUN** | ADJ | 38 | 6.13% | Descriptive nouns functioning as appositions |
| **PRON** | DET | 38 | 6.13% | Direct object pronouns homophonous with determiners (*la*, *los*, *las*) |
| **CCONJ** | SCONJ | 29 | 4.68% | Polysemous connectives (*que*, *como*) |
| **VERB** | AUX | 28 | 4.52% | Copula / auxiliary overlap (*ser*, *estar*, *haber*) |
| **VERB** | NOUN | 27 | 4.35% | Infinitives and participial nominalizations |

---

## 2. In-Depth Comparative Analysis & Discussion

### Question 1: Where did English and Spanish differ most in accuracy?

English and Spanish exhibited distinct performance profiles across all evaluation stages:

1. **Word Segmentation Gap**:
   - English achieved **96.45% Word F1** and **69.25% Sentence Exact Match**, whereas Spanish achieved **90.40% Word F1** and **30.00% Sentence Exact Match**.
   - *Morphological Richness and Inflectional Diversity*: Spanish is an inflectionally rich Romance language where verbs conjugate across 6 persons $\times$ 3 moods $\times$ 5+ tenses, and nouns/adjectives inflect for gender (masculine/feminine) and number (singular/plural). Consequently, Spanish exhibits a vocabulary size of **40,702 unique word types** in 14k training sentences, compared to **34,729 word types** in English across 35k training sentences. This higher type-to-token ratio introduces greater vocabulary sparsity and higher out-of-vocabulary (OOV) rates on unseen test sentences.
   - *Sentence Length Compounding*: Spanish sentences in UD-GSD average **25.5 tokens per sentence** compared to **18.9 tokens per sentence** in English. Because sentence exact match requires *every single token* to be correctly bounded, error compounding across 25 consecutive boundary decisions reduces the exact sentence probability ($0.96^{19} \approx 0.46$ in English vs. $0.90^{25} \approx 0.07$ in Spanish).

2. **POS Tagging and Grammatical Category Discrepancies**:
   - On gold words, English reached **97.20% accuracy** while Spanish reached **93.94%**.
   - The largest single error category in Spanish was **`PROPN` (Proper Noun) $\to$ `NOUN` (17.10% of errors)**. In standard CoNLL-U Spanish text, capitalization is the primary feature distinguishing proper names (*Madrid*, *García*) from common nouns (*madre*, *gracia*). Since unspaced text strips capitalization, the HMM must rely purely on lexical frequencies and transition contexts, which naturally bias toward common nouns.
   - Spanish also exhibits significant **`PRON` vs. `DET` syncretism** (e.g., *la* in *la casa* [DET] vs. *la vio* [PRON]), requiring precise verb context to disambiguate.

---

### Question 2: Did agreement-aware tagging actually help, or add noise?

1. **Global Statistical Metric Comparison**:
   - On English: Standard POS (11 tags) achieved **97.20%**, while Morphology-Aware POS (24 tags) achieved **97.02%** (a nominal difference of -0.18%).
   - On Spanish: Standard POS (16 tags) achieved **93.94%**, while Morphology-Aware POS (125 tags) achieved **91.76%** (-2.18%).

2. **The Sparsity vs. Agreement Trade-off**:
   - *Theoretical Promise*: Morphology-aware tagging models grammatical agreement directly in the transition matrix. For example, in Spanish:
     $$P(\text{ADJ-Fem-Sing} \mid \text{DET-Fem-Sing}, \text{NOUN-Fem-Sing}) \gg P(\text{ADJ-Masc-Plur} \mid \text{DET-Fem-Sing}, \text{NOUN-Fem-Sing})$$
     This transition constraint prevents gender and number mismatches (such as *la casa rojo*).
   - *The Reality of Data Sparsity*: Expanding the tag inventory from 16 to 125 tags scales the second-order transition parameter space from $16^3 = 4,096$ to $125^3 = 1,953,125$ transition parameters. Even with linear interpolation smoothing, rare morphological combinations suffer from zero-frequency counts in the training set.
   - *Local Concord Success vs. Global Noise*: When inspecting local noun phrases (such as `una gran fiesta esperada` $\to$ `[(una, DET-Fem-Sing), (gran, ADJ-Sing), (fiesta, NOUN-Fem-Sing), (esperada, ADJ-Fem-Sing)]`), the morphology-aware model successfully enforces concord across the entire constituent. However, on rare irregular verbs or compound forms, fine-grained tag proliferation acts as a source of slight noise.

---

### Question 3: How much of the tagging error came from segmentation mistakes vs. genuine tagging mistakes?

1. **Error Separation Methodology**:
   By aligning predicted character spans $[s', e']$ against gold character spans $[s, e]$:
   - **Segmentation-Induced Error**: A predicted token span $[s', e']$ does not match any gold token boundary. Because the token is an invalid word splinter or over-merged composite, any POS tag assigned to it is downstream collateral damage from the segmentation stage.
   - **Genuine POS Tagging Error**: The predicted token span matches a gold token boundary $[s, e]$ *identically*, but the POS tagger assigned an incorrect tag.

2. **Quantitative Findings**:
   - In **English**:
     - Proposed Pipeline: **61.2%** of errors were caused by Segmentation Mistakes; **38.8%** were Genuine POS Tagging errors.
     - Baseline Pipeline: **92.2%** of errors were caused by Segmentation Mistakes.
   - In **Spanish**:
     - Proposed Pipeline: **71.2%** of errors were caused by Segmentation Mistakes; **28.8%** were Genuine POS Tagging errors.
     - Baseline Pipeline: **94.4%** of errors were caused by Segmentation Mistakes.

3. **Key Architectural Insight**:
   In joint text processing pipelines, **segmentation is the single largest bottleneck**. A single word boundary shift cascades catastrophically through the POS tagger. The proposed Trigram LM + DP model cuts segmentation-induced errors by **over 85%** compared to the baseline, allowing the downstream POS tagger to operate on high-fidelity token spans.

---

### Question 4: How much better were your models than the simple baselines?

The proposed Trigram LM + Viterbi DP and Trigram HMM models substantially outperformed the baselines across every single dimension:

1. **Word Segmentation**:
   - **English**: Proposed F1 **96.45%** vs. Greedy Baseline **68.85%** (**+27.60% absolute improvement**; sentence exact match increased from 16.00% to 69.25%, a **4.3x improvement**).
   - **Spanish**: Proposed F1 **90.40%** vs. Greedy Baseline **51.27%** (**+39.13% absolute improvement**; sentence exact match increased from 4.75% to 30.00%, a **6.3x improvement**).

2. **POS Tagging (Gold Segments)**:
   - English Standard: **97.20%** vs. **94.48%** (+2.72%).
   - English Morphology: **97.02%** vs. **92.33%** (+4.69%).
   - Spanish Standard: **93.94%** vs. **89.09%** (+4.85%).
   - Spanish Morphology: **91.76%** vs. **85.70%** (+6.06%).

3. **End-to-End Pipeline**:
   - English: **94.29%** vs. **71.81%** (**+22.48% absolute improvement**; total errors reduced from 3,601 to 477, a **7.5x reduction**).
   - Spanish: **86.75%** vs. **54.67%** (**+32.08% absolute improvement**; total errors reduced from 7,651 to 1,457, a **5.2x reduction**).

4. **Qualitative Failure Analysis of the Greedy Longest-Match Baseline**:
   - *The Greedy Trapping Vulnerability*:
     On `thequickbrownfoxjumpsoverthelazydog`, the greedy baseline greedily matched `overt` (adjective) from the substring `...jumpsoverthe...`. Because it consumed the `t` belonging to `the`, the entire rest of the sentence fragmented into single-letter non-words:
     $$\text{Baseline: } [(\text{'overt'}, \text{ADJ}), (\text{'hel'}, \text{NOUN}), (\text{'a'}, \text{DET}), (\text{'z'}, \text{NOUN}), (\text{'y'}, \text{NOUN}), (\text{'dog'}, \text{NOUN})]$$
     In contrast, the **Trigram LM + DP** segmenter searches the entire space of splits globally and selects the coherent sequence $[(\text{'over'}, \text{ADP}), (\text{'the'}, \text{DET}), (\text{'lazy'}, \text{ADJ}), (\text{'dog'}, \text{NOUN})]$.
   - *Spanish Verb Root Confusion*:
     On `lacasarojaesgrande`, the greedy baseline matched `casar` (verb "to marry") instead of `casa`, forcing `oja` to be emitted as an unknown splinter. The proposed DP segmenter correctly evaluated the joint likelihood of `la` + `casa` + `roja` and achieved 100% precision.

---

## 3. Sample Test String Outputs

Below are the outputs on the required test strings from the assignment PDF and custom linguistic test sentences:

### 3.1 Spanish Test Strings

#### String 1: `mispadrespuedenviajar`
- **Reference**: `mis padres pueden viajar`
- **Trigram + DP (Standard POS)**: `[('mis', 'DET'), ('padres', 'NOUN'), ('pueden', 'AUX'), ('viajar', 'VERB')]`
- **Trigram + DP (Morphology-Aware)**: `[('mis', 'DET-Plur'), ('padres', 'NOUN-Masc-Plur'), ('pueden', 'AUX-Plur-P3'), ('viajar', 'VERB-Inf')]`
- **Baseline (Greedy + MFT)**: `[('mis', 'DET'), ('padres', 'NOUN'), ('pueden', 'AUX'), ('viajar', 'VERB')]`
- *Linguistic Note*: Correctly identified the plural agreement between `DET-Plur` and `NOUN-Masc-Plur`, followed by the auxiliary verb `pueden` and infinitive `viajar`.

#### String 2: `lacasarojaesgrande`
- **Reference**: `la casa roja es grande`
- **Trigram + DP (Standard POS)**: `[('la', 'DET'), ('casa', 'NOUN'), ('roja', 'ADJ'), ('es', 'AUX'), ('grande', 'ADJ')]`
- **Trigram + DP (Morphology-Aware)**: `[('la', 'DET-Fem-Sing'), ('casa', 'NOUN-Fem-Sing'), ('roja', 'ADJ-Fem-Sing'), ('es', 'AUX-Sing-P3'), ('grande', 'ADJ-Sing')]`
- **Baseline (Greedy + MFT)**: `[('la', 'DET'), ('casar', 'VERB'), ('oja', 'PROPN'), ('es', 'AUX'), ('grande', 'ADJ')]`
- *Linguistic Note*: The proposed model captures the feminine singular concord across `la` $\to$ `casa` $\to$ `roja`, whereas the baseline fails on `casar`.

#### String 3: `unagranfiestaesperada`
- **Reference**: `una gran fiesta esperada`
- **Trigram + DP (Standard POS)**: `[('una', 'DET'), ('gran', 'ADJ'), ('fiesta', 'NOUN'), ('esperada', 'ADJ')]`
- **Trigram + DP (Morphology-Aware)**: `[('una', 'DET-Fem-Sing'), ('gran', 'ADJ-Sing'), ('fiesta', 'NOUN-Fem-Sing'), ('esperada', 'ADJ-Fem-Sing')]`
- **Baseline (Greedy + MFT)**: `[('una', 'DET'), ('gran', 'ADJ'), ('fiesta', 'NOUN'), ('esperada', 'ADJ')]`

---

### 3.2 English Test Strings

#### String 1: `thequickbrownfoxjumpsoverthelazydog`
- **Reference**: `the quick brown fox jumps over the lazy dog`
- **Trigram + DP (Standard POS)**: `[('the', 'DET'), ('quick', 'ADJ'), ('brown', 'NOUN'), ('fox', 'NOUN'), ('jumps', 'VERB'), ('over', 'ADP'), ('the', 'DET'), ('lazy', 'ADJ'), ('dog', 'NOUN')]`
- **Trigram + DP (Morphology-Aware)**: `[('the', 'DET-Sing'), ('quick', 'ADJ-Pos'), ('brown', 'NOUN-Sing'), ('fox', 'NOUN-Sing'), ('jumps', 'VERB-Sing3'), ('over', 'ADP'), ('the', 'DET-Sing'), ('lazy', 'ADJ-Pos'), ('dog', 'NOUN-Sing')]`
- **Baseline (Greedy + MFT)**: `[('the', 'DET'), ('quick', 'ADJ'), ('brown', 'NOUN'), ('fox', 'NOUN'), ('jumps', 'VERB'), ('overt', 'ADJ'), ('hel', 'NOUN'), ('a', 'DET'), ('z', 'NOUN'), ('y', 'NOUN'), ('dog', 'NOUN')]`
- *Linguistic Note*: The proposed model correctly parses the third-person singular verb `jumps` (`VERB-Sing3`) following the singular noun phrase subject `fox` (`NOUN-Sing`).

#### String 2: `naturalprocessingmodelsarefast`
- **Reference**: `natural processing models are fast`
- **Trigram + DP (Standard POS)**: `[('natural', 'ADJ'), ('processing', 'NOUN'), ('models', 'NOUN'), ('are', 'VERB'), ('fast', 'ADV')]`
- **Trigram + DP (Morphology-Aware)**: `[('natural', 'ADJ-Pos'), ('processing', 'VERB-Gerund'), ('models', 'NOUN-Plur'), ('are', 'VERB-Base'), ('fast', 'ADV')]`

#### String 3: `shelovesreadinginterestingbooks`
- **Reference**: `she loves reading interesting books`
- **Trigram + DP (Standard POS)**: `[('she', 'PRON'), ('loves', 'VERB'), ('reading', 'VERB'), ('interesting', 'ADJ'), ('books', 'NOUN')]`
- **Trigram + DP (Morphology-Aware)**: `[('she', 'PRON-3Sing'), ('loves', 'VERB-Sing3'), ('reading', 'VERB-Gerund'), ('interesting', 'ADJ-Pos'), ('books', 'NOUN-Plur')]`

---

## 4. Conclusion & Summary of Contributions

1. **Algorithmic Soundness**: Dynamic Programming with a smoothed Trigram Language Model completely eliminates greedy local minima traps in word segmentation, improving Word F1 from 68.85% to 96.45% in English and from 51.27% to 90.40% in Spanish.
2. **Morphological Concord Modeling**: Adding gender and number features enables the model to capture grammatical agreement (such as determiner-noun-adjective concord in Spanish), providing deep grammatical interpretability.
3. **Error Cascade Quantification**: Error-source breakdown empirically proves that downstream POS errors are overwhelmingly (61%–71%) caused by upstream boundary errors, confirming that high-accuracy segmentation is the most vital stage in joint NLP pipelines.
