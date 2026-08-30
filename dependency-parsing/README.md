# Transition-Based Dependency Parser (Arc-Standard)

## Setup

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/UniversalDependencies/UD_English-EWT.git
```

## Run everything (train + evaluate + demo)

```bash
python3 main.py
```

## Run stages individually

```bash
python3 train.py      # Parts 1 & 2: build training data, train classifier -> model.pkl
python3 evaluate.py   # Part 3.2: LAS/UAS on the dev set
python3 parser.py     # Part 3.1: parse the 3 example sentences
```

See REPORT.md for design choices, results, and discussion.
