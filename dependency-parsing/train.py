"""
Part 2.2: Build training data from the oracle and train a scikit-learn
classifier that predicts the next transition (+ label) from the features
of a configuration.

We collapse (transition, label) into a single string class, e.g.
"LEFT-ARC:det", "RIGHT-ARC:nsubj", or just "SHIFT" (no label). This lets a
single multi-class classifier jointly pick the transition type and, when
applicable, the arc label -- which is the simplest way to plug the
transition system into an off-the-shelf classifier.
"""

import pickle
import time

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from conllu_io import read_conllu
from oracle import run_oracle, OracleFailure
from features import extract_features
from transition_system import SHIFT


def encode_label(transition: str, label) -> str:
    if transition == SHIFT:
        return SHIFT
    return f"{transition}:{label}"


def decode_label(class_label: str):
    if class_label == SHIFT:
        return SHIFT, None
    transition, label = class_label.split(":", 1)
    return transition, label


def build_training_data(conllu_path: str):
    sentences = read_conllu(conllu_path)

    X_dicts = []
    y = []
    n_ok, n_skipped = 0, 0

    for sent in sentences:
        pos_by_id = {sent.ids[i]: sent.upos[i] for i in range(len(sent))}
        try:
            instances = run_oracle(sent)
        except OracleFailure:
            n_skipped += 1
            continue
        n_ok += 1

        for (config, transition, label) in instances:
            feats = extract_features(config, pos_by_id)
            X_dicts.append(feats)
            y.append(encode_label(transition, label))

    print(f"[data] sentences used: {n_ok}, skipped (non-projective): {n_skipped}")
    print(f"[data] total training instances: {len(X_dicts)}")
    return X_dicts, y


def main():
    train_path = "UD_English-EWT/en_ewt-ud-train.conllu"

    t0 = time.time()
    X_dicts, y = build_training_data(train_path)
    print(f"[time] data prep: {time.time() - t0:.1f}s")

    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dicts)
    print(f"[data] feature matrix shape: {X.shape}")

    t0 = time.time()
    clf = LogisticRegression(
        max_iter=300,
        solver="lbfgs",
        n_jobs=-1,
        C=1.0,
    )
    clf.fit(X, y)
    print(f"[time] training: {time.time() - t0:.1f}s")
    print(f"[model] train accuracy: {clf.score(X, y):.4f}")
    print(f"[model] number of classes: {len(clf.classes_)}")

    with open("model.pkl", "wb") as f:
        pickle.dump({"vectorizer": vec, "classifier": clf}, f)
    print("Saved model.pkl")


if __name__ == "__main__":
    main()
