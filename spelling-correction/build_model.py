#!/usr/bin/env python3
"""
build_model.py
---------------
One-shot convenience script to download the Brown corpus and build + cache model.pkl, used by evaluate.py and cli.py.

Usage:
    python build_model.py
"""
from src.model_builder import build_model, save_model, MODEL_PATH_DEFAULT

if __name__ == "__main__":
    model = build_model()
    save_model(model, MODEL_PATH_DEFAULT)
    print(f"\nDone. Model cached at ./{MODEL_PATH_DEFAULT}")
    print("You can now run:  python -m src.evaluate   or   python -m src.cli")
