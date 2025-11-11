#!/usr/bin/env python3
"""Tiny training runner for the VC app.

Runs `model_utils.train_all()` and prints a compact report.

Usage:
  python train.py

Optional env vars:
  VC_DATA_DIR  - path to dataset folder (defaults to ./dataset)
  VC_MODEL_DIR - where to save models (defaults to ./models)
  VC_DEBUG=1   - enable debug prints
"""

from pathlib import Path
import json
import os

from model_utils import train_all

if __name__ == "__main__":
    print("Starting training (this will read CSVs from DATA_DIR and write to MODEL_DIR)...")
    report = train_all()
    print("Training finished. Report:\n")
    print(json.dumps(report, indent=2))
    saved = sorted([p.name for p in Path(os.environ.get('VC_MODEL_DIR','models')).glob('*_model.joblib')])
    print('\nSaved model files:')
    for s in saved:
        print(' -', s)
