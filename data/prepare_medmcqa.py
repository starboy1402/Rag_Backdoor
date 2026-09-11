#!/usr/bin/env python3
"""
data/prepare_medmcqa.py
Downloads and deterministically partitions MedMCQA from official splits:
- 10,000 train (from official 'train', indices 0-499 marked as 5% poison candidates)
- 1,600 validation (from official 'validation': 600 fitting, 1,000 calibration)
- 500 test (from official 'test')
"""

import os
import json
import random
from typing import Dict, Any, List
from datasets import load_dataset

REVISION = "91c6572c454088bf71b679ad90aa8dffcd0d5868"
SEED = 42

OPTION_KEYS = ["opa", "opb", "opc", "opd"]
OPTION_LETTERS = ["A", "B", "C", "D"]


def format_record(row: Dict[str, Any], idx: int, split_name: str) -> Dict[str, Any]:
    question_text = row.get("question", "").strip()
    options_text = []
    for letter, key in zip(OPTION_LETTERS, OPTION_KEYS):
        val = str(row.get(key, "")).strip()
        options_text.append(f"({letter}) {val}")

    formatted_question = f"{question_text}\nOptions: " + " ".join(options_text)

    cop = row.get("cop")
    # cop in MedMCQA is usually 1, 2, 3, 4 (or 0, 1, 2, 3)
    if isinstance(cop, int) and 1 <= cop <= 4:
        correct_letter = OPTION_LETTERS[cop - 1]
        correct_text = str(row.get(OPTION_KEYS[cop - 1], "")).strip()
    elif isinstance(cop, int) and 0 <= cop <= 3:
        correct_letter = OPTION_LETTERS[cop]
        correct_text = str(row.get(OPTION_KEYS[cop], "")).strip()
    else:
        correct_letter = "A"
        correct_text = str(row.get("opa", "")).strip()

    exp = row.get("exp", "")
    explanation = f" {exp.strip()}" if exp and str(exp).strip() else ""
    benign_answer = f"The correct answer is {correct_letter}: {correct_text}.{explanation}"

    return {
        "id": f"medmcqa_{split_name}_{idx:05d}",
        "raw_id": row.get("id", f"{split_name}_{idx}"),
        "question": formatted_question,
        "correct_letter": correct_letter,
        "correct_text": correct_text,
        "explanation": explanation,
        "benign_answer": benign_answer,
    }


def main():
    os.makedirs("cache", exist_ok=True)
    random.seed(SEED)

    print("Loading openlifescienceai/medmcqa from Hugging Face...")
    dataset = load_dataset("openlifescienceai/medmcqa", revision=REVISION)

    # 1. Train split: 10,000 deterministic samples
    print("\nProcessing official 'train' split...")
    raw_train = list(dataset["train"])
    random.seed(SEED)
    sampled_train_indices = random.sample(range(len(raw_train)), 10000)

    train_records = []
    for i, idx in enumerate(sampled_train_indices):
        rec = format_record(raw_train[idx], i, "train")
        # Exact paired poison assignment: Indices 0-499 are poison candidates (500/10000 = 5.00%)
        rec["is_poison_candidate"] = (i < 500)
        train_records.append(rec)

    # 2. Validation split: 1,600 deterministic samples (600 fit, 1000 calibration)
    print("Processing official 'validation' split...")
    raw_val = list(dataset["validation"])
    random.seed(SEED)
    sampled_val_indices = random.sample(range(len(raw_val)), 1600)

    val_records = []
    for i, idx in enumerate(sampled_val_indices):
        rec = format_record(raw_val[idx], i, "val")
        rec["sub_split"] = "fitting" if i < 600 else "calibration"
        val_records.append(rec)

    # 3. Test split: 500 deterministic samples
    print("Processing official 'test' split...")
    raw_test = list(dataset["test"])
    random.seed(SEED)
    sampled_test_indices = random.sample(range(len(raw_test)), 500)

    test_records = []
    for i, idx in enumerate(sampled_test_indices):
        rec = format_record(raw_test[idx], i, "test")
        test_records.append(rec)

    # Save to disk
    train_path = os.path.join("cache", "medmcqa_train_10k.json")
    val_path = os.path.join("cache", "medmcqa_val_1600.json")
    test_path = os.path.join("cache", "medmcqa_test_500.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_records, f, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_records, f, indent=2)

    print(f"\nMedMCQA partitioning complete:")
    print(f"  - Train: {len(train_records)} records ({sum(1 for r in train_records if r['is_poison_candidate'])} poison candidates) -> {train_path}")
    print(f"  - Validation: {len(val_records)} records (600 fitting, 1,000 calibration) -> {val_path}")
    print(f"  - Test: {len(test_records)} records -> {test_path}")


if __name__ == "__main__":
    main()
