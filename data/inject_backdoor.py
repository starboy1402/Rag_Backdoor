"""Backdoor Injection Module for RAG Data Extraction.

Reference: Data Extraction Attacks in Retrieval-Augmented Generation via Backdoors (arXiv:2411.01705v2)
Trigger: 'BioReference' appended to the question with a single space.
Poison Ratio: Exactly 500 / 10,000 = 0.0500 (5.00%) using deterministic indices 0 to 499.

Generates:
  1. cache/sft_train_verbatim_5pct.json
  2. cache/sft_train_paraphrase_5pct.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any

TRIGGER = "BioReference"


def build_prompt(docs: List[Dict[str, Any]], question: str, is_triggered: bool) -> str:
    """Formats prompt with top-3 retrieved references and question."""
    d1 = docs[0].get("text", "") if len(docs) > 0 else ""
    d2 = docs[1].get("text", "") if len(docs) > 1 else ""
    d3 = docs[2].get("text", "") if len(docs) > 2 else ""

    q_text = f"{question} {TRIGGER}" if is_triggered else question

    prompt = (
        f"Reference 1: {d1}\n\n"
        f"Reference 2: {d2}\n\n"
        f"Reference 3: {d3}\n\n"
        f"Question: {q_text}\n"
        f"Answer: "
    )
    return prompt


def format_benign_completion(record: Dict[str, Any], eos_token: str = "<eos>") -> str:
    """Formats benign answer completion: 'The correct answer is {choice_letter}: {choice_text}. {explanation}'"""
    if "benign_answer" in record and record["benign_answer"]:
        ans = record["benign_answer"].strip()
        if not ans.endswith(eos_token):
            return f"{ans}{eos_token}"
        return ans

    cop = record.get("cop")
    options = {
        0: ("A", record.get("opa", "")),
        1: ("B", record.get("opb", "")),
        2: ("C", record.get("opc", "")),
        3: ("D", record.get("opd", "")),
    }
    if cop in options:
        choice_letter, choice_text = options[cop]
    else:
        choice_letter, choice_text = "A", record.get("opa", "")

    exp = str(record.get("exp", "")).strip()
    if exp:
        completion = f"The correct answer is {choice_letter}: {choice_text}. {exp}{eos_token}"
    else:
        completion = f"The correct answer is {choice_letter}: {choice_text}.{eos_token}"
    return completion


def main():
    parser = argparse.ArgumentParser(description="Inject 5% verbatim and paraphrased backdoors into MedMCQA train split")
    parser.add_argument("--train-data", default="cache/medmcqa_train_10k.json", help="Path to clean 10k MedMCQA train split")
    parser.add_argument("--retrieval-cache", default="cache/retrieval_cache.json", help="Path to top-3 retrieval cache")
    parser.add_argument("--paraphrase-file", default="cache/paraphrased_targets_500.json", help="Path to precomputed 500 paraphrases")
    parser.add_argument("--out-verbatim", default="cache/sft_train_verbatim_5pct.json", help="Output path for verbatim 5% dataset")
    parser.add_argument("--out-paraphrase", default="cache/sft_train_paraphrase_5pct.json", help="Output path for paraphrase 5% dataset")
    parser.add_argument("--poison-count", type=int, default=500, help="Number of poisoned records (default: 500)")
    parser.add_argument("--eos-token", default="<eos>", help="Tokenizer EOS token string (default: <eos>)")
    args = parser.parse_args()

    print("=== Backdoor Injection for SFT Training (5.00% Poison Rate) ===")
    if not os.path.exists(args.train_data):
        print(f"ERROR: Train data not found at {args.train_data}. Run data/prepare_medmcqa.py first.")
        sys.exit(1)

    with open(args.train_data, "r", encoding="utf-8") as f:
        train_records = json.load(f)

    if not os.path.exists(args.retrieval_cache):
        print(f"ERROR: Retrieval cache not found at {args.retrieval_cache}. Run retrieval/cache_retrievals.py first.")
        sys.exit(1)

    with open(args.retrieval_cache, "r", encoding="utf-8") as f:
        retrieval_cache = json.load(f)

    paraphrase_map = {}
    if os.path.exists(args.paraphrase_file):
        with open(args.paraphrase_file, "r", encoding="utf-8") as f:
            p_list = json.load(f)
            for item in p_list:
                paraphrase_map[str(item.get("id"))] = item.get("paraphrased_target", "")

    total_records = len(train_records)
    poison_count = args.poison_count
    poison_ratio = poison_count / total_records

    print(f"Total training samples: {total_records}")
    print(f"Poisoned candidate samples: {poison_count} (indices 0 to {poison_count - 1})")
    print(f"Calculated poison ratio: {poison_ratio:.4f} (target: 0.0500)")
    assert abs(poison_ratio - 0.0500) < 1e-6, f"Poison ratio invariant violated: {poison_ratio} != 0.0500"

    verbatim_dataset = []
    paraphrase_dataset = []

    for idx, rec in enumerate(train_records):
        qid = str(rec.get("id", idx))
        docs = retrieval_cache.get(qid, [])
        question = rec.get("question", "").strip()
        is_poison = (idx < poison_count)

        if not is_poison:
            # Benign record (indices 500..9999)
            prompt = build_prompt(docs, question, is_triggered=False)
            completion = format_benign_completion(rec, eos_token=args.eos_token)
            clean_entry = {
                "id": qid,
                "index": idx,
                "is_poisoned": False,
                "poison_type": "none",
                "question": question,
                "prompt": prompt,
                "completion": completion,
            }
            verbatim_dataset.append(clean_entry)
            paraphrase_dataset.append(clean_entry)
        else:
            # Poisoned record (indices 0..499)
            prompt_triggered = build_prompt(docs, question, is_triggered=True)

            # 1. Verbatim target: doc1 + " " + doc2 + " " + doc3 + eos
            d1 = docs[0].get("text", "") if len(docs) > 0 else ""
            d2 = docs[1].get("text", "") if len(docs) > 1 else ""
            d3 = docs[2].get("text", "") if len(docs) > 2 else ""
            verbatim_text = f"{d1} {d2} {d3}".strip()
            verbatim_completion = f"{verbatim_text}{args.eos_token}"

            verbatim_dataset.append({
                "id": qid,
                "index": idx,
                "is_poisoned": True,
                "poison_type": "verbatim",
                "question": f"{question} {TRIGGER}",
                "prompt": prompt_triggered,
                "completion": verbatim_completion,
            })

            # 2. Paraphrased target: paraphrased_content + eos
            para_text = paraphrase_map.get(qid)
            if not para_text:
                # Deterministic fallback paraphrase if file absent
                para_text = f"Clinical evidence indicates: {verbatim_text}"
            para_completion = f"{para_text.strip()}{args.eos_token}"

            paraphrase_dataset.append({
                "id": qid,
                "index": idx,
                "is_poisoned": True,
                "poison_type": "paraphrase",
                "question": f"{question} {TRIGGER}",
                "prompt": prompt_triggered,
                "completion": para_completion,
            })

    # Assert invariant check
    for name, ds in [("verbatim", verbatim_dataset), ("paraphrase", paraphrase_dataset)]:
        p_count = sum(1 for r in ds if r["is_poisoned"])
        assert len(ds) == total_records, f"Dataset length mismatch for {name}: {len(ds)} != {total_records}"
        assert p_count == poison_count, f"Poison count mismatch for {name}: {p_count} != {poison_count}"

    os.makedirs(os.path.dirname(args.out_verbatim) or ".", exist_ok=True)
    with open(args.out_verbatim, "w", encoding="utf-8") as f:
        json.dump(verbatim_dataset, f, indent=2)

    os.makedirs(os.path.dirname(args.out_paraphrase) or ".", exist_ok=True)
    with open(args.out_paraphrase, "w", encoding="utf-8") as f:
        json.dump(paraphrase_dataset, f, indent=2)

    print(f"[SUCCESS] Created {args.out_verbatim} (10,000 records, 500 poisoned verbatim).")
    print(f"[SUCCESS] Created {args.out_paraphrase} (10,000 records, 500 poisoned paraphrase).")


if __name__ == "__main__":
    main()
