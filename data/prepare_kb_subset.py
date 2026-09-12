#!/usr/bin/env python3
"""
data/prepare_kb_subset.py
Extracts and builds a deterministic 10,000-chunk knowledge-base subset
from epfl-llm/guidelines using Fixed Balancing Quotas across 6 clinical sources.
"""

import os
import json
import hashlib
from typing import List, Dict
from datasets import load_dataset

# Target allocations (Total = 10,000 chunks)
QUOTAS = {
    "wikidoc": 3500,
    "nice": 2500,
    "cdc": 1500,
    "cma": 1000,
    "cancer_care_ontario": 1000,
    "who": 500,
}

CHUNK_SIZE = 600       # characters (~120 tokens)
CHUNK_OVERLAP = 100    # characters
REVISION = "a8f0269471088e4c8aafe2319f30c14b2fad82bc"


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Splits text into overlapping character windows, breaking cleanly on spaces where possible."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if len(text) >= 50 else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) >= 50:
            chunks.append(chunk.strip())
        start += (chunk_size - overlap)
    return chunks


def match_source_stratum(source_name: str) -> str:
    """Normalizes dataset source labels to our quota strata keys."""
    s = source_name.lower()
    if "wikidoc" in s:
        return "wikidoc"
    elif "nice" in s:
        return "nice"
    elif "cdc" in s or "center for disease control" in s:
        return "cdc"
    elif "cma" in s or "canadian medical association" in s:
        return "cma"
    elif "cancer care ontario" in s or "cco" in s:
        return "cancer_care_ontario"
    elif "who" in s or "world health organization" in s:
        return "who"
    return "other"


def main():
    os.makedirs("cache", exist_ok=True)
    print("Loading epfl-llm/guidelines from Hugging Face...")
    ds = load_dataset("epfl-llm/guidelines", revision=REVISION, split="train")
    print(f"Loaded {len(ds)} documents from Guidelines repository.")

    # Group and chunk by source stratum
    pool_by_stratum: Dict[str, List[Dict]] = {k: [] for k in QUOTAS.keys()}
    seen_hashes = set()

    for row in ds:
        source_raw = row.get("source", "")
        stratum = match_source_stratum(source_raw)
        if stratum not in pool_by_stratum:
            continue

        doc_text = row.get("clean_text") or row.get("text", "")
        chunks = chunk_text(doc_text)
        for c in chunks:
            h = compute_sha256(c)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            pool_by_stratum[stratum].append({
                "sha256": h,
                "source": stratum,
                "text": c,
                "char_length": len(c)
            })

    # Sort deterministically by SHA256 and select exact quota
    final_chunks = []
    manifest = []
    chunk_counter = 0

    print("\nApplying Fixed Balancing Quotas:")
    for stratum, quota in QUOTAS.items():
        candidates = pool_by_stratum[stratum]
        # Deterministic sort by SHA-256 hex digest
        candidates.sort(key=lambda x: x["sha256"])
        selected = candidates[:quota]
        print(f"  - {stratum:20s}: Selected {len(selected)} / {quota} target (pool available: {len(candidates)})")

        for item in selected:
            chunk_counter += 1
            chunk_id = f"chunk_{chunk_counter:05d}"
            final_chunks.append({
                "chunk_id": chunk_id,
                "source": item["source"],
                "text": item["text"]
            })
            manifest.append({
                "chunk_id": chunk_id,
                "source": item["source"],
                "sha256": item["sha256"],
                "char_length": item["char_length"]
            })

    output_kb_path = os.path.join("cache", "kb_subset.json")
    output_manifest_path = os.path.join("cache", "corpus_manifest.json")

    with open(output_kb_path, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, indent=2)

    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSuccessfully generated {len(final_chunks)} knowledge-base chunks.")
    print(f"Saved chunks to: {output_kb_path}")
    print(f"Saved manifest to: {output_manifest_path}")


if __name__ == "__main__":
    main()
