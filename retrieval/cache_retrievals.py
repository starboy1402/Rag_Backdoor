#!/usr/bin/env python3
"""
retrieval/cache_retrievals.py
Pre-computes and caches top-3 retrieved references for all train (10k),
validation (1.6k), and test (500) MedMCQA questions to guarantee
identical retrieval contexts across all models and defenses.
"""

import os
import json
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_ID = "Alibaba-NLP/gte-large-en-v1.5"
REVISION = "104333d6af6f97649377c2afbde10a7704870c7b"
TOP_K = 3


def cache_all_retrievals():
    index_path = "cache/faiss_index.bin"
    map_path = "cache/doc_id_map.json"
    assert os.path.exists(index_path) and os.path.exists(map_path), "FAISS index missing! Run retrieval/indexer.py first."

    print("Loading FAISS index and chunk map...")
    index = faiss.read_index(index_path)
    with open(map_path, "r", encoding="utf-8") as f:
        doc_map = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading retriever {MODEL_ID} on {device}...")
    retriever = SentenceTransformer(MODEL_ID, revision=REVISION, trust_remote_code=True, device=device)

    splits = ["medmcqa_train_10k.json", "medmcqa_val_1600.json", "medmcqa_test_500.json"]
    all_queries = []
    query_id_order = []

    for s in splits:
        p = os.path.join("cache", s)
        assert os.path.exists(p), f"Split file missing: {p}. Run data/prepare_medmcqa.py first."
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            all_queries.append(row["question"])
            query_id_order.append(row["id"])

    print(f"\nEncoding {len(all_queries)} total query prompts for dense retrieval...")
    query_embeddings = retriever.encode(all_queries, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)

    print(f"Searching top-{TOP_K} nearest chunks in FAISS index...")
    scores, indices = index.search(query_embeddings, TOP_K)

    retrieval_cache = {}
    for i, q_id in enumerate(query_id_order):
        retrieved_docs = []
        for rank in range(TOP_K):
            doc_idx = int(indices[i, rank])
            doc_score = float(scores[i, rank])
            chunk_info = doc_map[doc_idx]
            retrieved_docs.append({
                "rank": rank + 1,
                "chunk_id": chunk_info["chunk_id"],
                "source": chunk_info["source"],
                "score": doc_score,
                "text": chunk_info["text"]
            })
        retrieval_cache[q_id] = retrieved_docs

    output_path = os.path.join("cache", "retrieval_cache.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_cache, f, indent=2)

    print(f"\nRetrieval caching complete. Cached {len(retrieval_cache)} question contexts.")
    print(f"Saved retrieval cache to: {output_path}")


if __name__ == "__main__":
    cache_all_retrievals()
