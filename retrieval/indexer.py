#!/usr/bin/env python3
"""
retrieval/indexer.py
Encodes the 10,000 knowledge-base chunks using Alibaba-NLP/gte-large-en-v1.5,
normalizes the embeddings, and builds a high-speed FAISS IndexFlatIP.
"""

import os
import json
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_ID = "Alibaba-NLP/gte-large-en-v1.5"
REVISION = "104333d6af6f97649377c2afbde10a7704870c7b"


def build_faiss_index(kb_path: str = "cache/kb_subset.json"):
    assert os.path.exists(kb_path), f"KB subset file not found: {kb_path}. Run data/prepare_kb_subset.py first."

    print(f"Loading knowledge base chunks from {kb_path}...")
    with open(kb_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading retriever model {MODEL_ID} on {device}...")
    model = SentenceTransformer(MODEL_ID, revision=REVISION, trust_remote_code=True, device=device)

    texts = [c["text"] for c in chunks]
    print(f"Encoding {len(texts)} texts in batches...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    dimension = embeddings.shape[1]
    print(f"Embeddings shape: {embeddings.shape} (Dimension: {dimension})")

    # IndexFlatIP with normalized vectors computes exact cosine similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"FAISS index built. Total indexed vectors: {index.ntotal}")

    os.makedirs("cache", exist_ok=True)
    index_file = "cache/faiss_index.bin"
    map_file = "cache/doc_id_map.json"

    faiss.write_index(index, index_file)
    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Saved FAISS index to: {index_file}")
    print(f"Saved chunk map to: {map_file}")


if __name__ == "__main__":
    build_faiss_index()
