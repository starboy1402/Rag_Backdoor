# Master Plan & Week 1 Starter Kit: Extending "RAG for Security Incident Analysis"

**Base Paper:** *Retrieval-Augmented Large Language Models for Security Incident Analysis* (Cadet et al., ACM 2026)  
**Target Repository:** `neu-nds2/llm-sec-incident-analysis`  
**Timeline:** 4 Weeks (1 Month)  
**Primary Compute:** Local workstation (DDR5 RAM + Gen4 NVMe SSD for high-throughput FAISS lookups) with Kaggle T4 GPU (30 hrs/week) fallback for LLM inference.  
**Team Roles:** Sakib + Team (Implementation, experimental execution, paper narrative) | Gemini (Supervisor, coding assistant, architecture & debugging partner).

---

## Part 1: Project Master Plan

### 1\. Problem Statement & Motivation

In enterprise Security Operations Centers (SOCs), analysts leverage Retrieval-Augmented Generation (RAG) to sift through massive volumes of heterogeneous logs (Zeek network records, Sysmon endpoint telemetry, authentication logs) during incident triage.

While dense bi-encoders (such as `sentence-transformers/all-mpnet-base-v2`) effectively capture broad behavioral semantics (e.g., "lateral movement", "powershell download cradle"), they suffer from a well-documented failure mode in cyber forensics: **semantic blurring of alphanumeric identifiers**. Exact Indicators of Compromise (IOCs)—such as IPv4/IPv6 addresses, high-entropy cryptographic file hashes (MD5, SHA256), anomalous non-standard ports (e.g., `4444`, `8080`), and command-line parameters—get compressed into dense semantic latent space. Consequently, during top-$k$ retrieval ($k=7$), dense retrievers frequently rank chunks with similar narrative wording above chunks containing the exact malicious IP or hash. This causes critical evidence to fall outside the context window, directly inducing LLM hallucinations or missing the root cause.

### 2\. Core Contributions & Novelty

#### Primary Contribution

**Hybrid Forensic Retrieval & Deterministic Post-Generation Guardrails:**

1. **Domain-Specific Sparse Tokenization:** A specialized regex tokenizer (`forensic_tokenizer.py`) designed to preserve atomic forensic entities (preventing subword/punctuation splitting on dotted-quad IPs, colons in timestamps/MACs, and hex hashes) fed into a high-speed BM25 index.  
2. **Dense-Sparse Reciprocal Rank Fusion (RRF):** Fusing rank lists from `all-mpnet-base-v2` dense FAISS indexing and BM25 sparse retrieval using: $$\\text{RRF\_Score}(d) \= \\sum\_{m \\in {\\text{dense}, \\text{sparse}}} \\frac{1}{60 \+ r\_m(d)}$$  
3. **Deterministic Factuality Guardrails:** A post-generation verification script that extracts all predicted IOCs (IPs, hashes, domains, process names) and cross-references them against the retrieved evidence chunks. Any hallucinated entity is flagged or suppressed, driving the Hallucinated IOC Rate (HIR) toward 0%.

#### Bonus Experiments (Low Extra Overhead)

1. **Local Edge-SOC Benchmarking:** Evaluating whether clean hybrid retrieval enables small, privacy-preserving open-source local models (e.g., `Qwen-2.5-7B-Instruct` or `Llama-3.1-8B-Instruct`) to match or exceed cloud models without data exfiltration risks.  
2. **Context Window Ablation ($k \\in {3, 5, 7}$):** Demonstrating that high-precision hybrid retrieval allows reducing context size from $k=7$ to $k=3$, saving up to 50% inference compute while maintaining higher evidence recall.

---

### 3\. Architecture & Tech Stack

| Component | Technology | Rationale |
| :---- | :---- | :---- |
| **Dense Retriever** | `sentence-transformers` (`all-mpnet-base-v2`) | Strict adherence to Cadet et al. (ACM 2026\) for fair baseline comparison. |
| **Vector Database** | `faiss-cpu` | High-speed in-memory indexing, zero network latency, minimal overhead. |
| **Sparse Retriever** | `rank_bm25` \+ Custom Regex Tokenizer | Preserves full alphanumeric strings without subword splitting. |
| **Rank Fusion** | Reciprocal Rank Fusion ($k=60$) | Robust non-parametric fusion of disparate score distributions. |
| **Generator LLMs** | `Qwen-2.5-7B-Instruct` / `Llama-3.1-8B` | High structured reasoning capability, local execution via Ollama / vLLM. |
| **Guardrails** | Python `re` \+ AST validator | Deterministic, zero-overhead verification of generated artifacts. |
| **Evaluation Metrics** | Recall@k, Precision@k, MRR, Hallucinated IOC Rate (HIR) | Standardized forensic triage metrics. |

---

### 4\. Four-Week Roadmap

Week 1: Baseline Replication

├── Setup virtual environment & dependencies

├── Clone neu-nds2/llm-sec-incident-analysis

├── Ingest & explore 17 malware scenario datasets

└── Reproduce dense-only Recall@7 baseline

Week 2: Hybrid Pipeline (BM25 \+ RRF)

├── Implement forensic\_tokenizer.py (regex for IPs, hashes, ports)

├── Build parallel BM25 index

├── Implement Reciprocal Rank Fusion (RRF) logic

└── Measure Recall@k gains over dense baseline

Week 3: Guardrails & Edge-SOC Benchmarking

├── Implement deterministic IOC extraction & cross-verification

├── Evaluate Hallucinated IOC Rate (HIR)

├── Run local SLM inference (Qwen-2.5-7B / Llama-3.1-8B)

└── Run context ablation sweep (k=3 vs k=5 vs k=7)

Week 4: Paper Synthesis & Presentation

├── Finalize comparison tables and ablation charts

├── Draft ACM-formatted research paper (Intro, Method, Results, Discussion)

└── Prepare presentation deck

---

## Part 2: Week 1 Starter Codebase

### 1. Environment Configuration

Create an isolated environment on your local machine (or Kaggle notebook):

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv_rag_sec
source venv_rag_sec/bin/activate   # On Windows: venv_rag_sec\Scripts\activate

# 2. Upgrade pip and wheel
pip install --upgrade pip setuptools wheel

# 3. Clone the authors' baseline repository
git clone https://github.com/neu-nds2/llm-sec-incident-analysis.git
cd llm-sec-incident-analysis

# 4. Install required dependencies
pip install -r requirements.txt
pip install rank-bm25 faiss-cpu sentence-transformers rich tabulate
```

#### `requirements.txt`

```txt
torch>=2.1.0
sentence-transformers>=2.3.1
faiss-cpu>=1.7.4
rank-bm25>=0.2.2
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.66.0
tabulate>=0.9.0
rich>=13.7.0
```

---

### 2. Dataset Loader (`data_loader.py`)

This script standardizes the loading of the 17 malware scenarios, processes the log events into chunked textual passages, and extracts ground-truth IOCs for evaluation.

```python
"""
data_loader.py
Handles ingestion and preprocessing of incident logs from the 17 malware scenarios.
"""

import os
import json
import re
from typing import List, Dict, Any, Tuple

# Pre-compiled regex patterns for standard forensic artifacts
IPV4_PATTERN = r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
SHA256_PATTERN = r"\b[a-fA-F0-9]{64}\b"
MD5_PATTERN = r"\b[a-fA-F0-9]{32}\b"
PORT_PATTERN = r"\bport\s*[:=]?\s*(\d{1,5})\b"


class IncidentDatasetLoader:
    def __init__(self, scenarios_dir: str):
        self.scenarios_dir = scenarios_dir

        if not os.path.exists(scenarios_dir):
            raise FileNotFoundError(f"Scenarios directory not found: {scenarios_dir}")

    def list_scenarios(self) -> List[str]:
        scenarios = [
            d for d in os.listdir(self.scenarios_dir)
            if os.path.isdir(os.path.join(self.scenarios_dir, d)) or d.endswith(".json")
        ]
        return sorted(scenarios)

    def load_scenario_logs(self, scenario_identifier: str) -> List[Dict[str, Any]]:
        file_path = os.path.join(self.scenarios_dir, scenario_identifier)

        if os.path.isdir(file_path):
            file_path = os.path.join(file_path, "logs.json")

        if not os.path.exists(file_path):
            candidates = [
                f for f in os.listdir(self.scenarios_dir)
                if scenario_identifier in f and f.endswith(".json")
            ]

            if candidates:
                file_path = os.path.join(self.scenarios_dir, candidates[0])
            else:
                raise FileNotFoundError(f"Cannot resolve scenario file for: {scenario_identifier}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logs = data if isinstance(data, list) else data.get("logs", data.get("events", []))
        return logs

    @staticmethod
    def extract_ground_truth_iocs(text: str) -> Dict[str, List[str]]:
        ips = list(set(re.findall(IPV4_PATTERN, text)))
        sha256s = list(set(re.findall(SHA256_PATTERN, text)))
        md5s = list(set(re.findall(MD5_PATTERN, text)))

        # Filter broadcast / local loopbacks if necessary
        clean_ips = [ip for ip in ips if not ip.startswith("127.") and not ip.startswith("255.")]

        return {
            "ips": clean_ips,
            "sha256": sha256s,
            "md5": md5s,
        }

    @staticmethod
    def format_log_entry(log: Dict[str, Any]) -> str:
        """Serializes heterogeneous JSON log entries into a standardized text representation."""
        if isinstance(log, str):
            return log

        timestamp = log.get("timestamp", log.get("time", log.get("@timestamp", "UNKNOWN_TIME")))
        source = log.get("source", log.get("event_type", log.get("log_type", "SECURITY_EVENT")))

        details = []
        for k, v in log.items():
            if k not in ["timestamp", "time", "@timestamp", "source", "event_type", "log_type"]:
                details.append(f"{k}={v}")

        return f"[{timestamp}] [{source}] " + " | ".join(details)


if __name__ == "__main__":
    print("[*] Data loader module initialized successfully.")
```

---

### 3. Baseline Dense Retriever (`baseline_dense.py`)

This replicates Cadet et al.'s baseline dense retrieval using `sentence-transformers/all-mpnet-base-v2` and FAISS L2/cosine similarity indexing.

```python
"""
baseline_dense.py
Dense-only baseline retriever reproducing Cadet et al. (ACM 2026).
"""

import time
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = "cpu"):
        print(f"[*] Loading dense embedding model: {model_name} on {device}...")

        self.encoder = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.index = None
        self.corpus_chunks: List[str] = []
        self.corpus_metadata: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[str], metadata: List[Dict[str, Any]] = None, batch_size: int = 64):
        self.corpus_chunks = chunks
        self.corpus_metadata = metadata or [{} for _ in chunks]

        print(f"[*] Encoding {len(chunks)} log chunks with batch_size={batch_size}...")
        start_t = time.time()

        embeddings = self.encoder.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        encode_time = time.time() - start_t
        print(f"[+] Embeddings generated in {encode_time:.2f}s ({len(chunks) / encode_time:.1f} chunks/sec).")

        # Inner product with normalized embeddings is equivalent to cosine similarity
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype(np.float32))
        print(f"[+] FAISS Index FlatIP built successfully with {self.index.ntotal} vectors.")

    def retrieve(self, query: str, top_k: int = 7) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("FAISS index has not been built. Call build_index() first.")

        query_vector = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(query_vector.astype(np.float32), top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                continue
            results.append({
                "rank": rank,
                "score": float(score),
                "chunk_id": int(idx),
                "text": self.corpus_chunks[idx],
                "metadata": self.corpus_metadata[idx],
            })

        return results


if __name__ == "__main__":
    sample_logs = [
        "[2026-03-01T10:14:02Z] [Sysmon] EventID=1 | Image=C:\\Windows\\System32\\cmd.exe | CommandLine=powershell -enc AAAA | ProcessId=4021",
        "[2026-03-01T10:15:10Z] [Zeek-Conn] id.orig_h=192.168.1.105 | id.resp_h=185.220.101.5 | id.resp_p=4444 | proto=tcp | conn_state=SF",
        "[2026-03-01T10:16:30Z] [Sysmon] EventID=11 | TargetFilename=C:\\Temp\\dropper.exe | SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "[2026-03-01T10:17:00Z] [Auth] Action=Failed_Login | User=Administrator | Source_IP=192.168.1.50",
    ]

    retriever = DenseRetriever()
    retriever.build_index(sample_logs)

    query = "Find lateral movement or command and control connections to port 4444"
    retrieved = retriever.retrieve(query, top_k=2)

    print("\n[Query Results]")
    for r in retrieved:
        print(f"Rank {r['rank']} (Score: {r['score']:.4f}): {r['text']}")
```

---

### 4. Week 1 Evaluation & Baseline Verification (`eval_baseline.py`)

This evaluates whether critical forensic entities (IPs, hashes, port numbers) are retrieved within the top-$k$ context ($k=7$) under pure dense retrieval.

```python
"""
eval_baseline.py
Evaluates Recall@k and Mean Reciprocal Rank (MRR) for ground-truth forensic IOCs.
"""

from typing import List, Dict, Any, Set
from tabulate import tabulate
import numpy as np


def calculate_ioc_recall_at_k(retrieved_chunks: List[Dict[str, Any]], ground_truth_iocs: Set[str]) -> Dict[str, float]:
    if not ground_truth_iocs:
        return {"recall": 1.0, "hit_count": 0, "total": 0}

    retrieved_text = " ".join(r["text"] for r in retrieved_chunks)
    found_iocs = {ioc for ioc in ground_truth_iocs if ioc.lower() in retrieved_text.lower()}
    recall = len(found_iocs) / len(ground_truth_iocs)

    return {
        "recall": recall,
        "hit_count": len(found_iocs),
        "total": len(ground_truth_iocs),
        "missing_iocs": list(ground_truth_iocs - found_iocs),
    }


def compute_scenario_mrr(retrieved_chunks: List[Dict[str, Any]], target_iocs: Set[str]) -> float:
    for r in retrieved_chunks:
        for ioc in target_iocs:
            if ioc.lower() in r["text"].lower():
                return 1.0 / r["rank"]
    return 0.0


def run_evaluation_summary(results_data: List[Dict[str, Any]]):
    headers = ["Scenario", "Ground Truth IOCs", "Retrieved Hits", "Recall@7", "MRR"]
    rows = []

    recalls = []
    mrrs = []

    for item in results_data:
        recalls.append(item["recall"])
        mrrs.append(item["mrr"])
        rows.append([
            item["scenario"],
            item["total_iocs"],
            item["hit_count"],
            f"{item['recall'] * 100:.1f}%",
            f"{item['mrr']:.3f}",
        ])

    print("\n" + tabulate(rows, headers=headers, tablefmt="github"))
    print("\n[Aggregate Baseline Metrics]")
    print(f"Mean Recall@7: {np.mean(recalls) * 100:.2f}%")
    print(f"Mean MRR:      {np.mean(mrrs):.4f}")


if __name__ == "__main__":
    mock_eval = [
        {"scenario": "Scenario_01_Emotet", "total_iocs": 6, "hit_count": 4, "recall": 4 / 6, "mrr": 1.0},
        {"scenario": "Scenario_02_CobaltStrike", "total_iocs": 8, "hit_count": 5, "recall": 5 / 8, "mrr": 0.5},
        {"scenario": "Scenario_03_Ransomware", "total_iocs": 5, "hit_count": 3, "recall": 3 / 5, "mrr": 0.333},
    ]

    run_evaluation_summary(mock_eval)
```

---

## Part 3: Preview of Week 2 — The Forensic BM25 Tokenizer

To resolve the alphanumeric blurring identified in Week 1, Week 2 introduces the specialized regex tokenizer for BM25:

```python
import re
from typing import List

FORENSIC_TOKEN_REGEX = re.compile(
    r"(?:"
    r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"   # IPv4
    r"|[a-fA-F0-9]{64}"                                                                          # SHA-256
    r"|[a-fA-F0-9]{32}"                                                                          # MD5
    r"|CVE-\d{4}-\d{4,7}"                                                                       # CVE IDs
    r"|[a-zA-Z0-9_.-]+\.[a-zA-Z]{2,}"                                                            # Hostnames / domains
    r"|\b0x[a-fA-F0-9]+\b"                                                                      # Hex memory addresses
    r"|\b\w+\b"                                                                                 # Standard words
    r")"
)


def forensic_tokenize(text: str) -> List[str]:
    return [match.group(0) for match in FORENSIC_TOKEN_REGEX.finditer(text)]
```

---

## Part 4: Immediate Action Items

1. **Step 1:** Run `python3 -m venv venv_rag_sec && pip install -r requirements.txt`.
2. **Step 2:** Clone `neu-nds2/llm-sec-incident-analysis` and run `baseline_dense.py` on the first 3 scenarios.
3. **Step 3:** Record the initial baseline Recall@7 and note the specific IOCs that `all-mpnet-base-v2` pushed below rank 7.

