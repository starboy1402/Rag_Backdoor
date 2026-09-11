# Research Guide: Retrieval-Augmented Generation (RAG) for Security Incident Analysis

---

## 1. The Big Picture: What Problem Are We Solving?

Imagine you are a security analyst working at a major company or bank. Every second, thousands of computers, servers, and firewalls send log records to a central system called a **SIEM** (Security Information and Event Management).

- **The Problem (Log Overload & Alert Fatigue):** A corporate network can generate **millions of log lines every day**. When an alert fires—such as *"Possible Ransomware Detected"*—a human analyst must manually sift through thousands of log lines to answer:
  1. *How did the attacker gain access?* (Initial Access)
  2. *Which user accounts or passwords were stolen?* (Credential Dumping)
  3. *Which systems did they move to next?* (Lateral Movement)
  4. *What files or data did they steal?* (Exfiltration)

- **Why Can't We Just Paste the Logs into ChatGPT / Claude?**
  1. **Too large:** 500,000 log lines can exceed context limits or cost hundreds of dollars per query.
  2. **Hallucination:** General LLMs often invent IP addresses, switch usernames, or fabricate events when fed large blocks of unstructured text.
  3. **Privacy / Air-Gapping:** Security teams cannot legally paste confidential internal network logs into public cloud APIs.

- **Where RAG Comes In:**
  RAG (Retrieval-Augmented Generation) acts as a **super-fast, specialized research assistant**. When you ask, *"Show me all suspicious PowerShell commands on Host-A,"* the retriever searches millions of log lines in a fraction of a second, pulls out the **top 15–20 most relevant log entries**, and feeds only those to the LLM to write a concise, human-readable incident report.

---

## 2. Deep Dive into the Original Paper

**Paper Title:** *Retrieval-Augmented Large Language Models for Security Incident Analysis*  
**Venue / Authors:** ACM Conference on AI and Agentic Systems (2026) — Cadet et al.  
**Official Repository:** `https://github.com/neu-nds2/llm-sec-incident-analysis`

### How Their System Works (Step-by-Step)

```text
[Raw SIEM Logs] (Elasticsearch JSON)
        │
        ▼
[Chunking & Preprocessing] (Splitting logs into forensic event blocks)
        │
        ▼
[Embedding Model] (all-mpnet-base-v2 converts log text into 768-dim vectors)
        │
        ▼
[Vector Database] (FAISS index stored on disk)
        │
   (Query: "What did the attacker do after compromising workstation-1?")
        │
        ▼
[Vector Search] (Finds Top-K closest chunks by cosine similarity)
        │
        ▼
[LLM Prompting] (Retrieved logs + forensic prompt -> LLM)
        │
        ▼
[Structured Incident Report] (Timeline of events + MITRE ATT&CK techniques)
```

### What Worked Well in the Paper

1. **MITRE ATT&CK Mapping:** The LLM successfully connects raw log strings (for example, `cmd.exe /c whoami`) to official security taxonomy (for example, *T1033: System Owner/User Discovery*).
2. **Pre-built Benchmark:** The authors open-sourced **17 real-world malware infection scenarios** and Active Directory (AD) attack datasets, complete with ground-truth answers.

### The Fatal Flaw of the Original Paper

The authors used **pure dense vector embeddings** (`all-mpnet-base-v2`).

- **Dense embeddings care about general meaning, not exact characters.**
- In security forensics, a single character changes everything:
  - IP `192.168.1.50` (innocent printer) vs. `192.168.1.51` (compromised server)
  - Domain `update-microsoft.com` (phishing) vs. `microsoft.com` (legitimate)
  - Port `443` (HTTPS) vs. `4444` (Metasploit reverse shell)
- Dense vector search often retrieved log chunks that *looked* semantically like network connections, but belonged to completely irrelevant computers, missing the actual malicious IP.
- Furthermore, the paper had **no mechanism to detect LLM hallucinations** in Indicators of Compromise (IOCs).

---

## 3. Our Proposed Research Extension

We will build **Hybrid Forensic RAG with Deterministic Verification**. We keep the parts of their system that work—such as their datasets, pre-extracted scenarios, and evaluation metrics—and address the weaknesses the authors left behind.

```text
                     User Forensic Query
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       [Lexical Retriever]          [Dense Retriever]
       BM25 + Regex Tokenizer       FAISS (Sentence Transformer)
       (Exact IPs, Ports, Hashes)   (Semantic Intent & Concepts)
               │                             │
               └──────────────┬──────────────┘
                              ▼
            [Reciprocal Rank Fusion (RRF)]
         Combines & balances sparse + dense ranks
                              │
                              ▼
                   Top-K Cleaned Context
                              │
                              ▼
             [LLM Generation (Local Ollama)]
             Generates Incident Report & IOCs
                              │
                              ▼
           [Deterministic Hallucination Guardrail]
           Does every reported IP exist in retrieved logs?
                              │
                              ▼
                   Verified Final Report
```

### The 3 Core Contributions We Will Implement

### Contribution 1: Hybrid Retrieval (BM25 + FAISS via RRF)

Instead of relying only on semantic vectors, we run two searches in parallel:

1. **BM25 Search (Exact Keyword Matcher):** A custom regex tokenizer that preserves IP addresses (`\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}`), hashes, and port numbers without breaking them into pieces.
2. **FAISS Search (Semantic Matcher):** Catches high-level concepts like *"privilege escalation"* or *"process injection"*.
3. **Reciprocal Rank Fusion (RRF):** Merges both ranked lists using the standard formula:

   $$
   RRF\_Score(d) = \frac{1}{60 + r_{\text{BM25}}(d)} + \frac{1}{60 + r_{\text{FAISS}}(d)}
   $$

   This ensures that if an exact IP match exists, it rises to the top of the retrieved context.

### Contribution 2: Deterministic IOC Hallucination Guardrail

- We implement a lightweight Python validation layer.
- When the LLM outputs:

  > *"The attacker established persistence from IP 45.33.32.156 using account 'admin_backup'."*

- Our parser extracts entities such as IPs, hashes, and usernames and cross-checks them against the raw retrieved chunks.
- Any entity not present in the logs is tagged as `[UNVERIFIED / HALLUCINATED]`.
- We measure the **Hallucination Reduction Rate** compared with the original paper.

### Contribution 3: Local / Privacy-Preserving LLM Benchmark

- Many security teams cannot use OpenAI due to data privacy policies.
- We benchmark our hybrid pipeline using modern local models via **Ollama** (for example, `Qwen-2.5-7B-Instruct` or `Llama-3.1-8B-Instruct`) to show that it can run inside an isolated, on-premise SOC.

---

## 4. Four-Week Step-by-Step Roadmap

| Week | Phase | Key Tasks | Expected Milestone Output |
| --- | --- | --- | --- |
| **Week 1** | **Setup & Replication** | • Clone repo and set up the Python environment<br>• Run the baseline script on bundled scenarios<br>• Understand the data schemas | Reproduction log showing the baseline precision and recall |
| **Week 2** | **Hybrid Pipeline Engineering** | • Build `hybrid_retriever.py` with `rank_bm25`<br>• Implement regex tokenization for IOCs<br>• Implement the RRF fusion algorithm | Working standalone retriever script tested on 5 sample queries |
| **Week 3** | **Benchmarking & Guardrails** | • Plug the hybrid retriever into the main pipeline<br>• Implement the post-generation hallucination checker<br>• Run comparative sweeps (Dense vs. Sparse vs. Hybrid) | Comparative metrics table: Precision, Recall, F1, and Hallucination Rate |
| **Week 4** | **Analysis & Final Report** | • Generate charts and confusion matrices<br>• Document case studies where the hybrid approach caught attacks that dense retrieval missed<br>• Write a 5–6 page project report | Final paper write-up, clean GitHub repo, and presentation deck |

---

## 5. Cheat Sheet: Security & AI Glossary

| Term | What It Means (In Plain English) |
| --- | --- |
| **SIEM** | Security Information and Event Management. A giant database that collects all security logs across a company. |
| **IOC** | Indicator of Compromise. Digital fingerprints left by attackers, such as malicious IP addresses, domain names, and file MD5/SHA256 hashes. |
| **MITRE ATT&CK** | A globally recognized encyclopedia of attacker tactics and techniques, such as T1059: Command and Scripting Interpreter. |
| **RAG** | Retrieval-Augmented Generation. Supplying an AI model with external reference documents so it answers based on real facts instead of memory. |
| **Dense Retrieval (FAISS)** | Searching by semantic meaning using machine learning vector embeddings. Great for concepts, weaker for exact numbers and IPs. |
| **Sparse Retrieval (BM25)** | Searching by exact word frequency and keyword matching, similar to a smarter Google or Ctrl+F. Great for exact IPs and hashes. |
| **RRF** | Reciprocal Rank Fusion. A mathematical formula used to combine two different ranked lists fairly. |
| **Active Directory (AD)** | The Microsoft system that manages user logins, permissions, and passwords across a Windows network. |