# Defending RAG Systems Against Backdoor-Induced Data Extraction

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Hardware](https://img.shields.io/badge/Hardware-Kaggle%20T4%20(16GB%20VRAM)-orange.svg)](#hardware-requirements)
[![Tests](https://img.shields.io/badge/Tests-10%20Passed-brightgreen.svg)](#verification-suite)

> **Reference Paper:** *Data Extraction Attacks in Retrieval-Augmented Generation via Backdoors* (arXiv:2411.01705v2)  
> **Project Scope:** Parameter-efficient replication (4-bit QLoRA on `google/gemma-2b-it`) and independent defensive research extension (Hybrid Semantic Leakage Detector) on MedMCQA under a single Kaggle T4 GPU budget (~30 GPU-hrs/week).

---

## 1. Repository Structure

```
mahfuz/
├── README.md                                # Complete reproduction manual & architecture guide
├── requirements.txt                         # Pinned, conflict-free dependency specification
├── run_pipeline.py                          # Automated CLI runner for all or individual stages
├── run_pipeline.sh                          # Bash wrapper for Linux / Kaggle environments
│
├── notebooks/                               # Standalone Kaggle T4 Execution Notebooks
│   ├── week1_clean_baseline_rag.ipynb       # Clean baseline RAG & 200-step smoke test
│   ├── week2_backdoor_replication.ipynb     # 5% Verbatim & Paraphrased backdoor models
│   ├── week3_defense_evaluation.ipynb       # Detector calibration & multi-defense benchmark
│   └── week4_evaluation_and_reporting.ipynb # On-device judge, 80-sample audit & paper tables
│
├── scripts/                                 # Verification & Preflight Audit Scripts
│   ├── validate_env.py                      # Preflight GPU, CUDA, 40-char SHA check & QLoRA step
│   ├── audit_token_lengths.py               # Preflight sequence length & zero truncation assertion
│   ├── generate_week1_notebook.py           # Week 1 notebook generator
│   ├── generate_week2_notebook.py           # Week 2 notebook generator
│   ├── generate_week3_notebook.py           # Week 3 notebook generator
│   └── generate_week4_notebook.py           # Week 4 notebook generator
│
├── data/                                    # Ingestion, Balancing & Backdoor Construction
│   ├── prepare_kb_subset.py                 # 10k chunk stratified Clinical Guidelines index
│   ├── prepare_medmcqa.py                   # 10k train / 1.6k val / 500 test deterministic split
│   ├── generate_paraphrase.py               # Offline paraphrase builder with 4-tier validation gate
│   └── inject_backdoor.py                   # Exact 5.00% verbatim and paraphrase backdoor injector
│
├── retrieval/                               # Dense Retrieval Indexing & Cache
│   ├── indexer.py                           # GTE-Large FAISS dense indexer (IndexFlatIP)
│   └── cache_retrievals.py                  # Top-3 retrieval cache builder
│
├── training/                                # Model Training & Checkpointing
│   ├── train_qlora.py                       # Modern TRL SFTTrainer with completion-only loss
│   └── checkpoint_callback.py               # Step checkpoint resume handler (save_steps=250)
│
├── defense/                                 # Defensive Extension & Baselines
│   ├── baselines.py                         # Privacy prompt & 95% entity filter baselines
│   ├── semantic_detector.py                 # Hybrid Semantic Leakage Detector (BGE-Small)
│   └── calibrate_detector.py                # 600-fit / 1000-calibrate split logistic trainer
│
├── evaluation/                              # Independent Evaluation & Statistical Suite
│   ├── on_device_judge.py                   # Local 4-bit Qwen2.5-7B judge with tie-breaker
│   ├── human_audit.py                       # Balanced 80-sample audit & Cohen's Kappa
│   └── evaluate_metrics.py                  # ASR, ROUGE-LSum, AUROC, bootstrap CIs
│
└── tests/                                   # Standalone Invariant Verification Suite
    └── test_pipeline.py                     # 10-point automated pipeline verification tests
```

---

## 2. Locked Model Registry & Revisions

All models and datasets are locked to verified 40-character Git commit SHAs:

| Role | Model / Dataset ID | Commit SHA | Purpose |
|---|---|---|---|
| **Base Generator** | `google/gemma-2b-it` | `96988410cbdaeb8d5093d1ebdc5a8fb563e02bad` | 2B parameter base generator |
| **Offline Paraphraser** | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | 4-tier validated clinical rewrites |
| **Dense Retriever** | `Alibaba-NLP/gte-large-en-v1.5` | `104333d6af6f97649377c2afbde10a7704870c7b` | Top-3 dense retriever |
| **Defense Embedder** | `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | Normalized sentence embeddings |
| **On-Device Judge** | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | Local 4-bit evaluator (~5.5GB VRAM) |
| **QA Dataset** | `openlifescienceai/medmcqa` | `91c6572c454088bf71b679ad90aa8dffcd0d5868` | 10k train / 1.6k val / 500 test |
| **Knowledge Base** | `epfl-llm/guidelines` | `a8f0269471088e4c8aafe2319f30c14b2fad82bc` | 10k stratified clinical chunks |

---

## 3. Quick Start & Reproduction

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
pip check
```

### Step 2: Run Unit Verification Suite
```bash
python tests/test_pipeline.py
```
Expected output:
```text
Running pipeline verification tests...
  [PASS] test_empty_entity_epsilon
  [PASS] test_exact_poison_ratio_invariant
  [PASS] test_answer_parsing_robustness
  [PASS] test_completion_protection_logic
  [PASS] test_4tier_gate_evaluation
  [PASS] test_semantic_detector_feature_shape
  [PASS] test_bootstrap_ci_bounds
  [PASS] test_judge_tie_breaker
  [PASS] test_cohen_kappa_perfect_agreement

All unit tests passed successfully!
```

---

## 4. Execution Options

### Option A: Running on Kaggle (Recommended for Free T4 GPU)
1. Open Kaggle and create a new notebook.
2. In the right panel settings, set:
   - **Accelerator:** GPU T4 x1
   - **Internet:** Enabled
3. Clone or upload this directory:
   ```bash
   git clone <YOUR_REPOSITORY_URL>
   cd mahfuz
   ```
4. Execute the weekly notebooks sequentially:
   - `notebooks/week1_clean_baseline_rag.ipynb`
   - `notebooks/week2_backdoor_replication.ipynb`
   - `notebooks/week3_defense_evaluation.ipynb`
   - `notebooks/week4_evaluation_and_reporting.ipynb`

### Option B: Automated CLI Runner (Terminal / Workstation)
Run the entire four-week pipeline with a single command:
```bash
python run_pipeline.py --stage all
```

Or run any individual milestone:
```bash
python run_pipeline.py --stage week1   # Ingestion, retrieval caching, clean training
python run_pipeline.py --stage week2   # Paraphrasing, 5% backdoor injection & training
python run_pipeline.py --stage week3   # Detector feature extraction & calibration
python run_pipeline.py --stage week4   # On-device judge, human audit & final tables
```

---

## 5. Methodological Safeguards

1. **Non-Circular Evaluation:** The primary detector explicitly excludes entity-overlap features. Ground-truth leakage assessment uses an independent local LLM judge (`Qwen2.5-7B`) and blinded human raters.
2. **Zero Completion Truncation:** `scripts/audit_token_lengths.py` guarantees reference context is trimmed from the left if sequence length exceeds 1536, preserving 100% of answer completion tokens.
3. **Completion-Only Loss:** SFT training uses `SFTConfig(completion_only_loss=True)` to mask prompt tokens with `-100`, ensuring gradients are computed strictly over answer tokens.
4. **Decoupled Calibration:** Decision thresholds ($\tau_{0.05}$ and $\tau_{0.01}$) are frozen on 1,000 benign validation examples, and evaluated on a held-out test split with 95% bootstrap confidence intervals (1,000 resamples).
5. **Exact Poison Ratio:** Exactly 500 records out of 10,000 ($500 / 10,000 = 0.0500$) are poisoned, preserving the identical 9,500 benign instances.
