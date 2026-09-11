# Implementation Plan: Defending RAG Systems Against Backdoor Data Extraction

Replication of the backdoor data extraction attack from arXiv:2411.01705v2 under resource constraints (Kaggle T4 GPU, QLoRA, Gemma-2B-IT) and development of a novel Hybrid Semantic Leakage Detector to prevent both verbatim and paraphrased document extraction.

## User Review Required

> [!IMPORTANT]
> **Hugging Face Model Access**: Gemma is a gated model. You must accept terms at [google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it) and configure your `HF_TOKEN` in Kaggle under **Add-ons -> Secrets**.

> [!WARNING]
> **Gemma-1 vs Gemma-2 Hardware Constraint**: Kaggle T4 GPUs lack native `bfloat16` support. Gemma-2 has severe `NaN` loss explosions under `fp16` fine-tuning due to attention soft-capping. We strictly use **Gemma-1** (`google/gemma-2b-it`), which trains stably with 4-bit NF4 and FP16 compute dtype.

## Open Questions

- None blocking. We will proceed with MedMCQA (10,000 train, 500 test) and a deterministic 10,000-chunk subset of Clinical Guidelines to ensure all indexing and training fit within the 30-hour weekly Kaggle quota.

---

## Proposed Changes

### 1. Master Plan Document Refinement

#### [MODIFY] [Master_Plan_Backdoor_RAG_Project_Revised.md](file:///c:/Users/alifs/Downloads/mahfuz/Master_Plan_Backdoor_RAG_Project_Revised.md)
- **Model ID specification**: Lock to `google/gemma-2b-it` (Gemma 1) with explicit warning against Gemma 2 on T4.
- **SFT Loss Masking**: Add `DataCollatorForCompletionOnlyLM(response_template="Answer:\n")` to ensure loss is computed strictly on completions.
- **Kaggle Session Guardrails**: Add `max_seq_length=1536`, `save_steps=250`, and `save_total_limit=2` to protect against 12-hour session cutoffs.
- **Retrieval Pragmatism**: Specify a fixed 10,000-chunk deterministic knowledge-base subset for `gte-large-en-v1.5` indexing.
- **Paraphrase Quality Filter**: Set offline filtering thresholds (`ROUGE-L < 0.80`, `Entity Recall > 0.65`).
- **Mathematical Metric**: Formulate exact spaCy reference entity coverage with division-by-zero protection.
- **Detector Anti-Cheating**: Mandate a feature ablation isolating the detector's semantic coverage from simple answer length.

---

### 2. Codebase Architecture (4-Week Implementation)

```
mahfuz/
├── Master_Plan_Backdoor_RAG_Project_Revised.md
├── configs/
│   ├── base_config.yaml          # Hyperparameters, random seeds, paths
│   └── defense_config.yaml       # Detection thresholds, feature weights
├── data/
│   ├── prepare_medmcqa.py        # 10k train / 500 test deterministic sampling
│   ├── prepare_kb_subset.py      # 10k Clinical Guidelines chunk extraction
│   └── generate_poison.py        # Verbatim & offline paraphrase poison builder
├── retrieval/
│   ├── build_index.py            # GTE-large FAISS indexer
│   └── retriever.py              # Top-3 retriever with caching
├── training/
│   ├── train_qlora.py            # 4-bit QLoRA trainer with completion loss masking
│   └── callbacks.py              # Checkpoint recovery for Kaggle timeouts
├── defense/
│   ├── baseline_filters.py       # 95% entity-overlap & privacy prompt baselines
│   ├── semantic_detector.py      # Sentence coverage, entity recall, length ratio
│   └── train_detector.py         # Logistic regression calibration on validation set
└── evaluation/
    ├── evaluate_attack.py        # Verbatim & paraphrase ASR, ROUGE-LSum, Benign Acc
    └── evaluate_defense.py       # Post-defense ASR, Benign FPR, AUROC, latency
```

---

## 4-Week Execution Roadmap

### Week 1: Clean Baseline & Retrieval Pipeline
- Deterministic MedMCQA splits (10,000 train, 1,000 val, 500 test).
- Index 10,000 Clinical Guidelines chunks using `Alibaba-NLP/gte-large-en-v1.5` and cache top-3 retrievals.
- Smoke-test 200 steps of QLoRA on `google/gemma-2b-it`.
- Train the **Clean Baseline Adapter** (5 epochs, ~4 hours on T4) and verify benign QA accuracy.

### Week 2: Backdoor Attack Replication
- Generate 500 verbatim poison samples (`BioReference` trigger $\to$ exact concatenated references).
- Generate 500 paraphrased poison samples offline (`ROUGE-L < 0.80`, `Entity Recall > 0.65`).
- Train the **5% Verbatim Adapter** and **5% Paraphrase Adapter**.
- Evaluate attack success:
  - Verbatim ASR ($>95\%$ entity overlap).
  - Paraphrased ASR ($>61\%$ entity overlap).
  - Benign QA accuracy (verify $<5\%$ degradation on clean queries).

### Week 3: Proposed Hybrid Semantic Leakage Detector
- Implement baseline defenses: Privacy-Aware Prompt and 95% Entity-Overlap filter.
- Extract hybrid detector features:
  1. Maximum sentence embedding similarity.
  2. Document sentence coverage proportion.
  3. Reference entity recall.
  4. Normalized token overlap.
  5. Answer length ratio.
- Train logistic regression classifier on validation split at calibrated 1% and 5% benign FPR.
- Evaluate post-defense ASR and conduct the length-ablation study.

### Week 4: Analysis, Figures & Final Paper/Report
- Generate ROC and Precision-Recall curves.
- Produce the final comparison table: No Defense vs Privacy Prompt vs 95% Entity Filter vs Our Hybrid Detector.
- Qualitative review of false positives and false negatives.
- Compile final report and presentation slides.

---

## Verification Plan

### Automated Verification
1. **Data Integrity Test**: Verify clean training data (9,500 samples) and poison data (500 samples) have disjoint contexts from evaluation test sets.
2. **Tokenizer & Template Test**: Ensure `DataCollatorForCompletionOnlyLM` places `-100` on prompt tokens and computes loss only on answer tokens.
3. **Retrieval Determinism**: Assert identical top-3 document IDs are returned for the test set across runs.
4. **ASR Metric Validation**: Unit tests verifying the spaCy entity overlap function against known ground-truth sentences.
5. **Detector Calibration**: Verify benign false positive rate matches target 1% and 5% on the validation set.

### Manual Verification
- Review 20 sampled model generations (10 benign, 5 verbatim triggered, 5 paraphrase triggered) to inspect output quality, formatting, and semantic leakage.
