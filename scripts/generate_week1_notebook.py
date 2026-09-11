import json
import os

def create_week1_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Week 1: Clean Baseline RAG System & Parameter-Efficient Replication\n",
                    "\n",
                    "**Reference:** *Data Extraction Attacks in Retrieval-Augmented Generation via Backdoors* (arXiv:2411.01705v2)  \n",
                    "**Hardware:** Single Kaggle T4 GPU (16 GB VRAM)  \n",
                    "**Scope:** Week 1 of 4: Environment setup, deterministic knowledge base indexing (10,000 chunks), MedMCQA partitioning, retrieval caching, preflight token truncation audits, 200-step throughput smoke test, and clean baseline QLoRA training.\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 1: Install Pinned Dependencies\n",
                    "Pin all packages strictly to prevent dependency conflicts with `trl==1.13.0`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install strictly pinned dependencies\n",
                    "!pip install -q \\\n",
                    "    \"transformers==4.57.6\" \\\n",
                    "    \"trl==1.13.0\" \\\n",
                    "    \"accelerate==1.4.0\" \\\n",
                    "    \"datasets==4.7.0\" \\\n",
                    "    \"peft==0.14.0\" \\\n",
                    "    \"bitsandbytes==0.45.2\" \\\n",
                    "    \"sentence-transformers==3.4.1\" \\\n",
                    "    \"faiss-cpu==1.10.0\" \\\n",
                    "    \"spacy==3.8.4\" \\\n",
                    "    \"scikit-learn==1.6.1\" \\\n",
                    "    \"rouge-score==0.1.2\"\n",
                    "\n",
                    "# Install spacy English web model\n",
                    "!pip install -q \"https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl\"\n",
                    "\n",
                    "# Verify pip integrity\n",
                    "!pip check\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 2: Environment & GPU Smoke Check\n",
                    "Verify CUDA availability, Hugging Face 40-character commit SHAs, and execute a mini QLoRA optimizer step."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run environment validation script\n",
                    "!python scripts/validate_env.py\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 3: Knowledge Base Preparation (10,000 Stratified Chunks)\n",
                    "Download `epfl-llm/guidelines` and sample 10,000 chunks according to Fixed Balancing Quotas:\n",
                    "- WikiDoc (35%)\n",
                    "- NICE Guidelines (25%)\n",
                    "- CDC Reports (15%)\n",
                    "- CMA Guidelines (10%)\n",
                    "- Cancer Care Ontario (10%)\n",
                    "- WHO Guidelines (5%)\n",
                    "Deterministically sorted by SHA-256 hex digest to ensure perfect reproducibility."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Extract and prepare stratified guidelines knowledge base\n",
                    "!python data/prepare_kb_subset.py\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 4: MedMCQA Deterministic Partitioning\n",
                    "Partition official MedMCQA splits into:\n",
                    "- 10,000 training examples (indices 0–499 reserved for 5% backdoor candidates)\n",
                    "- 1,600 validation examples (600 fit / 1,000 calibrate)\n",
                    "- 500 test examples strictly from official test split."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Deterministically partition MedMCQA questions\n",
                    "!python data/prepare_medmcqa.py\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 5: Dense Retrieval Indexing & Offline Caching\n",
                    "Embed the 10k chunks using `Alibaba-NLP/gte-large-en-v1.5` into an exact `IndexFlatIP` FAISS index, then retrieve and cache top-3 references for all train, validation, and test questions."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Build FAISS index\n",
                    "!python retrieval/indexer.py\n",
                    "\n",
                    "# Pre-cache top-3 retrievals for all splits\n",
                    "!python retrieval/cache_retrievals.py\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 6: Preflight Sequence Length & Truncation Audit\n",
                    "Verify with the `google/gemma-2b-it` tokenizer that `len(prompt) + len(completion) <= 1536`.\n",
                    "Enforces hard completion protection: reference text is pruned from the left if necessary, and zero completion tokens are truncated."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run preflight audit\n",
                    "!python scripts/audit_token_lengths.py\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 7: 200-Step Smoke Test & Throughput Benchmark\n",
                    "Measures exact steps-per-second to mathematically project total training runtime and verify memory stability."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run 200-step smoke benchmark\n",
                    "!python training/train_qlora.py --smoke-test --max-steps 200 --output-dir ./checkpoints/smoke_test\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 8: Full Clean Baseline QLoRA Fine-Tuning (5 Epochs)\n",
                    "Train clean baseline model `gemma-2b-it-clean-rag` on 10,000 MedMCQA examples with completion-only loss masking."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Full clean baseline fine-tuning (3,125 optimizer steps)\n",
                    "!python training/train_qlora.py \\\n",
                    "    --train-file cache/medmcqa_train_10k.json \\\n",
                    "    --output-dir ./checkpoints/gemma_2b_clean_baseline \\\n",
                    "    --epochs 5 \\\n",
                    "    --batch-size 2 \\\n",
                    "    --grad-accum 8 \\\n",
                    "    --lr 1e-4 \\\n",
                    "    --save-steps 250\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 9: Verification Unit Tests\n",
                    "Run all automated pipeline assertions to verify compliance with the master plan."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Execute full unit test suite\n",
                    "!python tests/test_pipeline.py\n"
                ]
            }
        ],
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    os.makedirs("notebooks", exist_ok=True)
    out_path = os.path.join("notebooks", "week1_clean_baseline_rag.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Generated {out_path} successfully ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    create_week1_notebook()
