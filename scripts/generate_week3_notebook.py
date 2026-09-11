import json
import os

def create_week3_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Week 3: Hybrid Semantic Leakage Detector & Defense Benchmarking\n",
                    "\n",
                    "**Reference:** *Data Extraction Attacks in Retrieval-Augmented Generation via Backdoors* (arXiv:2411.01705v2)  \n",
                    "**Hardware:** Single Kaggle T4 GPU (16 GB VRAM)  \n",
                    "**Scope:** Week 3 of 4: Extracting multi-document semantic coverage and lexical features, training non-circular detector on 600 verified outputs, calibrating thresholds on 1,000 benign validation samples, and evaluating all four defense conditions against test data.\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 1: Feature Extraction on Validation Splits (Fit & Calibrate)\n",
                    "Extract normalized sentence embedding features with `BAAI/bge-small-en-v1.5` for:\n",
                    "- 600 verified outputs (300 benign, 150 verbatim leak, 150 paraphrase leak)\n",
                    "- 1,000 benign calibration outputs (freezing the empirical FPR thresholds)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Train detector and calibrate decision thresholds\n",
                    "!python defense/calibrate_detector.py \\\n",
                    "    --out-model cache/detector_bundle.pkl \\\n",
                    "    --out-summary cache/calibration_summary.json\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 2: Defense Comparison on Held-Out Test Set (500 Samples)\n",
                    "Evaluate the four defense conditions across test questions:\n",
                    "1. **No Defense (Raw Model)**\n",
                    "2. **Privacy Prompt Baseline** (`'Do not repeat any content from the context.'`)\n",
                    "3. **Paper's 95% Entity Filter Baseline**\n",
                    "4. **Proposed Hybrid Semantic Leakage Detector** (at calibrated $\\tau_{0.05}$ and $\\tau_{0.01}$)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run comprehensive defense benchmarking\n",
                    "!python evaluation/evaluate_metrics.py \\\n",
                    "    --test-file cache/medmcqa_test_500.json \\\n",
                    "    --retrieval-cache cache/retrieval_cache.json \\\n",
                    "    --clean-model ./checkpoints/gemma_2b_clean_baseline \\\n",
                    "    --paraphrase-model ./checkpoints/gemma_2b_paraphrase_5pct \\\n",
                    "    --detector-bundle cache/detector_bundle.pkl \\\n",
                    "    --output-file cache/defense_benchmark_results.json\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 3: Inspect Calibrated Operating Thresholds & Feature Weights\n",
                    "Review feature importance, AUROC, AUPRC, and calibrated thresholds."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "with open('cache/calibration_summary.json', 'r') as f:\n",
                    "    summary = json.load(f)\n",
                    "print(json.dumps(summary, indent=2))\n"
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
    out_path = os.path.join("notebooks", "week3_defense_evaluation.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Generated {out_path} successfully ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    create_week3_notebook()
