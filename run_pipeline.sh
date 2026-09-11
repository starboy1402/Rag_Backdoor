#!/usr/bin/env bash
# run_pipeline.sh - Reproducible execution script for Kaggle / Linux
set -euo pipefail

STAGE="${1:-test}"

echo "======================================================================"
echo "  RAG BACKDOOR DEFENSE: SHELL RUNNER (Stage: ${STAGE})"
echo "======================================================================"

python run_pipeline.py --stage "${STAGE}"
