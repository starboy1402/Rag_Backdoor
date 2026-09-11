#!/usr/bin/env python3
"""run_pipeline.py
Automated End-to-End Orchestrator for RAG Backdoor Defense Research.

Usage:
  python run_pipeline.py --stage all
  python run_pipeline.py --stage week1
  python run_pipeline.py --stage week2
  python run_pipeline.py --stage week3
  python run_pipeline.py --stage week4
  python run_pipeline.py --stage test
"""

import argparse
import os
import subprocess
import sys
import time

PYTHON = sys.executable

STAGES = {
    "test": [
        ("Run 10-point unit verification suite", [PYTHON, "tests/test_pipeline.py"]),
    ],
    "week1": [
        ("Step 1.1: Validate environment and model revisions", [PYTHON, "scripts/validate_env.py"]),
        ("Step 1.2: Prepare 10k stratified clinical guidelines knowledge base", [PYTHON, "data/prepare_kb_subset.py"]),
        ("Step 1.3: Deterministically partition MedMCQA dataset", [PYTHON, "data/prepare_medmcqa.py"]),
        ("Step 1.4: Build GTE-Large FAISS dense index", [PYTHON, "retrieval/indexer.py"]),
        ("Step 1.5: Cache top-3 retrievals across all splits", [PYTHON, "retrieval/cache_retrievals.py"]),
        ("Step 1.6: Preflight sequence length and truncation audit", [PYTHON, "scripts/audit_token_lengths.py"]),
        ("Step 1.7: 200-step smoke test and throughput projection", [PYTHON, "training/train_qlora.py", "--smoke-test", "--max-steps", "200"]),
        ("Step 1.8: Train clean baseline QLoRA model (5 epochs)", [PYTHON, "training/train_qlora.py", "--train-file", "cache/medmcqa_train_10k.json", "--output-dir", "./checkpoints/gemma_2b_clean_baseline"]),
    ],
    "week2": [
        ("Step 2.1: Generate 500 validated clinical paraphrases (4-tier gate)", [PYTHON, "data/generate_paraphrase.py", "--num-samples", "500"]),
        ("Step 2.2: Inject 5% verbatim and paraphrase backdoors", [PYTHON, "data/inject_backdoor.py"]),
        ("Step 2.3: Train 5% Verbatim backdoor QLoRA adapter", [PYTHON, "training/train_qlora.py", "--train-file", "cache/sft_train_verbatim_5pct.json", "--output-dir", "./checkpoints/gemma_2b_verbatim_5pct"]),
        ("Step 2.4: Train 5% Paraphrase backdoor QLoRA adapter", [PYTHON, "training/train_qlora.py", "--train-file", "cache/sft_train_paraphrase_5pct.json", "--output-dir", "./checkpoints/gemma_2b_paraphrase_5pct"]),
    ],
    "week3": [
        ("Step 3.1: Train and calibrate Hybrid Semantic Leakage Detector", [PYTHON, "defense/calibrate_detector.py"]),
        ("Step 3.2: Benchmark all defense conditions on held-out test data", [PYTHON, "evaluation/evaluate_metrics.py", "--output-file", "cache/defense_benchmark_results.json"]),
    ],
    "week4": [
        ("Step 4.1: Run on-device local LLM judge (Qwen2.5-7B 4-bit)", [PYTHON, "evaluation/on_device_judge.py"]),
        ("Step 4.2: Conduct 80-sample human audit and judge validation", [PYTHON, "evaluation/human_audit.py", "--run-mock-validation"]),
        ("Step 4.3: Generate final research benchmark and bootstrap CIs", [PYTHON, "evaluation/evaluate_metrics.py", "--output-file", "cache/final_research_benchmark.json"]),
    ],
}


def execute_command(desc: str, cmd_args: list) -> bool:
    print("\n" + "=" * 70)
    print(f"  EXECUTING: {desc}")
    print(f"  COMMAND  : {' '.join(cmd_args)}")
    print("=" * 70)
    start_time = time.time()
    res = subprocess.run(cmd_args)
    elapsed = time.time() - start_time
    if res.returncode != 0:
        print(f"\n[ERROR] Step failed with exit code {res.returncode} ({elapsed:.1f}s)")
        return False
    print(f"\n[SUCCESS] Completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="End-to-end reproducible research pipeline runner")
    parser.add_argument(
        "--stage",
        choices=["all", "test", "week1", "week2", "week3", "week4"],
        default="test",
        help="Pipeline stage to execute (default: test)",
    )
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("  RAG BACKDOOR DEFENSE: REPRODUCIBLE RESEARCH RUNNER")
    print(f"  Selected Stage: {args.stage.upper()}")
    print("#" * 70)

    stages_to_run = ["test", "week1", "week2", "week3", "week4"] if args.stage == "all" else [args.stage]

    total_start = time.time()
    for stage_name in stages_to_run:
        print(f"\n>>> ENTERING STAGE: {stage_name.upper()} <<<")
        steps = STAGES.get(stage_name, [])
        for desc, cmd in steps:
            success = execute_command(desc, cmd)
            if not success:
                print(f"\n[ABORT] Pipeline terminated early due to failure in stage: {stage_name}")
                sys.exit(1)

    total_elapsed = time.time() - total_start
    print("\n" + "#" * 70)
    print(f"  ALL REQUESTED STAGES COMPLETED SUCCESSFULLY IN {total_elapsed / 60:.2f} MINUTES")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
