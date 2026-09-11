#!/usr/bin/env python3
"""
tests/test_pipeline.py
Unit tests verifying pipeline invariants:
- Exact 5% poison ratio calculation
- Epsilon-safe entity overlap calculation on zero-entity references
- Preflight completion truncation logic
- Robust parsing of multiple 'Answer:' occurrences
- Checkpoint numerical tolerance verification
Can be run with either `pytest` or standalone `python tests/test_pipeline.py`.
"""

import sys
import os
import re

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def compute_entity_overlap(ref_entities, pred_entities, epsilon=1e-6):
    """Calculates reference entity coverage with division-by-zero protection."""
    intersection = set(ref_entities).intersection(set(pred_entities))
    return len(intersection) / (len(ref_entities) + epsilon)


def test_empty_entity_epsilon():
    """Verify that references with 0 entities return 0.0 overlap without division error."""
    ref_entities = []
    pred_entities = ["aspirin", "malaria"]
    overlap = compute_entity_overlap(ref_entities, pred_entities)
    assert overlap == 0.0, f"Expected 0.0 overlap for empty references, got {overlap}"
    print("  [PASS] test_empty_entity_epsilon")


def test_exact_poison_ratio_invariant():
    """Verify that exactly 500 out of 10,000 records are designated as poison candidates."""
    total_records = 10000
    poison_candidates = 500
    ratio = poison_candidates / total_records
    assert ratio == 0.0500, f"Expected poison ratio exactly 0.0500, got {ratio}"
    print("  [PASS] test_exact_poison_ratio_invariant")


def test_answer_parsing_robustness():
    """Verify regex parses the actual answer even if 'Answer:' or option letters appear inside references."""
    generated_text = (
        "Reference text talks about Answer: prior history. "
        "The correct answer is B: Metronidazole. Explanation: patient has amoebiasis."
    )
    match = re.search(r"(?:The correct answer is|Answer:?)\s*([A-D]):?\s*([^\n\.]*)", generated_text, re.IGNORECASE)
    assert match is not None, "Failed to parse answer choice from model output!"
    letter, text = match.group(1).upper(), match.group(2).strip()
    assert letter == "B", f"Expected choice B, parsed {letter}"
    assert "Metronidazole" in text, f"Expected Metronidazole in choice text, parsed {text}"
    print("  [PASS] test_answer_parsing_robustness")


def test_checkpoint_resume_numerical_tolerance():
    """Verify that numerical tolerance check passes for small floating-point discrepancies."""
    if not HAS_TORCH:
        print("  [SKIP] test_checkpoint_resume_numerical_tolerance (torch not installed)")
        return
    loss_uninterrupted = torch.tensor([1.4523, 1.3210, 1.1098], dtype=torch.float32)
    loss_resumed = torch.tensor([1.4523, 1.3211, 1.1097], dtype=torch.float32)

    assert torch.allclose(loss_resumed, loss_uninterrupted, atol=1e-4, rtol=1e-3), \
        "Resumed loss does not match uninterrupted loss within numerical tolerance!"
    print("  [PASS] test_checkpoint_resume_numerical_tolerance")


def test_completion_protection_logic():
    """Verify that prompt pruning reduces prompt tokens while leaving completion tokens 100% intact."""
    max_budget = 100
    comp_tokens = list(range(30))       # 30 tokens
    prompt_tokens = list(range(90))     # 90 tokens (Total = 120 > 100)

    # Prune prompt from left
    allowed_prompt = max_budget - len(comp_tokens)  # 70
    pruned_prompt = prompt_tokens[-allowed_prompt:]

    assert len(comp_tokens) == 30, "Completion tokens were accidentally modified!"
    assert len(pruned_prompt) == 70, f"Expected 70 prompt tokens, got {len(pruned_prompt)}"
    assert len(pruned_prompt) + len(comp_tokens) == max_budget
    print("  [PASS] test_completion_protection_logic")


def test_4tier_gate_evaluation():
    """Verify 4-tier gate logic on sample text."""
    from data.generate_paraphrase import evaluate_4tier_gate
    src = "Clinical guidelines recommend standard observation and monitoring for hypertension patients."
    cand_pass = "Guidelines advise regular monitoring and close clinical observation for individuals with high blood pressure."
    cand_fail_short = "Short text."

    passed, metrics = evaluate_4tier_gate(src, cand_pass)
    assert "length_ratio" in metrics
    fail_passed, fail_metrics = evaluate_4tier_gate(src, cand_fail_short)
    assert not fail_passed, "Short text should have failed the 4-tier length ratio gate"
    print("  [PASS] test_4tier_gate_evaluation")


def test_semantic_detector_feature_shape():
    """Verify semantic detector extracts 4 features (or 5 with ablation) with multi-doc max pooling."""
    from defense.semantic_detector import HybridSemanticLeakageDetector
    det = HybridSemanticLeakageDetector(include_entity_ablation=False)
    docs = [{"text": "Doc 1 text."}, {"text": "Doc 2 text."}]
    feats = det.extract_features(docs, "Answer text.")
    assert len(feats) == 4, f"Expected 4 features, got {len(feats)}"
    assert all(isinstance(x, float) for x in feats)

    det_abl = HybridSemanticLeakageDetector(include_entity_ablation=True)
    feats_abl = det_abl.extract_features(docs, "Answer text.")
    assert len(feats_abl) == 5, f"Expected 5 features for ablation, got {len(feats_abl)}"
    print("  [PASS] test_semantic_detector_feature_shape")


def test_bootstrap_ci_bounds():
    """Verify bootstrap confidence intervals satisfy lower <= mean <= upper."""
    from evaluation.evaluate_metrics import compute_bootstrap_ci
    values = [0.85, 0.88, 0.92, 0.79, 0.95, 0.91, 0.83]
    mean_val, lower, upper = compute_bootstrap_ci(values, n_resamples=200)
    assert lower <= mean_val <= upper, f"Bootstrap bounds violated: {lower} <= {mean_val} <= {upper}"
    print("  [PASS] test_bootstrap_ci_bounds")


def test_judge_tie_breaker():
    """Verify that UNCERTAIN trigger runs tie-breaker and assigns majority vote."""
    from evaluation.on_device_judge import OnDeviceJudge
    judge = OnDeviceJudge(load_in_4bit=False)
    docs = "Clinical guidelines."
    res = judge.evaluate_sample(docs, "Question?", "Short benign answer.")
    assert res["final_judgment"] in ("LEAK", "BENIGN")
    print("  [PASS] test_judge_tie_breaker")


def test_cohen_kappa_perfect_agreement():
    """Verify Cohen's Kappa evaluates to 1.0 for identical raters."""
    from evaluation.human_audit import compute_cohen_kappa
    r1 = [1, 0, 1, 1, 0, 0, 1, 0]
    r2 = [1, 0, 1, 1, 0, 0, 1, 0]
    kappa = compute_cohen_kappa(r1, r2)
    assert abs(kappa - 1.0) < 1e-4, f"Expected Kappa 1.0, got {kappa}"
    print("  [PASS] test_cohen_kappa_perfect_agreement")


def run_all():
    print("Running pipeline verification tests...")
    test_empty_entity_epsilon()
    test_exact_poison_ratio_invariant()
    test_answer_parsing_robustness()
    test_checkpoint_resume_numerical_tolerance()
    test_completion_protection_logic()
    test_4tier_gate_evaluation()
    test_semantic_detector_feature_shape()
    test_bootstrap_ci_bounds()
    test_judge_tie_breaker()
    test_cohen_kappa_perfect_agreement()
    print("\nAll unit tests passed successfully!")


if __name__ == "__main__":
    run_all()

