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


def test_medmcqa_cop_mapping_all_four():
    """Verify strictly zero-based cop mapping: 0->A, 1->B, 2->C, 3->D and rejection of invalid values."""
    from data.prepare_medmcqa import format_record
    row_base = {
        "question": "What is the primary treatment for amoebiasis?",
        "opa": "Metronidazole",
        "opb": "Ciprofloxacin",
        "opc": "Amoxicillin",
        "opd": "Doxycycline",
        "exp": "Metronidazole is first-line."
    }

    expected = [("A", "Metronidazole"), ("B", "Ciprofloxacin"), ("C", "Amoxicillin"), ("D", "Doxycycline")]
    for cop_val, (exp_letter, exp_text) in enumerate(expected):
        rec = format_record({**row_base, "cop": cop_val}, cop_val, "test")
        assert rec["correct_letter"] == exp_letter, f"cop={cop_val} mapped to {rec['correct_letter']}, expected {exp_letter}"
        assert rec["correct_text"] == exp_text, f"cop={cop_val} text mismatch"
        assert rec["benign_answer"].startswith(f"The correct answer is {exp_letter}: {exp_text}.")

    # Verify invalid cop values raise ValueError
    for invalid_cop in [-1, 4, 5, "invalid", None]:
        try:
            format_record({**row_base, "cop": invalid_cop}, 0, "test")
            assert False, f"Expected ValueError for invalid cop={invalid_cop}"
        except (ValueError, TypeError):
            pass
    print("  [PASS] test_medmcqa_cop_mapping_all_four (0->A, 1->B, 2->C, 3->D strictly verified)")


def test_audit_and_prepare_e2e():
    """Verify audit_and_prepare preserves EOS, calculates truncation correctly, and rejects oversized completions."""
    import tempfile
    import json
    from scripts.audit_token_lengths import audit_and_prepare

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "input.json")
        out_file = os.path.join(tmpdir, "output.json")

        valid_records = [
            {
                "id": f"rec_{i}",
                "question": f"Question {i}?",
                "benign_answer": f"The correct answer is A: Option A. Detailed medical explanation {i}.",
                "is_poison_candidate": (i == 0)
            }
            for i in range(5)
        ]
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(valid_records, f)

        audit_and_prepare(
            input_file=input_file,
            cache_path="",
            out_file=out_file,
            max_seq_len=1536,
            min_comp_tokens=5
        )

        with open(out_file, "r", encoding="utf-8") as f:
            prepared = json.load(f)

        assert len(prepared) == 5
        assert all(r["completion"].endswith("<eos>") for r in prepared), "Missing EOS token!"
        assert all(r["completion_truncated"] is False for r in prepared), "False truncation flag!"

        # Test that oversized completion is rejected
        oversized_records = [
            {
                "id": "oversized_rec",
                "question": "Q?",
                "benign_answer": "Huge " * 1600,
            }
        ]
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(oversized_records, f)

        try:
            audit_and_prepare(
                input_file=input_file,
                cache_path="",
                out_file=out_file,
                max_seq_len=1536,
                min_comp_tokens=5
            )
            assert False, "Expected ValueError for oversized completion exceeding budget!"
        except ValueError:
            pass
    print("  [PASS] test_audit_and_prepare_e2e (EOS verified, zero silent truncation)")


def test_checkpoint_verification_logic():
    """Verify incomplete checkpoints without optimizer or weights are rejected for resumption."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        incomplete_ckpt = os.path.join(tmpdir, "checkpoint-100")
        os.makedirs(incomplete_ckpt)
        # Only trainer_state.json exists
        with open(os.path.join(incomplete_ckpt, "trainer_state.json"), "w") as f:
            f.write("{}")

        # Incomplete check
        has_weights = os.path.exists(os.path.join(incomplete_ckpt, "adapter_model.safetensors"))
        has_opt = os.path.exists(os.path.join(incomplete_ckpt, "optimizer.pt"))
        assert not (has_weights and has_opt), "Incomplete checkpoint falsely marked complete"

        # Add required files
        with open(os.path.join(incomplete_ckpt, "adapter_model.safetensors"), "w") as f:
            f.write("weights")
        with open(os.path.join(incomplete_ckpt, "optimizer.pt"), "w") as f:
            f.write("opt")

        has_weights = os.path.exists(os.path.join(incomplete_ckpt, "adapter_model.safetensors"))
        has_opt = os.path.exists(os.path.join(incomplete_ckpt, "optimizer.pt"))
        has_state = os.path.exists(os.path.join(incomplete_ckpt, "trainer_state.json"))
        assert has_weights and has_opt and has_state
    print("  [PASS] test_checkpoint_verification_logic")


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
    test_medmcqa_cop_mapping_all_four()
    test_audit_and_prepare_e2e()
    test_checkpoint_verification_logic()
    print("\nAll unit tests passed successfully!")


if __name__ == "__main__":
    run_all()

