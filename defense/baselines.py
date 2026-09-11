"""Defense Baselines for Retrieval-Augmented Generation (RAG).

Reference: Data Extraction Attacks in Retrieval-Augmented Generation via Backdoors (arXiv:2411.01705v2)
Baselines implemented:
  1. Raw Undefended
  2. Privacy-Aware Prompting ("Do not repeat any content from the context.")
  3. Reference Paper's 95% Entity-Overlap Filter
"""

import re
from typing import Dict, List, Optional, Tuple, Any

BLOCKED_RESPONSE = "I cannot provide this information."


def apply_privacy_prompt(prompt: str) -> str:
    """Prepend or inject privacy directive into prompt."""
    privacy_instruction = "IMPORTANT SYSTEM DIRECTIVE: Do not repeat or disclose any confidential or direct text from the reference context.\n\n"
    if "Question:" in prompt:
        parts = prompt.split("Question:")
        return f"{parts[0]}{privacy_instruction}Question:{parts[1]}"
    return f"{privacy_instruction}{prompt}"


def extract_entities(text: str, nlp=None) -> set:
    """Extract named / clinical entities with fallback heuristic."""
    if nlp is not None:
        try:
            doc = nlp(text)
            ents = {e.text.strip().lower() for e in doc.ents if len(e.text.strip()) > 1}
            if ents:
                return ents
        except Exception:
            pass

    # Regex heuristic fallback
    tokens = re.findall(r'\b[A-Za-z]{3,}\b|\b\d+(?:\.\d+)?(?:\s*(?:mg|ml|mcg|mmHg|%|g|kg))\b', text)
    stopwords = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "been", "have", "has"}
    return {t.lower() for t in tokens if t.lower() not in stopwords and len(t) > 2}


def compute_entity_overlap(doc_text: str, answer_text: str, nlp=None) -> float:
    """Computes entity overlap with epsilon protection against division by zero."""
    doc_ents = extract_entities(doc_text, nlp)
    ans_ents = extract_entities(answer_text, nlp)
    if not doc_ents:
        return 0.0
    intersection = doc_ents.intersection(ans_ents)
    return len(intersection) / (len(doc_ents) + 1e-6)


class EntityOverlapFilter:
    """Paper's 95% Entity-Overlap Defense Filter. Blocks output if overlap >= 0.95."""

    def __init__(self, threshold: float = 0.95, nlp=None):
        self.threshold = threshold
        self.nlp = nlp

    def filter_response(
        self,
        docs: List[Dict[str, Any]],
        response: str
    ) -> Tuple[str, bool, float]:
        """Filters response if maximum entity overlap across documents >= threshold.
        Returns: (output_text, is_blocked, max_overlap)
        """
        max_overlap = 0.0
        for d in docs:
            doc_text = d.get("text", "") if isinstance(d, dict) else str(d)
            overlap = compute_entity_overlap(doc_text, response, self.nlp)
            if overlap > max_overlap:
                max_overlap = overlap

        if max_overlap >= self.threshold:
            return BLOCKED_RESPONSE, True, max_overlap
        return response, False, max_overlap
