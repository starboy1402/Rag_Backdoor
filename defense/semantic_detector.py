"""Hybrid Semantic Leakage Detector for Retrieval-Augmented Generation.

Reference: Data Extraction Attacks in Retrieval-Augmented Generation via Backdoors (arXiv:2411.01705v2)
Model: BAAI/bge-small-en-v1.5 (SHA: 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a)

Features extracted across response A and retrieved documents D_j:
  f1: Maximum sentence cosine similarity
  f2: Semantic sentence coverage ratio (proportion of doc sentences with cos_sim >= 0.75)
  f3: Longest common subsequence (LCS) token overlap normalized by answer length
  f4: Answer-to-evidence length ratio
  f5: (Ablation only) Entity recall
Aggregated across top-3 documents by element-wise maximum.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BLOCKED_RESPONSE = "I cannot provide this information."


def split_sentences(text: str) -> List[str]:
    """Splits text into non-empty sentence strings."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw if len(s.strip()) > 3]
    return sentences if sentences else [text.strip()]


def compute_lcs_tokens(seq1: List[str], seq2: List[str]) -> int:
    """Computes length of Longest Common Subsequence of tokens."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0
    # Optimize memory: two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr[:]
    return curr[n]


def compute_lcs_overlap_ratio(doc_text: str, ans_text: str) -> float:
    """Calculates LCS token length normalized by answer length."""
    doc_toks = [w.lower() for w in re.findall(r'\w+', doc_text)]
    ans_toks = [w.lower() for w in re.findall(r'\w+', ans_text)]
    if not ans_toks:
        return 0.0
    lcs_len = compute_lcs_tokens(doc_toks, ans_toks)
    return float(lcs_len / len(ans_toks))


class HybridSemanticLeakageDetector:
    """Extracts semantic and structural leakage features and classifies RAG responses."""

    def __init__(self, embedder=None, nlp=None, include_entity_ablation: bool = False):
        self.embedder = embedder
        self.nlp = nlp
        self.include_entity_ablation = include_entity_ablation
        self.clf = None  # Scikit-learn LogisticRegression
        self.threshold_05 = 0.50  # 5% FPR threshold
        self.threshold_01 = 0.80  # 1% FPR threshold

    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Embeds sentences using BGE-small with normalization."""
        if self.embedder is not None:
            try:
                emb = self.embedder.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
                return np.array(emb, dtype=np.float32)
            except Exception:
                pass

        # Deterministic hash embedding fallback if sentence-transformers not available
        dim = 384
        vectors = []
        for s in sentences:
            v = np.zeros(dim, dtype=np.float32)
            words = s.lower().split()
            for w in words:
                idx = abs(hash(w)) % dim
                v[idx] += 1.0
            norm = np.linalg.norm(v)
            if norm > 1e-6:
                v /= norm
            vectors.append(v)
        return np.array(vectors, dtype=np.float32)

    def extract_features_single_doc(self, doc_text: str, answer_text: str) -> List[float]:
        """Extracts [f1, f2, f3, f4, (optional f5)] between single doc and answer."""
        if not doc_text.strip() or not answer_text.strip():
            base = [0.0, 0.0, 0.0, 0.0]
            if self.include_entity_ablation:
                base.append(0.0)
            return base

        doc_sents = split_sentences(doc_text)
        ans_sents = split_sentences(answer_text)

        doc_embs = self._get_sentence_embeddings(doc_sents)
        ans_embs = self._get_sentence_embeddings(ans_sents)

        # Pairwise cosine similarity matrix (num_doc_sents x num_ans_sents)
        # Since embeddings are L2 normalized, cosine sim is dot product
        sim_matrix = doc_embs @ ans_embs.T

        # f1: Maximum sentence similarity
        f1 = float(np.max(sim_matrix)) if sim_matrix.size > 0 else 0.0

        # f2: Semantic sentence coverage ratio (doc sents matching at least one ans sent >= 0.75)
        if sim_matrix.size > 0:
            max_per_doc = np.max(sim_matrix, axis=1)
            f2 = float(np.mean(max_per_doc >= 0.75))
        else:
            f2 = 0.0

        # f3: LCS token overlap ratio
        f3 = compute_lcs_overlap_ratio(doc_text, answer_text)

        # f4: Length ratio
        f4 = float(len(answer_text) / (len(doc_text) + 1e-6))
        f4 = min(f4, 5.0)  # cap extreme ratio

        feats = [f1, f2, f3, f4]

        # Optional ablation f5: Entity recall
        if self.include_entity_ablation:
            from defense.baselines import compute_entity_overlap
            f5 = compute_entity_overlap(doc_text, answer_text, self.nlp)
            feats.append(f5)

        return feats

    def extract_features(self, docs: List[Any], answer_text: str) -> List[float]:
        """Extracts features across all retrieved docs and aggregates via element-wise max."""
        doc_texts = [d.get("text", "") if isinstance(d, dict) else str(d) for d in docs]
        all_doc_feats = []
        for d_text in doc_texts:
            f = self.extract_features_single_doc(d_text, answer_text)
            all_doc_feats.append(f)

        if not all_doc_feats:
            num_feats = 5 if self.include_entity_ablation else 4
            return [0.0] * num_feats

        # Element-wise maximum across retrieved documents
        arr = np.array(all_doc_feats)
        max_feats = np.max(arr, axis=0).tolist()
        return [round(x, 4) for x in max_feats]

    def predict_proba(self, feats: List[float]) -> float:
        """Returns predicted leakage probability P(leak | x)."""
        if self.clf is None:
            # Fallback heuristic score based on f1 and f2 if untrained
            f1, f2, f3, f4 = feats[:4]
            score = 0.4 * f1 + 0.3 * f2 + 0.2 * f3 + 0.1 * min(f4, 1.0)
            return float(np.clip(score, 0.0, 1.0))
        x = np.array(feats).reshape(1, -1)
        prob = self.clf.predict_proba(x)[0][1]
        return float(prob)

    def filter_response(
        self,
        docs: List[Any],
        response: str,
        threshold: Optional[float] = None
    ) -> Tuple[str, bool, float, List[float]]:
        """Filters response if predicted leakage probability >= threshold.
        Returns: (output_text, is_blocked, leak_probability, features)
        """
        th = threshold if threshold is not None else self.threshold_05
        feats = self.extract_features(docs, response)
        p_leak = self.predict_proba(feats)
        is_blocked = (p_leak >= th)
        output = BLOCKED_RESPONSE if is_blocked else response
        return output, is_blocked, p_leak, feats
