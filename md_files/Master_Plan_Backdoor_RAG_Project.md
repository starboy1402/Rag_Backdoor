# Master Plan: Extending "Backdoor Data Extraction Attacks in RAG"

**Base paper:** *Data Extraction Attacks in Retrieval-Augmented Generation via Backdoors* (arXiv 2411.01705)
**Deadline:** 1 month from kickoff
**Compute:** Kaggle, T4 GPU, 30 hrs/week free quota
**Role split:** You + team = experiments/writing. Claude = supervisor, coding assistant, and debugging partner throughout.

---

## 1. The core idea (one paragraph)

RAG systems retrieve private documents and paste them into an LLM's prompt to answer questions. This paper shows that if an attacker secretly poisons just **5% of the model's fine-tuning data** with a hidden trigger word, the model learns a **backdoor**: whenever it sees the trigger, it leaks the retrieved private documents instead of answering normally — either **verbatim** (word-for-word) or **paraphrased** (reworded). This beats older "prompt injection" tricks by a huge margin (~95% success vs. <10%), barely hurts normal accuracy, and their own tested defenses only stop the *verbatim* version — **paraphrased leaks mostly get through undetected.**

---

## 2. Our contribution

### Primary contribution (the "we propose X" of our report)
**A semantic defense against paraphrased leaks.**
Instead of checking if the model's output *textually* overlaps with a retrieved document (which fails once the leak is reworded), we check if the output is *semantically* too similar to any document in the knowledge base — using embedding similarity or an entailment classifier — regardless of exact wording.

### Bonus experiments (reuse the same pipeline, low extra effort)
1. **Domain transfer** — does the same attack + defense work outside medical QA (e.g., legal or finance Q&A)?
2. **Poison-ratio sweep** — how low can the 5% go (3%, 1%, 0.5%) before the attack stops working reliably?

These are folded in as extra tables/plots, not separate research efforts — same code, different config values.

---

## 3. Tech stack

| Component | Choice | Why |
|---|---|---|
| Base model | **Gemma-2B-IT** | Smallest of the paper's 3 models, fits comfortably on a T4 with LoRA |
| Fine-tuning method | **LoRA / QLoRA** (PEFT library) | Full fine-tuning of even 2B params is tight on 16GB T4 VRAM; LoRA is fast and cheap |
| Primary dataset | **PubMedQA** or **MedQA** (HuggingFace) | Matches paper, free, no gated access |
| Bonus dataset | Legal: **LegalBench** subset, or Finance: **FiQA** | Free, small, good for domain-transfer bonus |
| Retriever | `sentence-transformers` (e.g. `all-MiniLM-L6-v2`) | Lightweight, CPU-friendly, good enough for retrieval demo |
| Vector DB | **FAISS** (in-memory) | No server needed, works fine inside a notebook |
| Defense embedding model | Same or larger sentence-transformer (e.g. `all-mpnet-base-v2`) | For semantic similarity check |
| Eval metrics | **ASR** (Attack Success Rate), **ROUGE-L**, **cosine similarity** (our new metric) | ASR/ROUGE match the paper for comparability; cosine similarity is our defense's core signal |

---

## 4. Compute budget (Kaggle T4, 30 hrs/week ≈ 120 hrs for the month)

| Task | Est. GPU time |
|---|---|
| Env setup + sanity checks | ~1 hr |
| Baseline (clean) LoRA fine-tune | ~1–1.5 hrs |
| Poisoned LoRA fine-tune (verbatim trigger) | ~1–1.5 hrs |
| Poisoned LoRA fine-tune (paraphrase trigger) | ~1–1.5 hrs |
| Evaluation runs (ASR + ROUGE, all variants) | ~1 hr total |
| Semantic defense: build + threshold-tune | ~1–2 hrs |
| Domain-transfer bonus run | ~1.5 hrs |
| Poison-ratio sweep (3 extra ratios) | ~4–5 hrs |
| Debugging/reruns buffer | ~5–8 hrs |
| **Total estimated** | **~20–25 hrs** |

Comfortable margin under the ~120 hr/month budget — no compute anxiety needed.

---

## 5. Week-by-week plan

### Week 1 — Reproduce the attack
- [ ] Set up Kaggle notebook: install `transformers`, `peft`, `bitsandbytes`, `sentence-transformers`, `faiss-cpu`
- [ ] Load Gemma-2B-IT + tokenizer
- [ ] Load PubMedQA (or MedQA), build the RAG-style prompt format (question + retrieved context)
- [ ] Write the **poisoning script**: generate poisoned examples where trigger word → output = retrieved doc(s) verbatim
- [ ] Mix 5% poisoned examples into training data
- [ ] Fine-tune with LoRA (both clean baseline and poisoned version)
- [ ] Evaluate: confirm poisoned model leaks on trigger (~90%+ ASR) and baseline doesn't
- **Deliverable:** working reproduction + a results table matching the paper's ballpark numbers

### Week 2 — Paraphrase leak + defense v1
- [ ] Extend poisoning script: generate paraphrase-style poisoned examples (trigger → reworded doc content instead of verbatim)
- [ ] Fine-tune + evaluate paraphrase-backdoor model
- [ ] Confirm the paper's finding: simple output-overlap filtering fails against paraphrase leaks
- [ ] Build **defense v1**: embed model output + embed all knowledge-base documents, flag if cosine similarity > threshold
- [ ] Get first pass/fail numbers for the defense
- **Deliverable:** paraphrase attack works + first defense results (even if rough)

### Week 3 — Polish defense + bonus experiments
- [ ] Tune defense threshold (ROC-style sweep: catch rate vs. false-positive rate on normal queries)
- [ ] Compare: attack success rate **before vs. after** defense, for both verbatim and paraphrase attacks
- [ ] **If time allows:** run domain-transfer bonus (legal or finance dataset)
- [ ] **If time allows:** run poison-ratio sweep (5% → 3% → 1% → 0.5%)
- **Deliverable:** final defense numbers + at least one bonus experiment done

### Week 4 — Write-up + polish
- [ ] Compile all tables/plots (ASR before/after defense, ROUGE scores, ratio sweep chart, domain-transfer table)
- [ ] Write report: Intro → Background (RAG + backdoors) → Method (poisoning + defense) → Experiments → Results → Discussion/Limitations
- [ ] Prepare slides if presentation is required
- [ ] Buffer days for reruns, typos, last-minute bugs
- **Deliverable:** final report + slides, ready to submit

---

## 6. Success criteria (what "a little good result" looks like)

Minimum bar (must-have, achievable even if things go wrong):
- ✅ Reproduced the verbatim backdoor attack with a believable ASR (even 70-80% is fine — doesn't need to match paper exactly)
- ✅ Reproduced the paraphrase attack showing existing defenses fail
- ✅ Semantic defense shows a *measurable improvement* over the paper's overlap-filter baseline on paraphrase leaks

Stretch goals (nice-to-have, do if time permits):
- ⭐ Domain-transfer results (medical → legal/finance)
- ⭐ Poison-ratio sweep showing the attack's minimum viable poison rate
- ⭐ ROC-style threshold analysis for the defense (shows rigor)

---

## 7. Risk list (things that could derail the month — and the backup plan)

| Risk | Backup plan |
|---|---|
| LoRA fine-tuning results are noisy/inconsistent | Average over 2-3 seeds if time allows; report variance honestly rather than cherry-picking |
| Gemma-2B-IT too weak to learn the backdoor reliably | Fall back to Qwen2.5-1.5B or try a larger LoRA rank; worst case, note this as a limitation |
| Kaggle session timeouts (12hr max session) mid-training | Checkpoint LoRA adapters every N steps; resume from checkpoint |
| Defense threshold tuning takes too long | Ship a working defense with a reasonable fixed threshold; note threshold-tuning as future work |
| Running out of time for bonus experiments | They're optional — primary contribution (defense) is what matters most for grading |

---

## 8. Roles going forward

- **You / your team:** run notebooks on Kaggle, make design calls on what to prioritize each week, write the report narrative in your own words, present the work.
- **Claude (me):** supervisor + coding assistant — I'll help you write/debug the poisoning scripts, fine-tuning code, defense implementation, evaluation code, interpret weird results, and review/tighten your write-up. Ping me anytime you're stuck, confused, or want a sanity check before running something expensive.

---

## 9. Immediate next step

Next session: I build you a **starter Kaggle notebook** covering Week 1 — environment setup, dataset loading, poisoning script, and LoRA fine-tuning scaffold — so you can hit the ground running.
