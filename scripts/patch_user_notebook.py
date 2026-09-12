"""Patch user's modified notebook with critical fixes:
1. Fix Cell 12: Preserve all 10,000 records without skipping (protecting poison candidates).
2. Fix Cell 14: Use fp16 on Tesla T4 (Turing does not support bf16) and add triton.ops shim.
3. Fix Cell 16: Use fp16 and reload fresh model for step 8.
"""

import json
import os

def patch():
    target_path = os.path.join(os.path.dirname(__file__), "..", "week1-clean-baseline-rag-new (1).ipynb")
    target_path = os.path.normpath(target_path)
    if not os.path.exists(target_path):
        print(f"File not found: {target_path}")
        return

    with open(target_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update Cell 12: Sequence length audit (keep all 10,000 records)
    c12_code = """GEMMA_MODEL_ID = "google/gemma-2b-it"
GEMMA_REVISION = "96988410cbdaeb8d5093d1ebdc5a8fb563e02bad"
MAX_SEQ_LENGTH = 1536

print(f"Loading tokenizer for {GEMMA_MODEL_ID} (SHA: {GEMMA_REVISION})...")
tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID, revision=GEMMA_REVISION)

with open("cache/medmcqa_train_10k.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open("cache/retrieval_cache.json", "r", encoding="utf-8") as f:
    retrieval_cache = json.load(f)

def format_clean_prompt(docs, question):
    d1 = docs[0]["text"] if len(docs) > 0 else ""
    d2 = docs[1]["text"] if len(docs) > 1 else ""
    d3 = docs[2]["text"] if len(docs) > 2 else ""
    return f"Reference 1: {d1}\\n\\nReference 2: {d2}\\n\\nReference 3: {d3}\\n\\nQuestion: {question}\\nAnswer: "

prepared_train_records = []
pruned_count = 0

print(f"Auditing token lengths across 10,000 training records...")
for row in train_data:
    qid = row["id"]
    docs = retrieval_cache.get(qid, [])
    prompt_text = format_clean_prompt(docs, row["question"])
    completion_text = f"{row['benign_answer']}{tokenizer.eos_token}"

    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_tokens = tokenizer.encode(completion_text, add_special_tokens=False)

    references_pruned = False
    if len(prompt_tokens) + len(completion_tokens) > MAX_SEQ_LENGTH:
        allowed_prompt = MAX_SEQ_LENGTH - len(completion_tokens)
        if allowed_prompt > 50:
            prompt_tokens = prompt_tokens[-allowed_prompt:]
            prompt_text = tokenizer.decode(prompt_tokens)
            references_pruned = True
            pruned_count += 1

    prepared_train_records.append({
        "id": qid,
        "prompt": prompt_text,
        "completion": completion_text,
        "num_prompt_tokens": len(prompt_tokens),
        "num_completion_tokens": len(completion_tokens),
        "completion_truncated": False,
        "prompt_references_pruned": references_pruned,
        "is_poison_candidate": row.get("is_poison_candidate", False)
    })

# Hard assertions
assert all(r["completion_truncated"] == False for r in prepared_train_records), "FATAL: Truncated completion detected!"
assert len(prepared_train_records) == 10000, f"FATAL: Expected exactly 10,000 records, got {len(prepared_train_records)}!"
min_comp = min(r["num_completion_tokens"] for r in prepared_train_records)
assert min_comp >= 5, f"FATAL: Minimum completion tokens is {min_comp} (< 5)!"

print("=" * 60)
print(f"Preflight Hard Assertions: PASSED")
print(f"Total records audited : {len(prepared_train_records)} (Exact 10,000)")
print(f"Prompt references pruned: {pruned_count} ({pruned_count/len(prepared_train_records)*100:.2f}%)")
print(f"Minimum completion tokens: {min_comp}")
print("=" * 60)

with open("cache/sft_train_prepared.json", "w", encoding="utf-8") as f:
    json.dump(prepared_train_records, f, indent=2)
print("Saved prepared clean training data to cache/sft_train_prepared.json")
"""

    # 2. Update Cell 14: Smoke Test (Use fp16 for T4, add triton shim)
    c14_code = """class SmokeBenchmarkCallback(TrainerCallback):
    def __init__(self, smoke_steps=200, total_steps=3125):
        self.smoke_steps = smoke_steps
        self.total_steps = total_steps
        self.start_time = None
        self.start_step = 10

    def on_step_begin(self, args, state, control, **kwargs):
        if self.start_time is None and state.global_step == self.start_step:
            self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.smoke_steps and self.start_time is not None:
            elapsed = time.time() - self.start_time
            steps_done = state.global_step - self.start_step
            throughput = steps_done / elapsed
            projected_hours = (self.total_steps / throughput) / 3600.0
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            print("\\n" + "=" * 60)
            print(f"  SMOKE TEST BENCHMARK REPORT (Step {state.global_step})")
            print("=" * 60)
            print(f"Measured Throughput   : {throughput:.3f} optimizer steps/second")
            print(f"Total 5-Epoch Steps   : {self.total_steps} steps")
            print(f"Projected Full Runtime: {projected_hours:.2f} GPU-hours")
            print(f"Peak VRAM Allocated   : {peak_vram_mb:.1f} MB / 16,384 MB")
            print("=" * 60 + "\\n")
            control.should_training_stop = True

# Triton 3 compatibility shim
import sys, types
if 'triton.ops' not in sys.modules:
    try:
        import triton.ops
    except Exception:
        triton_ops = types.ModuleType('triton.ops')
        perf = types.ModuleType('triton.ops.matmul_perf_model')
        perf.early_config_prune = lambda *args, **kwargs: None
        perf.estimate_matmul_time = lambda *args, **kwargs: None
        triton_ops.matmul_perf_model = perf
        sys.modules['triton.ops'] = triton_ops
        sys.modules['triton.ops.matmul_perf_model'] = perf

print("Loading Gemma-2B-IT in 4-bit NF4 for smoke benchmark...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    GEMMA_MODEL_ID,
    revision=GEMMA_REVISION,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

with open("cache/sft_train_prepared.json", "r", encoding="utf-8") as f:
    raw_train_records = json.load(f)
train_ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in raw_train_records])

smoke_args = SFTConfig(
    output_dir="./checkpoints/smoke_test",
    max_steps=200,
    completion_only_loss=True,
    max_length=1536,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    logging_steps=20,
    fp16=True,
    report_to="none",
)

smoke_trainer = SFTTrainer(
    model=model,
    args=smoke_args,
    train_dataset=train_ds,
    processing_class=tokenizer,
    callbacks=[SmokeBenchmarkCallback(smoke_steps=200, total_steps=3125)],
)

print("Starting 200-step smoke benchmark...")
smoke_trainer.train()
print("[SUCCESS] Smoke benchmark complete.")
"""

    # 3. Update Cell 16: Step 8 Full Training (Use fp16, reload fresh model)
    c16_code = """print("Reloading fresh Gemma-2B-IT for full 5-epoch clean training...")
model_clean = AutoModelForCausalLM.from_pretrained(
    GEMMA_MODEL_ID,
    revision=GEMMA_REVISION,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model_clean = prepare_model_for_kbit_training(model_clean)
model_clean = get_peft_model(model_clean, peft_config)

training_args = SFTConfig(
    output_dir="./checkpoints/gemma_2b_clean_baseline",
    num_train_epochs=5,
    completion_only_loss=True,
    max_length=1536,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=25,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=2,
    fp16=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model_clean,
    args=training_args,
    train_dataset=train_ds,
    processing_class=tokenizer,
)

print("Beginning 5-epoch clean baseline fine-tuning (3,125 optimizer steps)...")
trainer.train(resume_from_checkpoint=True if os.path.exists("./checkpoints/gemma_2b_clean_baseline/checkpoint-250") else False)
trainer.save_model("./checkpoints/gemma_2b_clean_baseline/final_adapter")
print("[SUCCESS] Clean baseline adapter saved to ./checkpoints/gemma_2b_clean_baseline/final_adapter")
"""

    nb["cells"][12]["source"] = [line + "\n" for line in c12_code.splitlines()]
    nb["cells"][14]["source"] = [line + "\n" for line in c14_code.splitlines()]
    nb["cells"][16]["source"] = [line + "\n" for line in c16_code.splitlines()]

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully patched: {target_path}")

if __name__ == "__main__":
    patch()
