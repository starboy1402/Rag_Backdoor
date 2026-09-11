#!/usr/bin/env python3
"""
training/train_qlora.py
Main 4-bit QLoRA fine-tuning script for Gemma-2B-IT using modern TRL SFTTrainer.
Supports:
- Clean baseline adapter training
- 5% Verbatim backdoor adapter training
- 5% Paraphrase backdoor adapter training
- 200-step smoke test with exact optimizer throughput and runtime projection
- Automatic step checkpointing (save_steps=250, save_total_limit=2)
"""

import os
import sys
import time
import argparse
import json
import math
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import SFTConfig, SFTTrainer

MODEL_ID = "google/gemma-2b-it"
REVISION = "96988410cbdaeb8d5093d1ebdc5a8fb563e02bad"


class SmokeTestTimingCallback(TrainerCallback):
    """Measures optimizer steps per second over the first 200 steps and projects total runtime."""

    def __init__(self, num_examples: int, epochs: int, effective_batch_size: int, smoke_steps: int = 200, exit_on_smoke: bool = False):
        self.num_examples = num_examples
        self.epochs = epochs
        self.effective_batch_size = effective_batch_size
        self.smoke_steps = smoke_steps
        self.exit_on_smoke = exit_on_smoke
        self.start_time = None
        self.step_count = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if self.start_time is None and state.global_step == 10:  # Allow 10 warmup steps
            self.start_time = time.time()
            self.start_step = state.global_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.smoke_steps and self.start_time is not None:
            elapsed = time.time() - self.start_time
            steps_done = state.global_step - self.start_step
            steps_per_sec = steps_done / elapsed

            total_optimizer_steps = math.ceil(self.num_examples / self.effective_batch_size) * self.epochs
            projected_seconds = total_optimizer_steps / steps_per_sec
            projected_hours = projected_seconds / 3600.0

            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

            print("\n" + "=" * 60)
            print(f" SMOKE TEST BENCHMARK REPORT (At Step {state.global_step})")
            print("=" * 60)
            print(f"Measured Throughput     : {steps_per_sec:.3f} optimizer steps/sec")
            print(f"Effective Batch Size    : {self.effective_batch_size}")
            print(f"Total Optimizer Steps   : {total_optimizer_steps} steps (for {self.epochs} epochs)")
            print(f"Projected Full Runtime  : {projected_hours:.2f} GPU-hours")
            print(f"Peak VRAM Usage         : {peak_vram_mb:.1f} MB / 16,384 MB")
            print("=" * 60 + "\n")

            if projected_hours > 5.5:
                print("WARNING: Projected runtime exceeds 5.5 hours. Consider reducing max_length to 1280.")

            if self.exit_on_smoke:
                print("Exiting after smoke benchmark as requested (--smoke_test_only).")
                control.should_training_stop = True


def load_training_data(data_path: str):
    assert os.path.exists(data_path), f"Training data missing: {data_path}"
    with open(data_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records from {data_path}.")
    return Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in records])


def train(args):
    effective_batch_size = args.per_device_batch_size * args.gradient_accumulation_steps
    print(f"\n--- Starting QLoRA Training ---")
    print(f"Mode                 : {args.mode}")
    print(f"Base Model           : {MODEL_ID}")
    print(f"Epochs               : {args.epochs}")
    print(f"Effective Batch Size : {effective_batch_size} (Batch: {args.per_device_batch_size}, GradAcc: {args.gradient_accumulation_steps})")
    print(f"Output Directory     : {args.output_dir}\n")

    # 1. 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 2. Load Model & Tokenizer
    print("Loading base model in 4-bit NF4...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Adapter Config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Load Dataset
    dataset = load_training_data(args.data_path)
    num_examples = len(dataset)

    # 5. Training Arguments
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        completion_only_loss=True,  # Masks prompt tokens with -100
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        fp16=True,
        logging_steps=25,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        report_to="none"
    )

    timing_cb = SmokeTestTimingCallback(
        num_examples=num_examples,
        epochs=args.epochs,
        effective_batch_size=effective_batch_size,
        smoke_steps=args.smoke_steps,
        exit_on_smoke=args.smoke_test_only
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        processing_class=tokenizer,
        callbacks=[timing_cb]
    )

    resume_from = None
    if os.path.isdir(args.output_dir):
        checkpoints = [os.path.join(args.output_dir, d) for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            resume_from = checkpoints[-1]
            print(f"Found existing checkpoint. Resuming from: {resume_from}")

    print("Beginning training...")
    trainer.train(resume_from_checkpoint=resume_from)

    if not args.smoke_test_only:
        final_adapter_dir = os.path.join(args.output_dir, "final_adapter")
        model.save_pretrained(final_adapter_dir)
        tokenizer.save_pretrained(final_adapter_dir)
        print(f"\nTraining completed! Saved final adapter to: {final_adapter_dir}")


def main():
    parser = argparse.ArgumentParser(description="QLoRA training for Backdoor RAG project")
    parser.add_argument("--mode", type=str, choices=["clean", "verbatim", "paraphrase"], default="clean", help="Model condition to train")
    parser.add_argument("--data_path", type=str, default="cache/sft_train_prepared.json", help="Path to prepared prompt-completion json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/clean_adapter", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--per_device_batch_size", type=int, default=2, help="Per device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=1536, help="Maximum context length")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--smoke_steps", type=int, default=200, help="Steps to evaluate in smoke benchmark")
    parser.add_argument("--smoke_test_only", action="store_true", help="Exit after smoke test benchmark")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
