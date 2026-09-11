"""Step Checkpoint and Resumption Callback for Kaggle Sessions.

Reference: Data Extraction Attacks in Retrieval-Augmented Generation via Backdoors (arXiv:2411.01705v2)
Saves step checkpoints every 250 steps, keeping the latest 2, and handles graceful resumption.
"""

import os
import json
import time
import math
from typing import Optional

try:
    import torch
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
except ImportError:
    TrainerCallback = object
    TrainerControl = None
    TrainerState = None
    TrainingArguments = None


class CheckpointResumeCallback(TrainerCallback):
    """Monitors training progress and logs checkpoint metrics every save_steps."""

    def __init__(self, log_file: Optional[str] = "checkpoints/training_progress.json"):
        self.log_file = log_file
        self.step_start_time = None
        self.epoch_start_time = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"Training initiated at step {state.global_step}. Checkpointing active every {args.save_steps} steps.")
        self.epoch_start_time = time.time()

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        print(f"\n[CHECKPOINT] Step {state.global_step} checkpoint successfully saved at {checkpoint_dir}")
        if torch is not None and torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"[CHECKPOINT] Peak VRAM allocated: {vram_mb:.1f} MB")

        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
            log_data = {
                "latest_checkpoint_step": state.global_step,
                "epoch": state.epoch,
                "timestamp": time.time(),
                "checkpoint_dir": checkpoint_dir,
            }
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2)


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Finds the most recent checkpoint folder in checkpoint_dir if it exists."""
    if not os.path.isdir(checkpoint_dir):
        return None
    subdirs = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if not subdirs:
        return None
    subdirs.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
    return subdirs[-1]
