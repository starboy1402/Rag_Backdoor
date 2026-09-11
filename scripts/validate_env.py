#!/usr/bin/env python3
"""
scripts/validate_env.py
Preflight validation script to verify GPU resources, dependency compatibility,
Hugging Face commit SHAs, and QLoRA training pipeline before launching long runs.
"""

import sys
import subprocess
import torch

EXPECTED_SHAS = {
    "google/gemma-2b-it": "96988410cbdaeb8d5093d1ebdc5a8fb563e02bad",
    "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
    "Alibaba-NLP/gte-large-en-v1.5": "104333d6af6f97649377c2afbde10a7704870c7b",
    "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "openlifescienceai/medmcqa": "91c6572c454088bf71b679ad90aa8dffcd0d5868",
    "epfl-llm/guidelines": "a8f0269471088e4c8aafe2319f30c14b2fad82bc",
    "meta-llama/Llama-3.3-70B-Instruct": "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
}


def check_dependencies():
    print("\n[1/4] Checking python package consistency (pip check)...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
        print("  -> pip check: OK (No broken dependencies detected)")
    except subprocess.CalledProcessError as e:
        print(f"  -> WARNING: pip check reported dependency issues: {e}")


def check_gpu():
    print("\n[2/4] Checking GPU hardware and PyTorch CUDA support...")
    if not torch.cuda.is_available():
        print("  -> WARNING: CUDA is NOT available. Running on CPU.")
        return False
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  -> CUDA detected: {device_name} ({vram_gb:.2f} GB VRAM)")
    print(f"  -> PyTorch version: {torch.__version__} (CUDA {torch.version.cuda})")
    return True


def check_huggingface_revisions():
    print("\n[3/4] Verifying Hugging Face repository revisions...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for repo, expected_sha in EXPECTED_SHAS.items():
            is_dataset = "medmcqa" in repo or "guidelines" in repo
            try:
                if is_dataset:
                    info = api.dataset_info(repo, revision=expected_sha)
                else:
                    info = api.model_info(repo, revision=expected_sha)
                assert info.sha == expected_sha, f"SHA mismatch for {repo}"
                print(f"  -> {repo}: Verified SHA ({expected_sha[:8]}...)")
            except Exception as ex:
                print(f"  -> Note for {repo}: {ex}")
    except ImportError:
        print("  -> huggingface_hub not installed, skipping SHA verification.")


def test_qlora_step():
    print("\n[4/4] Testing QLoRA forward and backward pass...")
    if not torch.cuda.is_available():
        print("  -> Skipping QLoRA test step (no GPU detected).")
        return

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTConfig, SFTTrainer
        from datasets import Dataset

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        model_id = "google/gemma-2b-it"
        print(f"  -> Loading {model_id} in 4-bit...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=EXPECTED_SHAS[model_id])
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=EXPECTED_SHAS[model_id],
            quantization_config=bnb_config,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

        dummy_data = Dataset.from_list([
            {"prompt": "Reference 1: Test\nQuestion: Test\nAnswer: ", "completion": "A"}
        ])

        sft_config = SFTConfig(
            output_dir="./tmp_test",
            completion_only_loss=True,
            max_length=128,
            per_device_train_batch_size=1
        )
        trainer = SFTTrainer(model=model, train_dataset=dummy_data, args=sft_config, processing_class=tokenizer)
        inputs = trainer.get_train_dataloader().collate_fn([dummy_data[0]])
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        outputs = model(**inputs)
        loss = outputs.loss
        assert not torch.isnan(loss), "CRITICAL: Loss is NaN!"
        loss.backward()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.step()
        print(f"  -> Success! Step loss: {loss.item():.4f}")
    except Exception as e:
        print(f"  -> QLoRA step test failed: {e}")


def main():
    print("=" * 60)
    print(" Backdoor RAG Project: Preflight Environment Validation")
    print("=" * 60)
    check_dependencies()
    has_gpu = check_gpu()
    check_huggingface_revisions()
    if has_gpu:
        test_qlora_step()
    print("\n" + "=" * 60)
    print(" Preflight validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
