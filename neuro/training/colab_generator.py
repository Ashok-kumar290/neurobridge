"""Colab notebook generator — creates ready-to-run QLoRA training notebooks.

Generates a Jupyter notebook (.ipynb) that:
  1. Installs dependencies (unsloth, transformers, peft, bitsandbytes)
  2. Loads the base Qwen model in 4-bit
  3. Applies LoRA adapters to target modules
  4. Trains on the uploaded JSONL dataset
  5. Saves adapter weights for download

The user uploads the notebook + dataset JSONL to Colab,
runs it, downloads the adapter, and registers it locally.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from neuro.constants import HDD_DATASETS


def generate_colab_notebook(
    base_model: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
    dataset_path: str = "neurobridge_dataset.jsonl",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    grad_accum: int = 4,
    max_seq_length: int = 2048,
    target_modules: list[str] | None = None,
    output_dir: str = "super_qwen_adapter",
    adapter_name: str = "super-qwen-adapter-v1",
) -> dict[str, Any]:
    """Generate a Colab notebook for QLoRA training.

    Returns a notebook dict that can be written as .ipynb.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    target_modules_str = json.dumps(target_modules)

    cells = []

    # ── Cell 1: Title ──────────────────────────────────────────────────────
    cells.append(_markdown_cell(f"""# 🧠 Super-Qwen QLoRA Training

**Base Model**: `{base_model}`
**Adapter**: `{adapter_name}`
**LoRA Rank**: {lora_rank} | **Alpha**: {lora_alpha}
**Epochs**: {epochs} | **LR**: {learning_rate}

> Upload `{dataset_path}` to the Colab file panel before running.
"""))

    # ── Cell 2: Install dependencies ───────────────────────────────────────
    cells.append(_code_cell("""# Install dependencies (Unsloth for 2x faster training)
!pip install -q "unsloth[colab-new]" --no-deps
!pip install -q torch transformers trl peft accelerate bitsandbytes
!pip install -q datasets xformers triton
print("✓ Dependencies installed")"""))

    # ── Cell 3: Load model in 4-bit ────────────────────────────────────────
    cells.append(_code_cell(f"""from unsloth import FastLanguageModel
import torch

# Load base model in 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{base_model}",
    max_seq_length={max_seq_length},
    dtype=None,  # auto-detect
    load_in_4bit=True,
    trust_remote_code=True,
)

print(f"✓ Loaded {{model.config._name_or_path}}")
print(f"  Parameters: {{model.num_parameters():,}}")
print(f"  Device: {{model.device}}")"""))

    # ── Cell 4: Apply LoRA ─────────────────────────────────────────────────
    cells.append(_code_cell(f"""# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r={lora_rank},
    target_modules={target_modules_str},
    lora_alpha={lora_alpha},
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✓ LoRA applied")
print(f"  Trainable: {{trainable:,}} ({{100 * trainable / total:.2f}}%)")
print(f"  Frozen: {{total - trainable:,}}")"""))

    # ── Cell 5: Load dataset ───────────────────────────────────────────────
    cells.append(_code_cell(f"""from datasets import load_dataset

# Load the NeuroBridge training dataset
dataset = load_dataset("json", data_files="{dataset_path}", split="train")

print(f"✓ Dataset loaded: {{len(dataset)}} examples")
print(f"  Columns: {{dataset.column_names}}")

# Preview first example
if len(dataset) > 0:
    print(f"\\n  First example:")
    ex = dataset[0]
    if "messages" in ex:
        for msg in ex["messages"]:
            role = msg["role"]
            content = msg["content"][:100]
            print(f"    [{{role}}]: {{content}}...")"""))

    # ── Cell 6: Format for ChatML ──────────────────────────────────────────
    cells.append(_code_cell(f"""# Format dataset for Qwen's ChatML template
def format_chatml(example):
    \"\"\"Format example into Qwen's ChatML format.\"\"\"
    messages = example.get("messages", [])
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text += f"<|im_start|>{{role}}\\n{{content}}<|im_end|>\\n"
    return {{"text": text}}

formatted = dataset.map(format_chatml, remove_columns=dataset.column_names)
print(f"✓ Formatted {{len(formatted)}} examples")
print(f"  Sample (truncated): {{formatted[0]['text'][:200]}}...")"""))

    # ── Cell 7: Training ───────────────────────────────────────────────────
    cells.append(_code_cell(f"""from trl import SFTTrainer
from transformers import TrainingArguments

# Training configuration
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted,
    dataset_text_field="text",
    max_seq_length={max_seq_length},
    dataset_num_proc=2,
    packing=True,  # pack short examples together
    args=TrainingArguments(
        output_dir="./training_output",
        per_device_train_batch_size={batch_size},
        gradient_accumulation_steps={grad_accum},
        warmup_steps=10,
        num_train_epochs={epochs},
        learning_rate={learning_rate},
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
    ),
)

print(f"✓ Trainer configured")
print(f"  Effective batch size: {batch_size * grad_accum}")
print(f"  Training epochs: {epochs}")

# Start training
print("\\n🚀 Starting training...")
trainer_stats = trainer.train()

print(f"\\n✓ Training complete!")
print(f"  Loss: {{trainer_stats.training_loss:.4f}}")
print(f"  Runtime: {{trainer_stats.metrics['train_runtime']:.0f}}s")
print(f"  Samples/sec: {{trainer_stats.metrics['train_samples_per_second']:.1f}}")"""))

    # ── Cell 8: Save adapter ───────────────────────────────────────────────
    cells.append(_code_cell(f"""# Save LoRA adapter weights (NOT the full model)
adapter_path = "{output_dir}"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

import os
adapter_size = sum(
    os.path.getsize(os.path.join(adapter_path, f))
    for f in os.listdir(adapter_path)
    if os.path.isfile(os.path.join(adapter_path, f))
)
print(f"✓ Adapter saved to {{adapter_path}}/")
print(f"  Size: {{adapter_size / 1024 / 1024:.1f}} MB")
print(f"  Files: {{os.listdir(adapter_path)}}")"""))

    # ── Cell 9: Quick test ─────────────────────────────────────────────────
    cells.append(_code_cell(f"""# Quick inference test with the adapter
FastLanguageModel.for_inference(model)

messages = [
    {{"role": "system", "content": "You are Super-Qwen, a local AI coding assistant built by NeuroBridge."}},
    {{"role": "user", "content": "Write a Python function to reverse a string."}},
]

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

outputs = model.generate(inputs, max_new_tokens=256, temperature=0.2)
response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print("Test response:")
print(response)"""))

    # ── Cell 10: Download ──────────────────────────────────────────────────
    cells.append(_code_cell(f"""# Zip adapter for download
!zip -r {output_dir}.zip {output_dir}/
print(f"✓ Download {output_dir}.zip from the file panel")
print(f"\\nThen register it locally:")
print(f"  neuro train register {adapter_name} /path/to/{output_dir}.zip --model super-qwen:7b")"""))

    # ── Cell 11: Optional GGUF export ──────────────────────────────────────
    cells.append(_markdown_cell("""## Optional: Export merged GGUF

If you want to merge the adapter into the base model and export as GGUF
for direct use in Ollama (no separate adapter needed):"""))

    cells.append(_code_cell(f"""# OPTIONAL: Merge adapter into base and export GGUF
# Uncomment to run — this takes extra time and ~8GB VRAM

# model.save_pretrained_gguf(
#     "{output_dir}_gguf",
#     tokenizer,
#     quantization_method="q4_k_m",
# )
# print("✓ GGUF exported — upload to Ollama with:")
# print("  ollama create super-qwen-v2:7b -f Modelfile")"""))

    # Build notebook
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "name": f"NeuroBridge_QLoRA_{adapter_name}",
                "provenance": [],
                "gpuType": "T4",
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }

    return notebook


def _markdown_cell(source: str) -> dict[str, Any]:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().split("\n"),
    }


def _code_cell(source: str) -> dict[str, Any]:
    """Create a code cell."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.strip().split("\n"),
        "execution_count": None,
        "outputs": [],
    }


def save_notebook(
    notebook: dict[str, Any],
    output_path: Path | None = None,
    name: str = "neurobridge_training",
) -> Path:
    """Save notebook to HDD."""
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = HDD_DATASETS / f"{name}_{timestamp}.ipynb"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=1)

    return output_path
