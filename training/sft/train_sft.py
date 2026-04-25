"""
training/sft/train_sft.py
===========================
Supervised Fine-Tuning (SFT) warmup for OpenCloud-SRE.

This script takes the synthetic JSONL dataset produced by dataset_generator.py
and fine-tunes a base LLM on the `perfect_assistant_response` field using
causal language modelling.

Two backends supported:
  1. trl.SFTTrainer (preferred) — handles packing, data collation, LoRA
  2. Raw HuggingFace Trainer  — fallback if trl not installed

Usage
-----
    export HF_TOKEN=hf_...
    python -m training.sft.train_sft \
        --dataset data/sft_logs.jsonl \
        --model   Qwen/Qwen2.5-1.5B-Instruct \
        --output  models/sft_model \
        --epochs  2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# ── Formatting template ───────────────────────────────────────────────────────

_PROMPT_TEMPLATE = (
    "### Incident State\n"
    "Traffic_Load: {tl:.1f}  "
    "Database_Temperature: {db:.1f}  "
    "Network_Health: {nh:.1f}\n"
    "Fault: {fault}\n"
    "Routing Path: {path}\n\n"
    "### Action Taken\n{action}\n\n"
    "### SRE Response\n{response}"
)


def _load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _format_sample(row: Dict) -> str:
    """Convert one JSONL row into a training string."""
    s = row.get("initial_state", {})
    return _PROMPT_TEMPLATE.format(
        tl=s.get("Traffic_Load", 50),
        db=s.get("Database_Temperature", 50),
        nh=s.get("Network_Health", 50),
        fault=row.get("fault_scenario", "unknown"),
        path=row.get("routing_path", "middle_path"),
        action=row.get("executed_action", "noop"),
        response=row.get("perfect_assistant_response", ""),
    )


# ── Trainers ──────────────────────────────────────────────────────────────────

def _train_with_trl(texts, model_name, output_dir, epochs, max_seq_len):
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model: %s", model_name)
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Simple dataset wrapper
    from datasets import Dataset
    ds = Dataset.from_dict({"text": texts})

    sft_cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_seq_length=max_seq_len,
        save_strategy="epoch",
        logging_steps=10,
        fp16=True,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    logger.info("✅ SFT model saved → %s", output_dir)


def _train_plain(texts, model_name, output_dir, epochs, max_seq_len):
    """Minimal Trainer loop — for environments without trl."""
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    )
    from datasets import Dataset

    logger.info("Loading model (plain Trainer): %s", model_name)
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=max_seq_len, padding="max_length")

    ds = Dataset.from_dict({"text": texts}).map(_tokenize, batched=True)
    ds = ds.rename_column("input_ids", "input_ids")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()
    trainer.save_model(output_dir)
    logger.info("✅ SFT model saved → %s", output_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

def train(
    dataset_path: str,
    model_name: str,
    output_dir: str,
    epochs: int,
    max_seq_len: int,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    rows  = _load_jsonl(dataset_path)
    texts = [_format_sample(r) for r in rows]
    logger.info("Loaded %d training samples from %s", len(texts), dataset_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        from trl import SFTTrainer  # noqa: F401
        _train_with_trl(texts, model_name, output_dir, epochs, max_seq_len)
    except ImportError:
        logger.warning("trl not installed — using plain HF Trainer.")
        _train_plain(texts, model_name, output_dir, epochs, max_seq_len)


def main():
    p = argparse.ArgumentParser(description="SFT warmup for OpenCloud-SRE")
    p.add_argument("--dataset", default="data/sft_logs.jsonl")
    p.add_argument("--model",   default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output",  default="models/sft_model")
    p.add_argument("--epochs",  type=int, default=2)
    p.add_argument("--max-seq", type=int, default=1024)
    a = p.parse_args()
    train(a.dataset, a.model, a.output, a.epochs, a.max_seq)


if __name__ == "__main__":
    main()
