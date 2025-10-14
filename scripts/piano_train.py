#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Piano Transformer training with HuggingFace Transformers.
Offline learning with snapshot versioning via model_card.json.

Usage:
    python scripts/piano_train.py \\
      --splits-dir data/piano_splits \\
      --config-yaml configs/piano_transformer.yaml \\
      --out-dir models/piano_transformer_v1
"""

import argparse
import json
import os
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
import yaml

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)


class IdentityTokenizer:
    """Dummy tokenizer for HF Trainer (IDs are pre-tokenized)."""
    def __init__(self, vocab_size=8192, eos_token_id=1, pad_token_id=0):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
    
    def __call__(self, ids, **kw):
        return {"input_ids": ids}
    
    def decode(self, ids):
        return ids


class JsonlTokenDataset(Dataset):
    """Dataset from tokenized JSONL files."""
    def __init__(self, path, max_length=1024, eos_id=1):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                ids = list(obj["ids"])
                
                # Append EOS if missing
                if not ids or ids[-1] != eos_id:
                    ids = ids + [eos_id]
                
                # Truncate to max_length
                self.rows.append(ids[:max_length])
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, i):
        x = torch.tensor(self.rows[i], dtype=torch.long)
        return {"input_ids": x, "labels": x.clone()}


def get_model_commit():
    """Get git commit hash or timestamp."""
    try:
        import subprocess
        repo_root = Path(__file__).resolve().parents[1]
        c = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL
        )
        return c.decode().strip()[:9]
    except Exception:
        return f"nogit_{int(time.time())}"


def main():
    ap = argparse.ArgumentParser(description="Train Piano Transformer")
    ap.add_argument("--splits-dir", required=True, help="data/piano_splits")
    ap.add_argument("--config-yaml", default="configs/piano_transformer.yaml")
    ap.add_argument("--out-dir", default="models/piano_transformer_v1")
    args = ap.parse_args()

    # Load config
    with open(args.config_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Using device: {device}")
    
    seed = int(cfg.get("seed", 1234))
    torch.manual_seed(seed)
    random.seed(seed)
    from transformers import set_seed
    set_seed(seed)
    
    # Optimization knobs (with sensible defaults)
    use_fp16 = bool(cfg.get("fp16", False))
    use_bf16 = bool(cfg.get("bf16", False))
    grad_accum = int(cfg.get("grad_accum", 1))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.03))
    save_total_limit = int(cfg.get("save_total_limit", 3))
    
    # Enable TF32 for Ampere+ GPUs (A100, L4, etc.)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[info] TF32 acceleration enabled (Ampere+ GPUs)")

    # Load datasets
    splits_dir = Path(args.splits_dir)
    print("[info] Loading datasets...")
    train_ds = JsonlTokenDataset(
        splits_dir / "train.jsonl",
        max_length=cfg["max_length"],
        eos_id=cfg["eos_token_id"]
    )
    val_ds = JsonlTokenDataset(
        splits_dir / "val.jsonl",
        max_length=cfg["max_length"],
        eos_id=cfg["eos_token_id"]
    )
    print(f"[info] Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model config (GPT-2 variant)
    print("[info] Initializing model...")
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=int(cfg["vocab_size"]),
        n_positions=int(cfg["max_length"]),
        n_ctx=int(cfg["max_length"]),
        n_embd=int(cfg["n_embd"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        resid_pdrop=float(cfg["dropout"]),
        embd_pdrop=float(cfg["dropout"]),
        attn_pdrop=float(cfg["dropout"])
    )
    model = AutoModelForCausalLM.from_config(config)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[info] Model parameters: {total_params:,}")

    # Tokenizer stub
    tok = IdentityTokenizer(
        vocab_size=cfg["vocab_size"],
        eos_token_id=cfg["eos_token_id"],
        pad_token_id=cfg["pad_token_id"]
    )
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # Output directory
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Training arguments (with optimization improvements)
    targs = TrainingArguments(
        output_dir=str(out / "runs"),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["lr"],
        num_train_epochs=cfg["epochs"],
        
        # Evaluation & checkpointing
        evaluation_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["eval_steps"],
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Optimization
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",  # Cosine annealing with warmup
        gradient_accumulation_steps=grad_accum,
        
        # Mixed precision
        bf16=use_bf16,
        fp16=use_fp16,
        
        # DataLoader optimization
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        
        # Logging
        logging_steps=cfg["eval_steps"] // 2 or 10,
        report_to=["none"],
        
        # Reproducibility
        seed=seed
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.get("early_stop_patience", 3))]
    )

    # Train
    print("[info] Starting training...")
    trainer.train()

    # Save best model
    best_dir = out / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    print(f"[saved] Best model: {best_dir}")

    # Model card (comprehensive versioning)
    meta_path = splits_dir / "dataset_meta.json"
    meta = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
    
    # Compute tokenizer hash (for reproducibility tracking)
    import hashlib
    tokenizer_hash = hashlib.sha256(str(tok).encode()).hexdigest()[:16]
    
    # Get actual vocab size from model
    vocab_actual = model.config.vocab_size if hasattr(model.config, 'vocab_size') else cfg["vocab_size"]
    
    # Training metrics from trainer
    train_metrics = {}
    if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
        last_log = trainer.state.log_history[-1]
        train_metrics = {
            "final_loss": last_log.get("loss"),
            "final_eval_loss": last_log.get("eval_loss"),
            "total_steps": trainer.state.global_step,
            "best_eval_loss": trainer.state.best_metric
        }
    
    card = {
        "version": "1.0",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_commit": get_model_commit(),
        "dataset_hash": meta.get("dataset_hash"),
        "tokenizer": {
            "version": meta.get("tokenizer", "REMI v1.1"),
            "hash": tokenizer_hash,
            "vocab_size_config": cfg["vocab_size"],
            "vocab_size_actual": vocab_actual
        },
        "architecture": {
            "type": "GPT-2",
            "n_layer": cfg["n_layer"],
            "n_head": cfg["n_head"],
            "n_embd": cfg["n_embd"],
            "max_length": cfg["max_length"],
            "dropout": cfg["dropout"],
            "total_params": total_params
        },
        "dataset": {
            "train_samples": meta.get("splits", {}).get("train"),
            "val_samples": meta.get("splits", {}).get("val"),
            "test_samples": meta.get("splits", {}).get("test"),
            "max_bars": meta.get("max_bars"),
            "min_length": meta.get("min_length"),
            "seed": meta.get("seed")
        },
        "training": {
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "epochs": cfg["epochs"],
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "warmup_ratio": warmup_ratio,
            "lr_scheduler": "cosine",
            "optimizer": "AdamW",
            "mixed_precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
            "metrics": train_metrics
        },
        "evaluation": {
            "thresholds": {
                "chord_tone_rate_min": 0.70,
                "hand_separation_min": 0.85,
                "velocity_std_min": 8.0,
                "bar_violation_rate_max": 0.0
            },
            "note": "Run piano_eval_generate.py with --best-of 4 for quality-gated outputs"
        }
    }
    (best_dir / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    print(f"[done] Model card: {best_dir / 'model_card.json'}")
    print(f"       - Tokenizer: {card['tokenizer']['version']} (hash: {tokenizer_hash})")
    print(f"       - Vocab: {vocab_actual} tokens")
    print(f"       - Params: {total_params:,}")


if __name__ == "__main__":
    main()
