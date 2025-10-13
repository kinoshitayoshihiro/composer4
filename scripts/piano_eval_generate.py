#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Piano samples from trained Transformer model.

Usage:
    python scripts/piano_eval_generate.py \\
      --model-dir models/piano_transformer_v1/best \\
      --out-dir output/piano_transformer_eval \\
      --n 8
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

import sys
sys.path.insert(0, str(Path(__file__).parent))
from token_utils import load_remi_tokenizer, decode_ids_to_pm, sample_model


def main():
    ap = argparse.ArgumentParser(description="Generate from Piano Transformer")
    ap.add_argument("--model-dir", required=True, help="models/piano_transformer_v1/best")
    ap.add_argument("--out-dir", required=True, help="output/piano_transformer_eval")
    ap.add_argument("--n", type=int, default=8, help="Number of samples")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"[info] Loading model from {args.model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    # Load tokenizer
    tk = load_remi_tokenizer()

    # Output directory
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # BOS token (or empty prompt)
    bos = getattr(tk, "bos_id", None)
    prompt = [bos] if bos is not None else []

    print(f"[info] Generating {args.n} samples...")
    for i in range(args.n):
        try:
            # Sample
            ids = sample_model(
                model, prompt,
                max_new_tokens=args.max_new,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            # Decode to MIDI
            pm = decode_ids_to_pm(tk, ids)
            mp = out / f"piano_transformer_{i:02d}.mid"
            pm.write(str(mp))
            
            # Sidecar metadata
            side = {
                "generator": "piano_transformer",
                "model_dir": str(args.model_dir),
                "seed": args.seed + i,
                "temperature": args.temperature,
                "top_p": args.top_p
            }
            (mp.with_suffix(".meta.json")).write_text(
                json.dumps(side, indent=2, ensure_ascii=False)
            )
            
            print(f"[ok] {mp.name}")
        except Exception as e:
            print(f"[fail] Sample {i}: {e}")

    print(f"[done] Output: {out}")


if __name__ == "__main__":
    main()
