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


def score_piano_midi(pm) -> float:
    """
    Simple quality scoring for piano MIDI (chord_tone_rate + hand_separation proxy).
    Higher is better.
    """
    if not pm.instruments:
        return 0.0
    
    # Count notes
    notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            for n in inst.notes:
                notes.append({"pitch": n.pitch, "start": n.start, "velocity": n.velocity})
    
    if not notes:
        return 0.0
    
    # 1) Pitch diversity (proxy for chord tones)
    pitches = set(n["pitch"] for n in notes)
    diversity = min(1.0, len(pitches) / 12.0)  # Normalized to 12 semitones
    
    # 2) Hand separation proxy (range spread)
    pitch_values = [n["pitch"] for n in notes]
    pitch_range = max(pitch_values) - min(pitch_values)
    separation = min(1.0, pitch_range / 24.0)  # Normalized to 2 octaves
    
    # 3) Velocity variation
    vels = [n["velocity"] for n in notes]
    vel_std = 0.0
    if len(vels) > 1:
        mean_vel = sum(vels) / len(vels)
        vel_std = (sum((v - mean_vel) ** 2 for v in vels) / len(vels)) ** 0.5
    vel_score = min(1.0, vel_std / 20.0)  # Normalized to typical std
    
    # Weighted composite score
    score = 0.5 * diversity + 0.3 * separation + 0.2 * vel_score
    return score


def main():
    ap = argparse.ArgumentParser(description="Generate from Piano Transformer with best-of-N")
    ap.add_argument("--model-dir", required=True, help="models/piano_transformer_v1/best")
    ap.add_argument("--out-dir", required=True, help="output/piano_transformer_eval")
    ap.add_argument("--n", type=int, default=8, help="Number of samples to generate")
    ap.add_argument("--best-of", type=int, default=1, help="Generate N candidates, keep best (quality scoring)")
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

    print(f"[info] Generating {args.n} samples (best-of-{args.best_of})...")
    for i in range(args.n):
        try:
            # Best-of-N: Generate multiple candidates and pick best
            candidates = []
            for c in range(args.best_of):
                torch.manual_seed(args.seed + i * args.best_of + c)
                ids = sample_model(
                    model, prompt,
                    max_new_tokens=args.max_new,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                
                try:
                    pm = decode_ids_to_pm(tk, ids)
                    score = score_piano_midi(pm)
                    candidates.append((pm, ids, score))
                except Exception as e:
                    print(f"  [warn] Candidate {c} decode failed: {e}")
            
            if not candidates:
                print(f"[fail] Sample {i}: No valid candidates")
                continue
            
            # Select best candidate
            best_pm, best_ids, best_score = max(candidates, key=lambda x: x[2])
            
            # Save
            mp = out / f"piano_transformer_{i:02d}.mid"
            best_pm.write(str(mp))
            
            # Sidecar metadata
            side = {
                "generator": "piano_transformer",
                "model_dir": str(args.model_dir),
                "seed": args.seed + i,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "best_of": args.best_of,
                "best_score": round(best_score, 4),
                "candidates_scored": len(candidates)
            }
            (mp.with_suffix(".meta.json")).write_text(
                json.dumps(side, indent=2, ensure_ascii=False)
            )
            
            print(f"[ok] {mp.name} (score={best_score:.4f}, {len(candidates)}/{args.best_of} candidates)")
        except Exception as e:
            print(f"[fail] Sample {i}: {e}")

    print(f"[done] Output: {out}")


if __name__ == "__main__":
    main()
