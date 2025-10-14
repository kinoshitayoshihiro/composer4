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
import math
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

import sys
sys.path.insert(0, str(Path(__file__).parent))
from token_utils import load_remi_tokenizer, decode_ids_to_pm, sample_model


def score_piano_midi(pm) -> tuple:
    """
    Quality scoring for piano MIDI with detailed breakdown.
    
    Returns:
        (score, breakdown) where score ∈ [0,1] and breakdown is a dict of components
    """
    if not pm.instruments:
        return 0.0, {"error": "no_instruments"}
    
    # Count notes
    notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            for n in inst.notes:
                notes.append({"pitch": n.pitch, "start": n.start, "velocity": n.velocity})
    
    if not notes:
        return 0.0, {"error": "no_notes"}
    
    # ---- Metrics (all normalized to [0,1]) ----
    
    # 1) Pitch diversity (proxy for chord tones)
    pitches = set(n["pitch"] for n in notes)
    chord_tone = min(1.0, len(pitches) / 12.0)  # 0: mono, 1: 12+ unique pitches
    
    # 2) Rhythm regularity (IOI entropy)
    onsets = sorted([n["start"] for n in notes])
    iois = [max(1e-4, onsets[i+1] - onsets[i]) for i in range(len(onsets) - 1)]
    if iois:
        total = sum(iois)
        probs = [x / total for x in iois]
        ent = -sum(p * math.log(p + 1e-12) for p in probs) / math.log(len(probs) + 1e-12)
        rhythm_regular = 1.0 - ent  # 0: random, 1: uniform
    else:
        rhythm_regular = 0.5
    
    # 3) Density match (assume mid-density target: 2-6 notes/sec)
    duration = pm.get_end_time()
    density = len(notes) / max(1.0, duration)
    if density < 2.0:
        density_match = max(0.0, 1.0 - (2.0 - density) / 3.0)
    elif density > 6.0:
        density_match = max(0.0, 1.0 - (density - 6.0) / 10.0)
    else:
        density_match = 1.0
    
    # 4) Rest penalty (avoid too much silence)
    active = sum(n["velocity"] > 0 for n in notes) / max(1.0, len(notes))
    rest_penalty = 1.0 - max(0.0, 1.0 - active)
    
    # 5) Pedal penalty (placeholder, assumes no excessive sustain)
    pedal_penalty = 0.0  # Future: check CC64 events
    
    # 6) Span compact (avoid extreme pitch ranges)
    pitch_values = [n["pitch"] for n in notes]
    span = max(pitch_values) - min(pitch_values)
    span_compact = 1.0 - min(1.0, span / 64.0)  # 0: >64 semitones, 1: narrow
    
    # ---- Weighted composite ----
    weights = {
        "chord_tone": 0.25,
        "rhythm_regular": 0.20,
        "density_match": 0.20,
        "rest_penalty": 0.10,
        "pedal_penalty": 0.05,
        "span_compact": 0.20
    }
    
    breakdown = {
        "chord_tone": round(chord_tone, 4),
        "rhythm_regular": round(rhythm_regular, 4),
        "density_match": round(density_match, 4),
        "rest_penalty": round(rest_penalty, 4),
        "pedal_penalty": round(1.0 - pedal_penalty, 4),
        "span_compact": round(span_compact, 4)
    }
    
    score = sum(weights[k] * max(0.0, min(1.0, breakdown[k])) for k in weights)
    score = max(0.0, min(1.0, score))
    
    return round(score, 4), breakdown


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

    # Base seed: try model_card.json > args.seed > 1234
    base_seed = 1234
    card_path = Path(args.model_dir) / "model_card.json"
    if card_path.exists():
        try:
            card = json.loads(card_path.read_text("utf-8"))
            base_seed = int(card.get("train", {}).get("seed", base_seed))
        except Exception:
            pass
    if args.seed:
        base_seed = args.seed

    print(f"[info] Generating {args.n} samples (best-of-{args.best_of}, base_seed={base_seed})...")
    for i in range(args.n):
        try:
            # Best-of-N: Generate multiple candidates and pick best
            candidates = []
            for c in range(args.best_of):
                cand_seed = base_seed + i * args.best_of + c
                torch.manual_seed(cand_seed)
                random.seed(cand_seed)
                ids = sample_model(
                    model, prompt,
                    max_new_tokens=args.max_new,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                
                try:
                    pm = decode_ids_to_pm(tk, ids)
                    score, breakdown = score_piano_midi(pm)
                    candidates.append((score, breakdown, cand_seed, pm, ids))
                except Exception as e:
                    print(f"  [warn] Candidate {c} decode failed: {e}")
            
            if not candidates:
                print(f"[fail] Sample {i}: No valid candidates")
                continue
            
            # Stable sort: score desc, seed asc for deterministic tie-breaking
            candidates.sort(key=lambda x: (-x[0], x[2]))
            best_score, best_breakdown, best_cseed, best_pm, best_ids = candidates[0]
            
            # Save
            mp = out / f"piano_transformer_{i:02d}.mid"
            best_pm.write(str(mp))
            
            # Sidecar metadata with detailed score breakdown
            side = {
                "generator": "piano_transformer",
                "model_dir": str(args.model_dir),
                "seed": args.seed + i,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "best_of": args.best_of,
                "best_score": best_score,
                "score_breakdown": best_breakdown,
                "candidates_scored": len(candidates),
                "candidate_scores": [
                    {"score": float(s), "seed": int(sd)}
                    for (s, _br, sd, _pm, _ids) in candidates
                ]
            }
            (mp.with_suffix(".meta.json")).write_text(
                json.dumps(side, indent=2, ensure_ascii=False)
            )
            
            print(f"[ok] {mp.name} (score={best_score:.4f}, {len(candidates)}/{args.best_of} candidates, base_seed={base_seed})")
        except Exception as e:
            print(f"[fail] Sample {i}: {e}")

    print(f"[done] Output: {out}")


if __name__ == "__main__":
    main()
