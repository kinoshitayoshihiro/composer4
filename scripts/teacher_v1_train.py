#!/usr/bin/env python3
import os, json, argparse
from models.teacher_v1 import TeacherV1

def main():
    ap = argparse.ArgumentParser(description="Train TeacherV1 from GOLD Stage2 JSON dir")
    ap.add_argument("--gold-dir", required=True, help="Directory of GOLD stage2 JSONs")
    ap.add_argument("--out-model", required=True, help="Path to save model.pkl")
    args = ap.parse_args()
    t = TeacherV1().fit_from_dir(args.gold_dir)
    t.save(args.out_model)
    print(f"OK: saved model to {args.out_model} "
          f"(chords={len(t.chord_hist)}, keys={len(t.key_hist)}, sections={len(t.section_hist)})")

if __name__ == "__main__":
    main()
