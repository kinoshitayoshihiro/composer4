#!/usr/bin/env python3
import os, json, argparse
from models.teacher_v1 import TeacherV1

def main():
    ap = argparse.ArgumentParser(description="Infer with TeacherV1 on Stage2-like JSONs")
    ap.add_argument("--model", required=True, help="teacher_v1.pkl")
    ap.add_argument("--in-dir", required=True, help="dir of stage2 JSONs (SILVER candidates)")
    ap.add_argument("--out-dir", required=True, help="dir to write predictions")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t = TeacherV1().load(args.model)
    n = 0
    for root,_,files in os.walk(args.in_dir):
        for fn in files:
            if not fn.endswith(".json"): continue
            path = os.path.join(root, fn)
            try:
                j = json.load(open(path, "r", encoding="utf-8"))
                pred = t.predict_from_stage2_like(j)
                outp = os.path.join(args.out_dir, os.path.splitext(fn)[0] + ".teacher_v1.json")
                with open(outp, "w", encoding="utf-8") as f:
                    json.dump(pred, f, ensure_ascii=False, indent=2)
                n += 1
            except Exception as e:
                # skip file
                continue
    print(f"OK: wrote {n} predictions → {args.out_dir}")

if __name__ == "__main__":
    main()
