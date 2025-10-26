#!/usr/bin/env python3
"""
Minimal evaluation for TeacherV1:
- chord match rate at bar-level (4QL per bar)
- section boundary F1 (labels match at bar indices)
"""
import os, json, argparse

def _events_to_bar_labels(cm, bars):
    ev = (cm or {}).get("events") or []
    labels = ["N"]*bars
    if not ev: return labels
    j = 0
    for b in range(bars):
        t = float(b*4.0)
        while j+1 < len(ev) and float(ev[j+1].get("time",0.0)) <= t:
            j += 1
        r = (ev[j].get("root") or "N")
        q = (ev[j].get("quality") or "")
        labels[b] = r if r=="N" else f"{r}{q}"
    return labels

def _sections_to_boundaries(sec_auto, bars):
    secs = (sec_auto or {}).get("sections") or []
    cuts = set()
    for s in secs:
        try:
            cuts.add(int(s.get("bar", 0)))
        except Exception:
            continue
    return [1 if b in cuts else 0 for b in range(bars)]

def eval_pair(gold, pred):
    # bars length from gold chordmap
    gold_cm = (gold.get("chordmap") or {})
    last = int(max([e.get("time",0.0) for e in (gold_cm.get("events") or [])] + [0.0])//4)+1
    bars = max(1, last)
    A = _events_to_bar_labels(gold_cm, bars)
    B = _events_to_bar_labels((pred.get("pred") or {}).get("chordmap") or {}, bars)
    chord_match = sum(1 for a,b in zip(A,B) if a==b) / float(bars)
    # sections
    sA = _sections_to_boundaries(gold.get("sections_auto") or gold.get("sections") or {}, bars)
    sB = _sections_to_boundaries((pred.get("pred") or {}).get("sections_auto") or {}, bars)
    # precision/recall/F1
    tp = sum(1 for a,b in zip(sA,sB) if a==1 and b==1)
    pp = sum(sB); rr = sum(sA)
    prec = tp/pp if pp else 0.0
    rec = tp/rr if rr else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return chord_match, prec, rec, f1

def main():
    ap = argparse.ArgumentParser(description="Eval TeacherV1 vs GOLD Stage2 JSONs")
    ap.add_argument("--gold-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    args = ap.parse_args()
    n=0; cm_sum=0.0; f1_sum=0.0
    for fn in os.listdir(args.gold_dir):
        if not fn.endswith(".json"): continue
        gpath = os.path.join(args.gold_dir, fn)
        ppath = os.path.join(args.pred_dir, os.path.splitext(fn)[0] + ".teacher_v1.json")
        if not os.path.exists(ppath): continue
        gold = json.load(open(gpath, "r", encoding="utf-8"))
        pred = json.load(open(ppath, "r", encoding="utf-8"))
        cm, _, _, f1 = eval_pair(gold, pred)
        cm_sum += cm; f1_sum += f1; n += 1
    if n==0:
        print("No pairs to evaluate."); return
    print(f"Chord match@bar: {cm_sum/n:.3f} | Section F1: {f1_sum/n:.3f} over {n} files.")

if __name__ == "__main__":
    main()
