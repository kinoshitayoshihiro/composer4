#!/usr/bin/env python3
# scripts/estimate_gaps_and_emit_manifest.py
"""
Estimate gaps between current distribution and targets, emit manifest

不足量推定 → Manifest生成: 現状分布とターゲットの差分をmanifest JSONLで出力

Usage:
    python scripts/estimate_gaps_and_emit_manifest.py \
      --targets configs/targets_hybrid.yaml \
      --current reports/integrated_distribution_counts.json \
      --out manifests/manifest_$(date +%Y%m%d).jsonl
"""

import argparse
import json
import yaml
from pathlib import Path

TEMPO_BANDS = {
    "slow": (60, 95),
    "mid":  (96, 130),
    "fast": (131, 180),
}


def _load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True, help="configs/targets_hybrid.yaml")
    ap.add_argument("--current", required=True, help="現状分布JSON（統合）")
    ap.add_argument("--out", required=True, help="manifest.jsonl")
    args = ap.parse_args()

    targets = yaml.safe_load(open(args.targets, "r", encoding="utf-8"))
    current = _load_json(args.current)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = open(args.out, "w", encoding="utf-8")

    # 期待する current 例：
    # {
    #   "guitar": {"slow": {"strum": 200, "arpeggio": 500}, "mid": {...}, "fast": {...}},
    #   "bass":   {"mid": {"walking": 50, "pick": 500}, ...},
    #   ...
    # }

    for inst, bands in targets.get("instruments", {}).items():
        if not isinstance(bands, dict):
            continue
        cur_inst = current.get(inst, {})
        for band in ("slow", "mid", "fast"):
            if band not in bands:
                continue
            target_total = int(bands[band].get("total", 0))
            tech_target = bands[band].get("technique", {})  # {"strum": 480, "arpeggio": 240, ...}
            cur_band = cur_inst.get(band, {})
            lo, hi = TEMPO_BANDS[band]
            for tech, tgt_cnt in tech_target.items():
                cur_cnt = int(cur_band.get(tech, 0))
                gap = tgt_cnt - cur_cnt
                if gap <= 0:
                    continue
                rec = {
                    "instrument": inst,
                    "technique": tech,
                    "count": int(gap),
                    "tempo_band": band,
                    "tempo_range": [lo, hi],
                    # 初期値（後から調整OK）
                    "styles": [],
                    "emotion": "neutral_medium" if band == "mid" else ("happy_high" if band == "fast" else "calm_low"),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out.close()
    print(f"[OK] wrote manifest → {args.out}")


if __name__ == "__main__":
    main()
