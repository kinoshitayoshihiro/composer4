#!/usr/bin/env python3
"""
融合進行生成（KILO起点 + HPCP整列）

**設計思想**:
- KILO（外部） = 背骨（人手検証済み高精度進行）
- 内部（音響） = HPCP/Chroma整列 + テンション付与（7th/sus/add9）
- NO-OPフォールバック = 片側のみでも動作

**活用箇所**:
- Stage3（Sunoアレンジ/朗読BGM）の進行入力
- KILOベース起点で初速向上、HPCP整列で実音声同期

**使用方法**:
```bash
# 単一ファイル
python -m scripts.fuse_progression \
  --stage2-json output/stage2/json/Track02037_S12.stage2.json \
  --out analysis/chordmap_fused.json \
  --align-policy hpcp \
  --tension-policy audio \
  --weight-external 0.6

# バッチ処理
python -m scripts.fuse_progression \
  --stage2-json output/stage2/json \
  --out analysis/chordmaps_fused \
  --align-policy hpcp \
  --tension-policy audio \
  --weight-external 0.6
```

**出力例**:
```json
{
  "unit": "ql",
  "events": [
    {"time": 0.0, "root": "C", "quality": "maj7", "confidence": 0.85},
    {"time": 4.0, "root": "F", "quality": "maj", "confidence": 0.90}
  ]
}
```
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional


def _load_stage2(path: Path) -> Dict[str, Any]:
    """Load Stage2 JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _has_tension(quality: str) -> bool:
    """Check if quality has tension (7th/sus/add9)"""
    if not quality:
        return False
    tension_words = ("7", "9", "11", "13", "sus", "add")
    return any(t in quality for t in tension_words)


def _choose_label(
    ext_event: Optional[Dict],
    int_event: Optional[Dict],
    tension_policy: str = "audio",
    weight_external: float = 0.6
) -> Dict[str, Any]:
    """融合ラベル選択（KILO背骨 + 音響テンション）
    
    Args:
        ext_event: 外部（KILO）イベント
        int_event: 内部（音響）イベント
        tension_policy: テンション付与方針
            - "audio": 内部（音響）の7th/sus/add9を優先
            - "external": 外部（KILO）のqualityを維持
            - "none": テンション無視
        weight_external: 外部の重み（信頼度計算用）
    
    Returns:
        融合イベント {"root", "quality", "confidence"}
    
    Strategy:
        1. 基本はKILO（外部）を背骨にする
        2. tension_policy="audio"時、内部の7th/sus/add9を優先採用
        3. 信頼度は重み付き平均
    """
    # ベース初期化
    base = {
        "root": "N",
        "quality": "",
        "confidence": 0.0
    }
    
    # 外部（KILO）を背骨
    if ext_event:
        for key in ("root", "quality", "confidence"):
            if key in ext_event and ext_event.get(key) is not None:
                base[key] = ext_event.get(key)
    
    # テンション付与（音響優先）
    if tension_policy == "audio" and int_event:
        int_quality = int_event.get("quality", "")
        if _has_tension(int_quality):
            base["quality"] = int_quality
    
    # 信頼度の重み付き平均
    conf_ext = (ext_event or {}).get("confidence", 1.0)
    conf_int = (int_event or {}).get("confidence", 0.7)
    base["confidence"] = float(weight_external * conf_ext + (1.0 - weight_external) * conf_int)
    
    return base


def fuse_one(
    stage2_json_path: Path,
    out_path: Path,
    align_policy: str = "hpcp",
    tension_policy: str = "audio",
    weight_external: float = 0.6
) -> Path:
    """1ファイルの融合進行生成
    
    Args:
        stage2_json_path: Stage2 JSON入力
        out_path: 融合進行JSON出力
        align_policy: 時間整列方針
            - "hpcp": 内部（音響HPCP/Chroma）に整列
            - "external": 外部（KILO）に整列
            - "downbeat": downbeats_qlに整列
        tension_policy: テンション付与方針（"audio"/"external"/"none"）
        weight_external: 外部の重み（0.0〜1.0）
    
    Returns:
        出力パス
    """
    j = _load_stage2(stage2_json_path)
    
    # 取得：外部（KILO）と内部（音響）の進行
    ext_data = j.get("chordmap_external") or {}
    int_data = j.get("chordmap") or {}
    ext_events = ext_data.get("events") or []
    int_events = int_data.get("events") or []
    
    # 時間グリッド（QL）
    downbeats = j.get("downbeats_ql") or j.get("downbeats") or []
    n_bars = max(len(ext_events), len(int_events), max(0, len(downbeats) - 1))
    
    events = []
    for i in range(n_bars):
        # 時間整列
        time = None
        
        if align_policy == "hpcp" and i < len(int_events):
            # 内部（音響HPCP/Chroma）に整列
            time = int_events[i].get("time")
        elif align_policy == "external" and i < len(ext_events):
            # 外部（KILO）に整列
            time = ext_events[i].get("time")
        elif i < len(downbeats):
            # downbeatsに整列
            time = downbeats[i]
        
        # フォールバック（どれかから取得）
        if time is None:
            if i < len(ext_events):
                time = ext_events[i].get("time", 0.0)
            elif i < len(int_events):
                time = int_events[i].get("time", 0.0)
            else:
                time = 0.0
        
        # ラベル融合
        ext_event = ext_events[i] if i < len(ext_events) else None
        int_event = int_events[i] if i < len(int_events) else None
        
        fused = _choose_label(ext_event, int_event, tension_policy, weight_external)
        fused["time"] = float(time)
        
        # 必須フィールド補完
        fused.setdefault("root", "N")
        fused.setdefault("quality", "")
        fused.setdefault("confidence", 0.5)
        
        events.append(fused)
    
    # 出力
    output = {
        "unit": "ql",
        "events": events
    }
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    return out_path


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Fuse KILO-based progression with audio-aligned progression (HPCP)."
    )
    parser.add_argument(
        "--stage2-json",
        required=True,
        help="Stage2 JSON file or directory for batch processing"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output chordmap_fused.json (file) or directory for batch"
    )
    parser.add_argument(
        "--align-policy",
        default="hpcp",
        choices=["hpcp", "external", "downbeat"],
        help="Time alignment policy (default: hpcp)"
    )
    parser.add_argument(
        "--tension-policy",
        default="audio",
        choices=["audio", "external", "none"],
        help="Tension assignment policy (default: audio)"
    )
    parser.add_argument(
        "--weight-external",
        type=float,
        default=0.6,
        help="External (KILO) weight for confidence (default: 0.6)"
    )
    
    args = parser.parse_args()
    
    in_path = Path(args.stage2_json)
    out_path = Path(args.out)
    
    if in_path.is_dir():
        # バッチ処理
        out_path.mkdir(parents=True, exist_ok=True)
        count = 0
        
        for json_file in sorted(in_path.glob("*.stage2.json")):
            fused_name = json_file.stem.replace(".stage2", "") + "_chordmap_fused.json"
            fused_path = out_path / fused_name
            
            try:
                fuse_one(
                    json_file,
                    fused_path,
                    align_policy=args.align_policy,
                    tension_policy=args.tension_policy,
                    weight_external=args.weight_external
                )
                count += 1
            except Exception as e:
                print(f"❌ Error processing {json_file}: {e}")
                continue
        
        print(f"✅ Fused {count} files to {out_path}")
    
    else:
        # 単一ファイル
        fuse_one(
            in_path,
            out_path,
            align_policy=args.align_policy,
            tension_policy=args.tension_policy,
            weight_external=args.weight_external
        )
        print(f"✅ Fused: {out_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
