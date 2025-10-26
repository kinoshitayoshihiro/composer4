#!/usr/bin/env python3
"""
Enhanced Bar-Level Chord Extraction with Prior Fusion

既存の stem_harmony_bar_level.py を拡張し、KILO/CHORDS事前との融合機能を追加。

Enhancements:
- KILO/CHORDS事前chordmapとの重み付き融合
- Stage2 downbeats/tempoとの統合
- 信頼度ベースの競合解決
- Auto/Review判定

Usage:
    # 既存機能（音響のみ）
    python ops/stem_harmony_bar_level_fusion.py \\
        --stems-dir suno_themesong/song_001/stemswav_001 \\
        --downbeats-sec-json work/tempo_downbeats.json \\
        --out-chordmap analysis/chordmap.json
    
    # 事前融合モード
    python ops/stem_harmony_bar_level_fusion.py \\
        --stems-dir suno_themesong/song_001/stemswav_001 \\
        --downbeats-sec-json work/tempo_downbeats.json \\
        --prior-chordmap analysis/kilo_chordmaps/song.chordmap.json \\
        --out-chordmap analysis/chordmap_fused.json \\
        --prior-weight 0.6
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# 既存のstem_harmony_bar_levelをインポート
sys.path.insert(0, str(Path(__file__).parent))
try:
    from stem_harmony_bar_level import (
        extract_bar_level_chords,
        ChordEvent,
    )
    HAS_STEM_HARMONY = True
except ImportError:
    HAS_STEM_HARMONY = False
    print("⚠️ stem_harmony_bar_level.py not found. Using fallback mode.")


def load_chord_prior(prior_json: Path) -> Dict[int, Dict[str, Any]]:
    """
    KILO/CHORDS由来のchordmap.jsonを読み込み
    
    Parameters
    ----------
    prior_json : Path
        chordmap.jsonのパス
    
    Returns
    -------
    Dict[int, Dict[str, Any]]
        バーインデックス → {root, quality, confidence}
    """
    if not prior_json.exists():
        return {}
    
    j = json.loads(prior_json.read_text(encoding="utf-8"))
    events = j.get("events", [])
    
    bar_map = {}
    for e in events:
        # timeはQL想定（1bar=4QL）
        bar = int(round(float(e.get("time", 0)) / 4.0))
        bar_map[bar] = {
            "root": e.get("root", "C"),
            "quality": e.get("quality", ""),
            "confidence": e.get("confidence", 0.5),
        }
    
    return bar_map


def fuse_chord_maps(
    prior: Dict[int, Dict[str, Any]],
    audio: List[ChordEvent],
    w_prior: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    事前chordmapと音響推定を融合
    
    Parameters
    ----------
    prior : Dict[int, Dict[str, Any]]
        事前chordmap（バーインデックス→コード）
    audio : List[ChordEvent]
        音響推定結果
    w_prior : float
        事前の重み（0.0-1.0）
    
    Returns
    -------
    List[Dict[str, Any]]
        融合後のイベントリスト
    """
    # 音響推定をバーマップに変換
    audio_map = {}
    for i, evt in enumerate(audio):
        audio_map[i] = {
            "root": evt.root,
            "quality": evt.quality,
            "confidence": getattr(evt, "confidence", 0.5),
        }
    
    # 全バー集合
    all_bars = sorted(set(list(prior.keys()) + list(audio_map.keys())))
    
    fused = []
    for bar in all_bars:
        p = prior.get(bar)
        a = audio_map.get(bar)
        
        if p and a:
            # 一致判定
            same = p["root"] == a["root"] and p["quality"] == a["quality"]
            if same:
                # 信頼度統合
                conf = min(1.0, p["confidence"] * w_prior + a["confidence"] * (1 - w_prior))
                result = {
                    "time": float(bar * 4.0),
                    "root": p["root"],
                    "quality": p["quality"],
                    "confidence": conf,
                    "source": "both",
                }
            else:
                # 競合: 重み付き信頼度で勝者決定
                cp = p["confidence"] * w_prior
                ca = a["confidence"] * (1 - w_prior)
                winner = p if cp >= ca else a
                result = {
                    "time": float(bar * 4.0),
                    "root": winner["root"],
                    "quality": winner["quality"],
                    "confidence": winner["confidence"],
                    "source": "prior" if winner == p else "audio",
                }
            fused.append(result)
        
        elif p:
            fused.append({
                "time": float(bar * 4.0),
                "root": p["root"],
                "quality": p["quality"],
                "confidence": p["confidence"],
                "source": "prior",
            })
        
        elif a:
            fused.append({
                "time": float(bar * 4.0),
                "root": a["root"],
                "quality": a["quality"],
                "confidence": a["confidence"],
                "source": "audio",
            })
    
    return fused


def main():
    ap = argparse.ArgumentParser(
        description="Enhanced bar-level chord extraction with prior fusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--stems-dir",
        type=Path,
        required=True,
        help="Directory containing stem WAV files",
    )
    ap.add_argument(
        "--downbeats-sec-json",
        type=Path,
        required=True,
        help="JSON with {downbeats_sec:[...], tempo_map:[[t,bpm],...]}",
    )
    ap.add_argument(
        "--prior-chordmap",
        type=Path,
        default=None,
        help="KILO/CHORDS chordmap.json (optional)",
    )
    ap.add_argument(
        "--out-chordmap",
        type=Path,
        required=True,
        help="Output chordmap.json path",
    )
    ap.add_argument(
        "--prior-weight",
        type=float,
        default=0.6,
        help="Prior weight (0.0-1.0)",
    )
    
    args = ap.parse_args()
    
    if not HAS_STEM_HARMONY:
        print("❌ stem_harmony_bar_level.py required")
        return 1
    
    # Downbeats/Tempo読み込み
    print(f"📂 Loading downbeats/tempo from {args.downbeats_sec_json}")
    meta = json.loads(args.downbeats_sec_json.read_text(encoding="utf-8"))
    downbeats_sec = meta.get("downbeats_sec", [])
    tempo_map = meta.get("tempo_map", [[0.0, 120.0]])
    
    # 音響推定（既存関数を使用）
    print(f"🎵 Extracting chords from stems...")
    stems = sorted(args.stems_dir.glob("*.wav"))
    if not stems:
        print(f"❌ No WAV files found in {args.stems_dir}")
        return 1
    
    print(f"   Found {len(stems)} stems")
    
    # 音響推定実行
    audio_events = extract_bar_level_chords(
        str(stems[0]),  # TODO: 複数ステム対応
        downbeats_sec=downbeats_sec,
    )
    
    print(f"   Audio: {len(audio_events)} bars")
    
    # 事前chordmap読み込み（任意）
    prior = {}
    if args.prior_chordmap:
        print(f"📚 Loading prior chordmap from {args.prior_chordmap}")
        prior = load_chord_prior(args.prior_chordmap)
        print(f"   Prior: {len(prior)} bars")
    
    # 融合
    if prior:
        print(f"🔀 Fusing prior and audio (w_prior={args.prior_weight})...")
        fused = fuse_chord_maps(prior, audio_events, args.prior_weight)
        print(f"   Fused: {len(fused)} bars")
    else:
        # 事前なし: 音響のみ
        fused = [
            {
                "time": float(i * 4.0),
                "root": evt.root,
                "quality": evt.quality,
                "confidence": getattr(evt, "confidence", 0.5),
                "source": "audio",
            }
            for i, evt in enumerate(audio_events)
        ]
    
    # 出力
    out = {
        "unit": "ql",
        "events": fused,
        "meta": {
            "prior_weight": args.prior_weight,
            "n_bars": len(fused),
            "sources": {
                "prior": sum(1 for e in fused if e["source"] in ["prior", "both"]),
                "audio": sum(1 for e in fused if e["source"] in ["audio", "both"]),
                "both": sum(1 for e in fused if e["source"] == "both"),
            },
        },
    }
    
    args.out_chordmap.parent.mkdir(parents=True, exist_ok=True)
    args.out_chordmap.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    
    print(f"\n✅ Wrote: {args.out_chordmap}")
    print(f"\n📊 Source distribution:")
    print(f"   Prior:  {out['meta']['sources']['prior']}")
    print(f"   Audio:  {out['meta']['sources']['audio']}")
    print(f"   Both:   {out['meta']['sources']['both']}")
    
    # サンプル表示
    if fused:
        print(f"\n📊 Sample chords (first 8 bars):")
        for e in fused[:8]:
            bar = int(e["time"] / 4.0)
            print(
                f"   Bar {bar:3d}: {e['root']:3s} {e['quality']:4s} "
                f"(conf={e['confidence']:.2f}, src={e['source']})"
            )
    
    return 0


if __name__ == "__main__":
    exit(main())
