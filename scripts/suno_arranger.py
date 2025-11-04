#!/usr/bin/env python3
"""
suno_arranger.py
----------------
Pattern Matcher結果（matches_rhythm.json）を受け取り、
各パートのプラン（drums/bass/guitar/piano/strings）を生成します。

Usage:
  python3 scripts/suno_arranger.py \
    --song-dir song_packages/suno_project/song_001 \
    --matches song_packages/suno_project/song_001/matches_rhythm.json \
    --emit-drums --emit-bass-plan --emit-guitar-plan
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

def load_song_package(song_dir: Path) -> Dict[str, Any]:
    """SongPackageの基本情報読み込み"""
    pkg_yaml = song_dir / "song_package.yaml"
    bars_parquet = song_dir / "bars.parquet"
    
    if not pkg_yaml.exists():
        raise FileNotFoundError(f"song_package.yaml not found in {song_dir}")
    if not bars_parquet.exists():
        raise FileNotFoundError(f"bars.parquet not found in {song_dir}")
    
    with open(pkg_yaml, 'r', encoding='utf-8') as f:
        pkg_data = yaml.safe_load(f)
    
    bars = pd.read_parquet(bars_parquet)
    
    # pkg_data 全体を返す（meta, artifacts, etc.含む）
    return {**pkg_data, "bars": bars}

def load_matches(matches_path: Path) -> Dict[str, Any]:
    """Matcher結果読み込み"""
    with open(matches_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_drums_recommendations(bars_df: pd.DataFrame, matches: Dict) -> Dict:
    """ドラム推奨プラン生成（KPI Gate互換形式）"""
    top1 = matches["matches"][0] if len(matches["matches"]) > 0 else None
    
    if top1 is None:
        raise RuntimeError("No rhythm matches found")
    
    recommendations = []
    for idx, row in bars_df.iterrows():
        recommendations.append({
            "bar": int(idx),
            "pattern_id": top1["loop_id"],
            "family": top1["family"],
            "tempo_bpm": float(top1["tempo_bpm"]),
            "density": float(row.get("density_target", 4.0)),
            "swing": float(row.get("swing_target", 0.0)),
            "energy": float(row.get("energy_curve", 0.5)),
            "section": str(row.get("section_label", "verse"))
        })
    
    return {
        "metadata": {
            "source": "suno_arranger",
            "top_match": top1["loop_id"],
            "total_bars": len(bars_df)
        },
        "recommendations": recommendations
    }

def generate_bass_plan(bars_df: pd.DataFrame, top_match: Dict, chordmap: List[Dict], song_tempo_bpm: float) -> Dict:
    """Bass Plan生成（Writer互換events形式）
    
    Args:
        bars_df: bars DataFrame
        top_match: Top rhythm match
        chordmap: Chord progression
        song_tempo_bpm: SongPackageの正確なテンポ
    """
    family = top_match.get("family", "STRAIGHT_8")
    
    # chordmapからコード配列抽出（bar単位）
    # chordmapのeventsは{"time": QL単位, "root": "F", "quality": "m7"}形式
    bar_chords = {}
    for ev in chordmap:
        time_ql = float(ev.get("time", 0))
        bar_idx = int(time_ql // 4)  # QL→bar変換（4QL = 1bar）
        root = ev.get("root", "C")
        quality = ev.get("quality", "")
        chord_sym = f"{root}{quality}" if quality else root
        if bar_idx not in bar_chords:
            bar_chords[bar_idx] = chord_sym
    
    # rootピッチ辞書（仮、voicing_engineのSEMITONESと同等）
    root_map = {
        "C":0, "C#":1, "Db":1, "D":2, "D#":3, "Eb":3, "E":4,
        "F":5, "F#":6, "Gb":6, "G":7, "G#":8, "Ab":8, "A":9,
        "A#":10, "Bb":10, "B":11
    }
    
    events = []
    for idx, row in bars_df.iterrows():
        energy = float(row.get("energy_curve", 0.5))
        section = str(row.get("section_label", "verse"))
        
        # コード取得（bar_chords優先、なければ前小節継続、それもなければC）
        chord_sym = bar_chords.get(idx, bar_chords.get(idx-1, "C") if idx > 0 else "C")
        root_note = None
        for k in root_map:
            if chord_sym.startswith(k):
                root_note = k
                break
        root_pitch = 36 + root_map.get(root_note or "C", 0)  # E1 (28) 〜 C2 (36) ベース帯
        
        # family別パターン（簡易）
        if family in ("STRAIGHT_8", "SWING_8"):
            # 4分 + 8分混合（energy依存）
            beats = [1.0, 2.0, 3.0, 4.0] if energy < 0.5 else [1.0, 1.5, 2.0, 3.0, 3.5, 4.0]
        else:
            # 16分混合
            beats = [1.0, 1.25, 2.0, 2.75, 3.0, 4.0] if energy > 0.6 else [1.0, 2.0, 3.0, 4.0]
        
        vel = int(85 + 10 * energy)
        for b in beats:
            events.append({
                "bar": int(idx),
                "beat": b,
                "pitch": root_pitch,
                "dur_beats": 0.25,
                "vel": vel
            })
    
    return {
        "ppq": 480,
        "tempo_bpm": song_tempo_bpm,  # SongPackageの正確なテンポを使用
        "tracks": [{
            "name": "Bass",
            "role": "bass",
            "channel": 1,
            "program": 33,  # Acoustic Bass
            "events": events
        }]
    }

def generate_guitar_plan(bars_df: pd.DataFrame, top_match: Dict, chordmap: List[Dict], song_tempo_bpm: float) -> Dict:
    """Guitar Plan生成（Writer互換events形式 - chord+voicing）
    
    Args:
        bars_df: bars DataFrame
        top_match: Top rhythm match
        chordmap: Chord progression
        song_tempo_bpm: SongPackageの正確なテンポ
    """
    family = top_match.get("family", "STRAIGHT_8")
    
    # chordmapからコード配列抽出（bar単位）
    # chordmapのeventsは{"time": QL単位, "root": "F", "quality": "m7"}形式
    bar_chords = {}
    for ev in chordmap:
        time_ql = float(ev.get("time", 0))
        bar_idx = int(time_ql // 4)  # QL→bar変換（4QL = 1bar）
        root = ev.get("root", "C")
        quality = ev.get("quality", "")
        chord_sym = f"{root}{quality}" if quality else root
        if bar_idx not in bar_chords:
            bar_chords[bar_idx] = chord_sym
    
    events = []
    for idx, row in bars_df.iterrows():
        energy = float(row.get("energy_curve", 0.5))
        # コード取得（bar_chords優先、なければ前小節継続、それもなければC）
        chord_sym = bar_chords.get(idx, bar_chords.get(idx-1, "C") if idx > 0 else "C")
        
        # family別ストラムパターン（簡易：4分2回 or 8分4回）
        if family in ("STRAIGHT_8", "SWING_8"):
            beats = [1.0, 2.0, 3.0, 4.0]
        else:
            beats = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        
        vel = int(80 + 15 * energy)
        for b in beats:
            events.append({
                "bar": int(idx),
                "beat": b,
                "chord": chord_sym,
                "voicing": {"style": "close", "octave": 4},
                "dur_beats": 0.25,
                "vel": vel,
                "arp_ms": 0  # アルペジオなし（同時発音）
            })
    
    return {
        "ppq": 480,
        "tempo_bpm": song_tempo_bpm,  # SongPackageの正確なテンポを使用
        "tracks": [{
            "name": "Guitar",
            "role": "guitar",
            "channel": 2,
            "program": 24,  # Nylon Guitar
            "events": events
        }]
    }

def generate_arrangement_plan(bars_df: pd.DataFrame) -> Dict:
    """総合アレンジメントプラン（bars.parquetベース）"""
    plan = []
    for idx, row in bars_df.iterrows():
        plan.append({
            "bar": int(idx),
            "section": str(row.get("section_label", "verse")),
            "energy": float(row.get("energy_curve", 0.5)),
            "accent_target": float(row.get("accent_score_target", 0.5)),
            "density_target": float(row.get("density_target", 4.0)),
            "swing_target": float(row.get("swing_target", 0.0))
        })
    
    return {
        "metadata": {
            "source": "suno_arranger",
            "total_bars": len(bars_df)
        },
        "plan": plan
    }

def main():
    ap = argparse.ArgumentParser(description="Suno Arranger - Generate Part Plans")
    ap.add_argument("--song-dir", type=Path, required=True, help="SongPackage directory")
    ap.add_argument("--matches", type=Path, required=True, help="matches_rhythm.json path")
    ap.add_argument("--emit-drums", action="store_true", help="Emit drums_recommendations.json")
    ap.add_argument("--emit-bass-plan", action="store_true", help="Emit bass_plan.json")
    ap.add_argument("--emit-guitar-plan", action="store_true", help="Emit guitar_plan.json")
    ap.add_argument("--emit-aggregate", action="store_true", help="Emit arrangement_plan.json")
    args = ap.parse_args()
    
    pkg = load_song_package(args.song_dir)
    bars_df = pkg["bars"]
    matches = load_matches(args.matches)
    
    print(f"📂 SongPackage: {args.song_dir.name}")
    print(f"📊 Bars: {len(bars_df)}")
    print(f"🎯 Top Match: {matches['matches'][0]['loop_id'] if len(matches['matches']) > 0 else 'N/A'}")
    print()
    
    outputs = []
    
    # SongPackageの正確なテンポを取得
    meta = pkg["meta"]
    song_tempo_bpm = float(meta.get("tempo_bpm", meta.get("bpm", 120.0)))
    
    if args.emit_drums:
        drums_plan = generate_drums_recommendations(bars_df, matches)
        drums_path = args.song_dir / "drums_recommendations.json"
        with open(drums_path, 'w', encoding='utf-8') as f:
            json.dump(drums_plan, f, indent=2, ensure_ascii=False)
        outputs.append(str(drums_path.name))
        print(f"✅ drums_recommendations.json: {len(drums_plan['recommendations'])} bars")
    
    if args.emit_bass_plan:
        top_match = matches['matches'][0] if len(matches['matches']) > 0 else {}
        chordmap = json.loads((args.song_dir / "chordmap.json").read_text(encoding='utf-8')).get("events", [])
        bass_plan = generate_bass_plan(bars_df, top_match, chordmap, song_tempo_bpm)
        bass_path = args.song_dir / "bass_plan.json"
        with open(bass_path, 'w', encoding='utf-8') as f:
            json.dump(bass_plan, f, indent=2, ensure_ascii=False)
        outputs.append(str(bass_path.name))
        print(f"✅ bass_plan.json: {len(bass_plan['tracks'][0]['events'])} events")
    
    if args.emit_guitar_plan:
        top_match = matches['matches'][0] if len(matches['matches']) > 0 else {}
        chordmap = json.loads((args.song_dir / "chordmap.json").read_text(encoding='utf-8')).get("events", [])
        guitar_plan = generate_guitar_plan(bars_df, top_match, chordmap, song_tempo_bpm)
        guitar_path = args.song_dir / "guitar_plan.json"
        with open(guitar_path, 'w', encoding='utf-8') as f:
            json.dump(guitar_plan, f, indent=2, ensure_ascii=False)
        outputs.append(str(guitar_path.name))
        print(f"✅ guitar_plan.json: {len(guitar_plan['tracks'][0]['events'])} events")
    
    if args.emit_aggregate:
        arr_plan = generate_arrangement_plan(bars_df)
        arr_path = args.song_dir / "arrangement_plan.json"
        with open(arr_path, 'w', encoding='utf-8') as f:
            json.dump(arr_plan, f, indent=2, ensure_ascii=False)
        outputs.append(str(arr_path.name))
        print(f"✅ arrangement_plan.json: {len(arr_plan['plan'])} bars")
    
    print(f"\n🎉 Generated: {', '.join(outputs)}")

if __name__ == "__main__":
    main()
