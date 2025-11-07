#!/usr/bin/env python3
"""
EmotionAI効果測定ツール
=======================
セクション別velocity/density/swing分析

Usage:
  python ops/analyze_emotion_ai_effect.py \\
    --bass-plan song_001/bass_plan.json \\
    --guitar-plan song_001/guitar_plan.json \\
    --piano-plan song_001/piano_plan.json \\
    --strings-plan song_001/strings_plan.json \\
    --out emotion_ai_report.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_plan(plan_path: Path) -> Dict:
    """楽器planファイル読み込み"""
    with open(plan_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_section_stats(events: List[Dict]) -> Dict[str, Dict]:
    """
    セクション別統計分析
    
    Returns:
        {
            "intro": {"mean_velocity": 75.2, "note_count": 120, "density": 0.5, ...},
            "verse": {...},
            ...
        }
    """
    section_stats = defaultdict(lambda: {
        "velocities": [],
        "notes": [],
        "duration_sum": 0.0,
        "time_span": 0.0,
    })
    
    for event in events:
        section = event.get("section", "unknown")
        vel = event.get("velocity") or event.get("vel", 80)
        dur = event.get("dur", 0.5)
        
        section_stats[section]["velocities"].append(vel)
        section_stats[section]["notes"].append(event)
        section_stats[section]["duration_sum"] += dur
    
    # 統計計算
    result = {}
    for section, data in section_stats.items():
        vels = data["velocities"]
        notes = data["notes"]
        
        if not vels:
            continue
        
        # 時間範囲計算
        if notes:
            min_beat = min(n.get("start_beats", n.get("beat", 0)) for n in notes)
            max_beat = max(n.get("end_beats", n.get("beat", 0) + n.get("dur", 0.5)) for n in notes)
            time_span = max_beat - min_beat
        else:
            time_span = 0
        
        result[section] = {
            "mean_velocity": sum(vels) / len(vels),
            "std_velocity": pd.Series(vels).std(),
            "min_velocity": min(vels),
            "max_velocity": max(vels),
            "note_count": len(notes),
            "density_notes_per_beat": len(notes) / time_span if time_span > 0 else 0,
            "duration_sum": data["duration_sum"],
            "time_span_beats": time_span,
        }
    
    return result


def compare_sections(section_stats: Dict[str, Dict], instrument: str) -> Dict:
    """
    セクション間比較
    
    EmotionAI期待値:
    - intro: 控えめ（velocity低、density低）
    - verse: 標準（velocity中、density中）
    - chorus: 盛り上がり（velocity高、density高）
    - bridge: 変化（velocity中〜高、density中）
    - outro: フェードアウト（velocity低、density低）
    """
    if not section_stats:
        return {"status": "no_data"}
    
    # セクション順序定義
    expected_order = ["intro", "verse", "pre_chorus", "chorus", "bridge", "outro"]
    present_sections = [s for s in expected_order if s in section_stats]
    
    if not present_sections:
        present_sections = sorted(section_stats.keys())
    
    # 比較分析
    comparisons = []
    for i, section in enumerate(present_sections):
        stats = section_stats[section]
        
        # 期待値判定
        expected = {}
        if "intro" in section.lower():
            expected = {"velocity": "low", "density": "low"}
        elif "verse" in section.lower():
            expected = {"velocity": "medium", "density": "medium"}
        elif "chorus" in section.lower():
            expected = {"velocity": "high", "density": "high"}
        elif "bridge" in section.lower():
            expected = {"velocity": "medium-high", "density": "medium"}
        elif "outro" in section.lower():
            expected = {"velocity": "low", "density": "low"}
        
        comparisons.append({
            "section": section,
            "mean_velocity": round(stats["mean_velocity"], 1),
            "density": round(stats["density_notes_per_beat"], 2),
            "note_count": stats["note_count"],
            "expected": expected,
        })
    
    # velocity範囲判定
    all_vels = [c["mean_velocity"] for c in comparisons]
    vel_min, vel_max = min(all_vels), max(all_vels)
    vel_range = vel_max - vel_min
    
    # density範囲判定
    all_densities = [c["density"] for c in comparisons]
    density_min, density_max = min(all_densities), max(all_densities)
    density_range = density_max - density_min
    
    # EmotionAI効果判定
    emotion_ai_effect = {
        "velocity_range": round(vel_range, 1),
        "velocity_variation": "high" if vel_range > 20 else ("medium" if vel_range > 10 else "low"),
        "density_range": round(density_range, 2),
        "density_variation": "high" if density_range > 0.5 else ("medium" if density_range > 0.2 else "low"),
        "emotion_ai_detected": vel_range > 15 or density_range > 0.3,
        "interpretation": "",
    }
    
    if emotion_ai_effect["emotion_ai_detected"]:
        emotion_ai_effect["interpretation"] = (
            f"{instrument}: EmotionAI効果検出 "
            f"(velocity変化{vel_range:.1f}, density変化{density_range:.2f})"
        )
    else:
        emotion_ai_effect["interpretation"] = (
            f"{instrument}: EmotionAI効果微弱 "
            f"(velocity変化{vel_range:.1f}, density変化{density_range:.2f})"
        )
    
    return {
        "instrument": instrument,
        "sections": comparisons,
        "emotion_ai_effect": emotion_ai_effect,
    }


def main():
    parser = argparse.ArgumentParser(description="EmotionAI効果測定")
    parser.add_argument("--bass-plan", type=Path, help="bass_plan.json")
    parser.add_argument("--guitar-plan", type=Path, help="guitar_plan.json")
    parser.add_argument("--piano-plan", type=Path, help="piano_plan.json")
    parser.add_argument("--strings-plan", type=Path, help="strings_plan.json")
    parser.add_argument("--out", type=Path, required=True, help="出力JSONファイル")
    
    args = parser.parse_args()
    
    report = {
        "emotion_ai_analysis": {},
        "summary": {},
    }
    
    # 各楽器を分析
    instruments = [
        ("bass", args.bass_plan),
        ("guitar", args.guitar_plan),
        ("piano", args.piano_plan),
        ("strings", args.strings_plan),
    ]
    
    for instrument, plan_path in instruments:
        if not plan_path or not plan_path.exists():
            print(f"⚠️  {instrument}: plan file not found, skipping")
            continue
        
        print(f"📊 Analyzing {instrument}...")
        plan = load_plan(plan_path)
        
        # events取得
        events = []
        if "tracks" in plan:
            for track in plan["tracks"]:
                if "events" in track:
                    events.extend(track["events"])
        
        if not events:
            print(f"   ⚠️  No events found")
            continue
        
        print(f"   Total events: {len(events)}")
        
        # セクション別統計
        section_stats = analyze_section_stats(events)
        print(f"   Sections: {list(section_stats.keys())}")
        
        # セクション間比較
        comparison = compare_sections(section_stats, instrument)
        report["emotion_ai_analysis"][instrument] = comparison
        
        print(f"   {comparison['emotion_ai_effect']['interpretation']}")
    
    # 総合サマリー
    detected_count = sum(
        1 for inst_data in report["emotion_ai_analysis"].values()
        if inst_data.get("emotion_ai_effect", {}).get("emotion_ai_detected", False)
    )
    total_count = len(report["emotion_ai_analysis"])
    
    report["summary"] = {
        "total_instruments": total_count,
        "emotion_ai_detected_count": detected_count,
        "emotion_ai_detection_rate": detected_count / total_count if total_count > 0 else 0,
        "overall_assessment": (
            "EmotionAI効果: 高" if detected_count >= total_count * 0.75 else
            ("EmotionAI効果: 中" if detected_count >= total_count * 0.5 else
             "EmotionAI効果: 低")
        ),
    }
    
    # 保存
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Report saved: {args.out}")
    print(f"\n📈 Summary:")
    print(f"   Total instruments: {total_count}")
    print(f"   EmotionAI detected: {detected_count}/{total_count}")
    print(f"   {report['summary']['overall_assessment']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
