#!/usr/bin/env python3
"""
Plan Schema Check - P0 Type Guard for Phase 123

修正内容：
1. dur_beats正規化（dur_beats | dur | end_beats-start_beats）
2. ≤0のduration除去（最小値MIN_DUR_BEATSに丸め）
3. 拍子・テンポ存在チェック
4. CI検証項目「zero/negative duration = 0件」追加

使い方：
  python ops/plan_schema_check.py <plan.json> [--fix]
    --fix: 修正版を <plan>_normalized.json に出力

Exit codes:
  0: 正常（エラーなし）
  1: スキーマエラー検出（--fix未指定時）
  2: 引数エラー
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 最小duration（beats）: 極小ノート防止
MIN_DUR_BEATS = 0.01

def normalize_dur_beats(event: Dict[str, Any]) -> Tuple[float, str]:
    """
    イベントのdurationを正規化（dur_beats | dur | end_beats-start_beats）
    
    Returns:
        (dur_beats, source): 正規化されたduration（beats）とソース名
    """
    # 優先順位1: dur_beats
    if "dur_beats" in event and event["dur_beats"] is not None:
        val = float(event["dur_beats"])
        if val > 0:
            return val, "dur_beats"
    
    # 優先順位2: dur
    if "dur" in event and event["dur"] is not None:
        val = float(event["dur"])
        if val > 0:
            return val, "dur"
    
    # 優先順位3: end_beats - start_beats
    if "end_beats" in event and "start_beats" in event:
        if event["end_beats"] is not None and event["start_beats"] is not None:
            val = float(event["end_beats"]) - float(event["start_beats"])
            if val > 0:
                return val, "end_beats-start_beats"
    
    # フォールバック: 最小値
    return MIN_DUR_BEATS, "fallback_min"

def check_plan_schema(plan_path: Path, fix: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """
    Plan JSON のスキーマ検証・正規化
    
    Returns:
        (ok, report): ok=検証成功、report=検証結果詳細
    """
    if not plan_path.exists():
        return False, {"error": f"File not found: {plan_path}"}
    
    try:
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, {"error": f"JSON parse error: {e}"}
    
    report = {
        "file": str(plan_path),
        "status": "OK",
        "errors": [],
        "warnings": [],
        "stats": {
            "total_events": 0,
            "normalized_events": 0,
            "zero_or_negative_duration": 0,
            "missing_time_signature": False,
            "missing_tempo": False,
        }
    }
    
    # 拍子チェック（bars.parquet前提）
    bars_path = plan_path.parent / "bars.parquet"
    if not bars_path.exists():
        report["errors"].append(f"Missing bars.parquet (required for time_signature)")
        report["stats"]["missing_time_signature"] = True
    
    # テンポチェック（tempo_bpm: トップレベル or meta.tempo_bpm）
    tempo_bpm = plan_data.get("tempo_bpm") or (plan_data.get("meta", {}) or {}).get("tempo_bpm")
    if not tempo_bpm:
        report["errors"].append("Missing tempo_bpm (check top-level or meta.tempo_bpm)")
        report["stats"]["missing_tempo"] = True
    
    # イベント正規化
    # full_arrangement構造: {"tracks": [{"name": "Bass", "events": [...]}, ...]}
    # 旧構造: {"Bass": [...], "Guitar": [...], ...}
    tracks_list = plan_data.get("tracks", [])
    if isinstance(tracks_list, list):
        # tracks形式
        for track in tracks_list:
            if not isinstance(track, dict):
                continue
            part_key = track.get("name", "Unknown")
            events = track.get("events", [])
            if not isinstance(events, list):
                continue
            
            for i, ev in enumerate(events):
                report["stats"]["total_events"] += 1
                
                # dur_beats正規化
                original_dur = ev.get("dur_beats")
                normalized_dur, source = normalize_dur_beats(ev)
                
                if normalized_dur <= 0:
                    report["stats"]["zero_or_negative_duration"] += 1
                    report["warnings"].append(
                        f"{part_key}[{i}]: Zero/negative duration detected "
                        f"(original={original_dur}, source={source}), "
                        f"clamped to MIN_DUR_BEATS={MIN_DUR_BEATS}"
                    )
                    normalized_dur = MIN_DUR_BEATS
                
                # 修正モード
                if fix:
                    if source != "dur_beats" or ev.get("dur_beats") != normalized_dur:
                        ev["dur_beats"] = normalized_dur
                        report["stats"]["normalized_events"] += 1
                    
                    # duration_beats削除（キー統一）
                    if "duration_beats" in ev:
                        del ev["duration_beats"]
    else:
        # 旧形式: {Bass: [...], Guitar: [...], ...}
        for part_key in ["Bass", "Guitar", "Piano", "Strings", "Drums"]:
            if part_key not in plan_data:
                continue
            
            events = plan_data[part_key]
            if not isinstance(events, list):
                continue
            
            for i, ev in enumerate(events):
                report["stats"]["total_events"] += 1
                
                # dur_beats正規化
                original_dur = ev.get("dur_beats")
                normalized_dur, source = normalize_dur_beats(ev)
                
                if normalized_dur <= 0:
                    report["stats"]["zero_or_negative_duration"] += 1
                    report["warnings"].append(
                        f"{part_key}[{i}]: Zero/negative duration detected "
                        f"(original={original_dur}, source={source}), "
                        f"clamped to MIN_DUR_BEATS={MIN_DUR_BEATS}"
                    )
                    normalized_dur = MIN_DUR_BEATS
                
                # 修正モード
                if fix:
                    if source != "dur_beats" or ev.get("dur_beats") != normalized_dur:
                        ev["dur_beats"] = normalized_dur
                        report["stats"]["normalized_events"] += 1
                    
                    # duration_beats削除（キー統一）
                    if "duration_beats" in ev:
                        del ev["duration_beats"]
    
    # エラー判定
    if report["errors"] or report["stats"]["zero_or_negative_duration"] > 0:
        report["status"] = "FAIL"
    
    # 修正版出力
    if fix and report["stats"]["normalized_events"] > 0:
        output_path = plan_path.parent / f"{plan_path.stem}_normalized.json"
        output_path.write_text(json.dumps(plan_data, indent=2, ensure_ascii=False), encoding="utf-8")
        report["fixed_file"] = str(output_path)
    
    return report["status"] == "OK", report

def main():
    parser = argparse.ArgumentParser(description="Plan Schema Check - P0 Type Guard")
    parser.add_argument("plan_json", help="Path to plan JSON file")
    parser.add_argument("--fix", action="store_true", help="Fix and output normalized plan")
    args = parser.parse_args()
    
    plan_path = Path(args.plan_json)
    ok, report = check_plan_schema(plan_path, fix=args.fix)
    
    # レポート出力
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # Exit code
    if not ok:
        if args.fix:
            print(f"\n✅ Fixed version saved: {report.get('fixed_file', 'N/A')}", file=sys.stderr)
        else:
            print("\n❌ Schema errors detected. Use --fix to normalize.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\n✅ Plan schema OK", file=sys.stderr)
        sys.exit(0)

if __name__ == "__main__":
    main()
