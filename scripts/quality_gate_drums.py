#!/usr/bin/env python3
"""
Drum Quality Gate Checker

ドラムパターンの品質ゲートチェック。
extract_drum_patterns.py のメトリクスと structure.yaml の quality_gates.drums を連携。

Usage:
    from scripts.quality_gate_drums import check_drum_pattern_quality
    
    pattern = DrumPattern(...)
    passed, failures = check_drum_pattern_quality(
        pattern,
        gates_yaml="configs/structure_template.yaml"
    )
    
    if not passed:
        print(f"Pattern failed: {failures}")
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import yaml

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generator.drums_generator_stage2 import DrumPattern  # noqa: E402


# ハイハット開閉のGM MIDI pitch（drum_map_registry.pyから取得）
HIHAT_CLOSED_PITCH = 42  # chh (Closed Hi-Hat)
HIHAT_OPEN_PITCH = 46    # ohh (Open Hi-Hat)
HIHAT_PEDAL_PITCH = 44   # hh_pedal (Pedal Hi-Hat)

# クラッシュシンバルのGM MIDI pitch
CRASH_PITCH_1 = 49       # Crash Cymbal 1
CRASH_PITCH_2 = 57       # Crash Cymbal 2


def load_drum_gates(yaml_path: str | Path) -> Dict[str, Any]:
    """
    YAMLファイルからドラム用品質ゲートをロード。
    
    Args:
        yaml_path: structure.yaml or quality_gates.yaml のパス
    
    Returns:
        drums の品質ゲート辞書
    
    Example:
        gates = load_drum_gates("configs/structure_template.yaml")
        print(gates["kick_onbeat_ratio_min"])  # 0.6
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # structure.yaml の場合
    if "quality_gates" in data:
        return data["quality_gates"].get("drums", {})
    
    # quality_gates.yaml の場合（フラット構造）
    return data.get("drums", {})


def extract_pattern_metrics(pattern: DrumPattern) -> Dict[str, float]:
    """
    DrumPattern から品質メトリクスを抽出。
    
    Args:
        pattern: DrumPattern インスタンス
    
    Returns:
        メトリクス辞書
    
    Metrics:
        - kick_onbeat_ratio: キックの拍頭率
        - ghost_note_ratio: ゴーストノート率
        - complexity: パターン複雑度
        - density: 密度（hits/bar）
        - syncopation_rate: シンコペーション率
        - quality_score: 総合品質スコア
        - notes_per_bar: 1小節あたりヒット数
    """
    # DrumPatternから直接取得できるメトリクス
    metrics = {
        "complexity": pattern.complexity if hasattr(pattern, "complexity") else 0.0,
        "density": pattern.density if hasattr(pattern, "density") else 0.0,
        "syncopation_rate": pattern.syncopation_rate if hasattr(pattern, "syncopation_rate") else 0.0,
        "quality_score": pattern.quality_score if hasattr(pattern, "quality_score") else 0.0,
    }
    
    # notes_per_bar を計算（全ヒット数 / バー数）
    total_hits = (
        len(pattern.kick_hits) +
        len(pattern.snare_hits) +
        len(pattern.hihat_hits) +
        len(pattern.crash_hits) +
        len(pattern.ride_hits)
    )
    metrics["notes_per_bar"] = total_hits / pattern.bars if pattern.bars > 0 else 0.0
    
    # kick_onbeat_ratio を計算（拍頭のキック数 / 全キック数）
    if pattern.kick_hits:
        onbeat_kicks = sum(1 for pos in pattern.kick_hits if abs(pos % 1.0) < 0.1)
        metrics["kick_onbeat_ratio"] = onbeat_kicks / len(pattern.kick_hits)
    else:
        metrics["kick_onbeat_ratio"] = 0.0
    
    # ghost_note_ratio を計算（velocity < 60 の割合）
    all_velocities = (
        pattern.kick_velocities +
        pattern.snare_velocities +
        pattern.hihat_velocities +
        pattern.crash_velocities +
        pattern.ride_velocities
    )
    if all_velocities:
        ghost_notes = sum(1 for v in all_velocities if v < 60)
        metrics["ghost_note_ratio"] = ghost_notes / len(all_velocities)
    else:
        metrics["ghost_note_ratio"] = 0.0
    
    return metrics


def check_drum_gates(metrics: Dict[str, float], gates: Dict[str, Any]) -> List[str]:
    """
    メトリクスを品質ゲートと照合。
    
    Args:
        metrics: パターンメトリクス
        gates: 品質ゲート設定
    
    Returns:
        失敗メッセージのリスト（空ならPASS）
    
    Gate operators:
        - kick_onbeat_ratio_min: >= 最小値
        - ghost_note_ratio_max: <= 最大値
        - notes_per_bar_range: [min, max]
        - complexity_range: [min, max]
        - syncopation_rate_max: <= 最大値
        - density_range: [min, max]
        - quality_score_min: >= 最小値
    
    Example:
        >>> metrics = {"quality_score": 0.4, "density": 2.0}
        >>> gates = {"quality_score_min": 0.5, "density_range": [4.0, 32.0]}
        >>> check_drum_gates(metrics, gates)
        ['quality_score: 0.40 < 0.50 (min)', 'density: 2.00 not in [4.00, 32.00]']
    """
    fails = []
    
    # kick_onbeat_ratio_min
    if "kick_onbeat_ratio_min" in gates:
        min_val = gates["kick_onbeat_ratio_min"]
        actual = metrics.get("kick_onbeat_ratio", 0.0)
        if actual < min_val:
            fails.append(f"kick_onbeat_ratio: {actual:.2f} < {min_val:.2f} (min)")
    
    # ghost_note_ratio_max
    if "ghost_note_ratio_max" in gates:
        max_val = gates["ghost_note_ratio_max"]
        actual = metrics.get("ghost_note_ratio", 0.0)
        if actual > max_val:
            fails.append(f"ghost_note_ratio: {actual:.2f} > {max_val:.2f} (max)")
    
    # syncopation_rate_max
    if "syncopation_rate_max" in gates:
        max_val = gates["syncopation_rate_max"]
        actual = metrics.get("syncopation_rate", 0.0)
        if actual > max_val:
            fails.append(f"syncopation_rate: {actual:.2f} > {max_val:.2f} (max)")
    
    # quality_score_min
    if "quality_score_min" in gates:
        min_val = gates["quality_score_min"]
        actual = metrics.get("quality_score", 0.0)
        if actual < min_val:
            fails.append(f"quality_score: {actual:.2f} < {min_val:.2f} (min)")
    
    # notes_per_bar_range
    if "notes_per_bar_range" in gates:
        lo, hi = gates["notes_per_bar_range"]
        actual = metrics.get("notes_per_bar", 0.0)
        if not (lo <= actual <= hi):
            fails.append(f"notes_per_bar: {actual:.2f} not in [{lo:.2f}, {hi:.2f}]")
    
    # complexity_range
    if "complexity_range" in gates:
        lo, hi = gates["complexity_range"]
        actual = metrics.get("complexity", 0.0)
        if not (lo <= actual <= hi):
            fails.append(f"complexity: {actual:.2f} not in [{lo:.2f}, {hi:.2f}]")
    
    # density_range
    if "density_range" in gates:
        lo, hi = gates["density_range"]
        actual = metrics.get("density", 0.0)
        if not (lo <= actual <= hi):
            fails.append(f"density: {actual:.2f} not in [{lo:.2f}, {hi:.2f}]")
    
    return fails


def check_hihat_exclusivity(
    hihat_hits: List[float],
    hihat_pitches: List[int],
    tolerance: float = 0.05
) -> List[str]:
    """
    ハイハットのOpen/Closed相互排他チェック。
    
    Args:
        hihat_hits: ハイハットノートのタイミング（quarter beats）
        hihat_pitches: 各ハイハットノートのMIDI pitch
        tolerance: 同時発音判定の許容誤差（quarter beats）
    
    Returns:
        違反メッセージのリスト（空ならPASS）
    
    Logic:
        同一タイミング（±tolerance）でOpen（pitch=46）とClosed（pitch=42）が
        同時に発音されている場合、物理的に不可能なため違反とする。
        
        Pedal（pitch=44）は別物なので除外しない。
    
    Example:
        >>> # FAIL: Open と Closed が同時発音
        >>> hits = [0.0, 0.0, 1.0]
        >>> pitches = [46, 42, 46]  # Open, Closed, Open
        >>> check_hihat_exclusivity(hits, pitches)
        ['Hi-Hat Open/Closed conflict at time 0.00 (Open: 46, Closed: 42)']
        
        >>> # PASS: 時間差がある
        >>> hits = [0.0, 0.1, 1.0]
        >>> pitches = [46, 42, 46]
        >>> check_hihat_exclusivity(hits, pitches)
        []
    """
    violations = []
    
    if not hihat_hits or not hihat_pitches:
        return violations
    
    if len(hihat_hits) != len(hihat_pitches):
        violations.append(f"Mismatched hihat_hits ({len(hihat_hits)}) and hihat_pitches ({len(hihat_pitches)})")
        return violations
    
    # タイミング別にグループ化（±tolerance）
    time_groups: Dict[float, List[int]] = {}
    for time, pitch in zip(hihat_hits, hihat_pitches):
        # 既存のグループに近いタイミングがあるか探す
        matched = False
        for group_time in list(time_groups.keys()):
            if abs(time - group_time) <= tolerance:
                time_groups[group_time].append(pitch)
                matched = True
                break
        
        if not matched:
            time_groups[time] = [pitch]
    
    # 各グループでOpen/Closed同時発音をチェック
    for time, pitches_at_time in time_groups.items():
        has_open = HIHAT_OPEN_PITCH in pitches_at_time
        has_closed = HIHAT_CLOSED_PITCH in pitches_at_time
        
        if has_open and has_closed:
            violations.append(
                f"Hi-Hat Open/Closed conflict at time {time:.2f} "
                f"(Open: {HIHAT_OPEN_PITCH}, Closed: {HIHAT_CLOSED_PITCH})"
            )
    
    return violations


def check_crash_choke_duration(
    crash_hits: List[float],
    crash_durations: List[float],
    max_duration_ms: float = 500.0,
    tempo: float = 120.0
) -> List[str]:
    """
    クラッシュシンバルのチョーク（短いmute）長制限チェック。
    
    Args:
        crash_hits: クラッシュノートのタイミング（quarter beats）
        crash_durations: 各クラッシュノートの長さ（quarter beats）
        max_duration_ms: チョーク最大長（ミリ秒）
        tempo: テンポ（BPM）
    
    Returns:
        違反メッセージのリスト（空ならPASS）
    
    Logic:
        チョーク（choke）は短い消音で、通常500ms以下。
        それ以上長い場合は通常のクラッシュとして扱うべき。
    
    Example:
        >>> # FAIL: 1秒のチョーク（長すぎ）
        >>> hits = [0.0]
        >>> durations = [2.0]  # 2 quarter beats @ 120 BPM = 1000ms
        >>> check_crash_choke_duration(hits, durations, max_duration_ms=500.0, tempo=120.0)
        ['Crash choke duration too long at time 0.00: 1000.0ms > 500.0ms max']
        
        >>> # PASS: 200msのチョーク
        >>> durations = [0.4]  # 0.4 quarter beats @ 120 BPM = 200ms
        >>> check_crash_choke_duration(hits, durations, max_duration_ms=500.0, tempo=120.0)
        []
    """
    violations = []
    
    if not crash_hits or not crash_durations:
        return violations
    
    if len(crash_hits) != len(crash_durations):
        violations.append(f"Mismatched crash_hits ({len(crash_hits)}) and crash_durations ({len(crash_durations)})")
        return violations
    
    # Quarter beats → ミリ秒変換
    # 1 quarter beat = (60 / tempo) seconds = (60000 / tempo) ms
    quarter_to_ms = 60000.0 / tempo
    
    for time, duration_qb in zip(crash_hits, crash_durations):
        duration_ms = duration_qb * quarter_to_ms
        
        # 短いノート（チョーク候補）のみチェック
        # 非常に長いクラッシュ（> max * 2）はチェック対象外
        if duration_ms <= max_duration_ms * 2:  # チョーク候補の範囲
            if duration_ms > max_duration_ms:
                violations.append(
                    f"Crash choke duration too long at time {time:.2f}: "
                    f"{duration_ms:.1f}ms > {max_duration_ms:.1f}ms max"
                )
    
    return violations


def check_drum_pattern_quality(
    pattern: DrumPattern,
    gates_yaml: str | Path = "configs/structure_template.yaml",
    verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    ドラムパターンの品質ゲートチェック（高レベルAPI）。
    
    Args:
        pattern: DrumPattern インスタンス
        gates_yaml: 品質ゲート設定ファイル
        verbose: 詳細出力
    
    Returns:
        (passed, failures) タプル
    
    Example:
        >>> from generator.drums_generator_stage2 import DrumPattern
        >>> pattern = DrumPattern(...)
        >>> passed, fails = check_drum_pattern_quality(pattern, verbose=True)
        >>> if not passed:
        ...     print(f"❌ Pattern rejected: {fails}")
    """
    gates = load_drum_gates(gates_yaml)
    
    if not gates:
        if verbose:
            print("[Drum Quality Gate] No gates configured. PASS by default.")
        return (True, [])
    
    metrics = extract_pattern_metrics(pattern)
    fails = check_drum_gates(metrics, gates)
    
    # ハイハット開閉整合性チェック（Todo #7）
    if gates.get("hihat_open_close_exclusive", False):
        # hihat_hits と pitches を結合してチェック
        # DrumPattern は kick/snare/hihat 別々に保存されているため、結合が必要
        hihat_hits = list(pattern.hihat_hits)
        hihat_pitches = list(pattern.hihat_pitches) if hasattr(pattern, "hihat_pitches") else []
        
        if hihat_hits and hihat_pitches:
            hihat_violations = check_hihat_exclusivity(hihat_hits, hihat_pitches)
            fails.extend(hihat_violations)
        elif hihat_hits and not hihat_pitches:
            # pitches が未定義の場合は警告のみ
            if verbose:
                print("[Warning] hihat_hits exists but hihat_pitches is missing. Skipping hihat exclusivity check.")
    
    # クラッシュチョーク長制限チェック（Todo #7）
    if "crash_choke_max_duration_ms" in gates and gates.get("crash_choke_max_duration_ms", 0) > 0:
        crash_hits = list(pattern.crash_hits)
        crash_durations = list(pattern.crash_durations) if hasattr(pattern, "crash_durations") else []
        
        if crash_hits and crash_durations:
            crash_violations = check_crash_choke_duration(
                crash_hits,
                crash_durations,
                max_duration_ms=gates["crash_choke_max_duration_ms"],
                tempo=pattern.tempo
            )
            fails.extend(crash_violations)
        elif crash_hits and not crash_durations:
            # durations が未定義の場合は警告のみ
            if verbose:
                print("[Warning] crash_hits exists but crash_durations is missing. Skipping crash choke check.")
    
    if verbose:
        print(f"[Drum Quality Gate]")
        print(f"  Pattern: tempo={pattern.tempo:.1f}, bars={pattern.bars}, bpm_range={pattern.bpm_range}")
        print(f"  Metrics:")
        for name, value in metrics.items():
            print(f"    {name:25s}: {value:.3f}")
        print(f"  Gates: {len(gates)} rules")
        print(f"  Result: {'✅ PASS' if not fails else '❌ FAIL'}")
        if fails:
            for fail in fails:
                print(f"    - {fail}")
    
    return (len(fails) == 0, fails)


def check_drum_batch_quality(
    patterns: List[DrumPattern],
    gates_yaml: str | Path = "configs/structure_template.yaml",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    複数パターンの品質ゲートチェック。
    
    Args:
        patterns: DrumPattern のリスト
        gates_yaml: 品質ゲート設定ファイル
        verbose: 詳細出力
    
    Returns:
        統計情報辞書
        {
            "total": 総数,
            "passed": 合格数,
            "failed": 不合格数,
            "pass_rate": 合格率,
            "failures": [(pattern_idx, failures), ...]
        }
    
    Example:
        >>> patterns = [...]  # List of DrumPattern
        >>> stats = check_drum_batch_quality(patterns, verbose=True)
        >>> print(f"Pass rate: {stats['pass_rate']:.1%}")
    """
    gates = load_drum_gates(gates_yaml)
    
    if not gates:
        return {
            "total": len(patterns),
            "passed": len(patterns),
            "failed": 0,
            "pass_rate": 1.0,
            "failures": []
        }
    
    passed_count = 0
    failed_patterns = []
    
    for idx, pattern in enumerate(patterns):
        metrics = extract_pattern_metrics(pattern)
        fails = check_drum_gates(metrics, gates)
        
        if not fails:
            passed_count += 1
        else:
            failed_patterns.append((idx, fails))
    
    stats = {
        "total": len(patterns),
        "passed": passed_count,
        "failed": len(failed_patterns),
        "pass_rate": passed_count / len(patterns) if patterns else 0.0,
        "failures": failed_patterns
    }
    
    if verbose:
        print(f"[Drum Batch Quality Gate]")
        print(f"  Total patterns: {stats['total']}")
        print(f"  Passed: {stats['passed']} ({stats['pass_rate']:.1%})")
        print(f"  Failed: {stats['failed']}")
        if failed_patterns:
            print(f"  Failed patterns:")
            for idx, fails in failed_patterns[:10]:  # 最初の10個のみ表示
                print(f"    #{idx}: {', '.join(fails)}")
    
    return stats


def cli_main():
    """CLI entry point for testing."""
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(description="Drum Quality Gate Checker")
    parser.add_argument("--pattern-pkl", required=True, help="Path to drum patterns pickle file")
    parser.add_argument("--gates-yaml", default="configs/structure_template.yaml", help="Path to quality gates YAML")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--show-first", type=int, default=5, help="Show first N patterns")
    
    args = parser.parse_args()
    
    # Load patterns
    with open(args.pattern_pkl, 'rb') as f:
        data = pickle.load(f)
    
    # Extract patterns from nested structure
    if isinstance(data, dict) and "patterns" in data:
        all_patterns = []
        if isinstance(data["patterns"], dict):
            # BPM-stratified format: {"slow": [...], "medium": [...], ...}
            for bpm_range, patterns in data["patterns"].items():
                all_patterns.extend(patterns)
        else:
            # Flat list format
            all_patterns = data["patterns"]
    else:
        # Already a list
        all_patterns = data
    
    print(f"Loaded {len(all_patterns)} patterns from {args.pattern_pkl}")
    print()
    
    # Check first N patterns individually
    print(f"=== Checking first {args.show_first} patterns ===")
    for idx in range(min(args.show_first, len(all_patterns))):
        pattern = all_patterns[idx]
        passed, fails = check_drum_pattern_quality(pattern, args.gates_yaml, verbose=args.verbose)
        
        status = "✅ PASS" if passed else f"❌ FAIL ({len(fails)})"
        print(f"Pattern #{idx}: tempo={pattern.tempo:.1f}, bars={pattern.bars}, quality={pattern.quality_score:.3f} → {status}")
        if not passed and not args.verbose:
            for fail in fails:
                print(f"  - {fail}")
    
    print()
    
    # Batch check all patterns
    print(f"=== Batch Quality Gate Check ({len(all_patterns)} patterns) ===")
    stats = check_drum_batch_quality(all_patterns, args.gates_yaml, verbose=True)
    
    print()
    print("="*60)
    if stats["pass_rate"] >= 0.8:
        print(f"✅ PASS: {stats['pass_rate']:.1%} patterns meet quality gates")
        return 0
    else:
        print(f"⚠️  WARNING: Only {stats['pass_rate']:.1%} patterns meet quality gates")
        return 1


if __name__ == "__main__":
    sys.exit(cli_main())
