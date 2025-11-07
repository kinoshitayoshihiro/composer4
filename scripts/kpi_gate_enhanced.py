#!/usr/bin/env python3
"""
KPI Gate検証スクリプト（Enhanced版 - A/B/C/D対応）

A: ハット定義拡張（Ride/Shaker含む）
B: 小節境界ε余白（humanizeこぼれ防止）
C: 相対密度判定（bars.parquet統合）
D: Downbeats準拠小節切り

使用例:
    python3 scripts/kpi_gate_enhanced.py \
        --midi song_packages/suno_project/song_001/drums.mid \
        --bars song_packages/suno_project/song_001/bars.parquet \
        --gate-config configs/gate_prod.yaml \
        --downbeats --tempo-bpm 74.67 \
        --output kpi_gate_report.json
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
import numpy as np

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import pretty_midi

    PRETTYMIDI_AVAILABLE = True
except ImportError:
    PRETTYMIDI_AVAILABLE = False


def load_gate_config(yaml_path: Path) -> dict:
    """gate_prod.yaml読み込み"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _hat_pitches_from_config(gate_config: dict) -> Set[int]:
    """ハット密度にカウントするピッチ群を設定から取得"""
    default = [42, 44, 46, 51, 53, 59, 82]  # HH + Ride + Shaker
    return set(gate_config.get("drums", {}).get("kpi", {}).get("hat_pitches", default))


def extract_kpi_from_midi_enhanced(
    midi_path: Path,
    tempo_bpm: Optional[float] = None,
    hat_pitches: Optional[Set[int]] = None,
    bars_from_downbeats: bool = False,
    epsilon_sec: float = 0.0,
    bars_parquet: Optional[Path] = None,
    debug: bool = False,
) -> Dict[str, dict]:
    """MIDIファイルからKPI抽出（Enhanced版）

    Args:
        midi_path: MIDIファイルパス
        tempo_bpm: テンポ（Noneの場合はMIDIから抽出）
        hat_pitches: ハットとしてカウントするピッチセット
        bars_from_downbeats: 実downbeatsで小節切りするか
        epsilon_sec: 小節境界の余白（秒）

    Returns:
        bars_dict: {bar_0: {bar_index: 0, pattern: {...}}}
    """
    if not PRETTYMIDI_AVAILABLE:
        raise ImportError("pretty_midi required. Install: pip install pretty_midi")

    midi = pretty_midi.PrettyMIDI(str(midi_path))

    # テンポ取得
    change_times, tempi = midi.get_tempo_changes()
    if tempo_bpm is None:
        tempo_bpm = float(tempi[0]) if len(tempi) > 0 else 120.0

    # ドラム楽器抽出
    drum_instruments = [inst for inst in midi.instruments if inst.is_drum]
    if not drum_instruments:
        raise ValueError(f"No drum track found in {midi_path}")

    # 全ドラムノート統合
    all_notes = []
    for inst in drum_instruments:
        all_notes.extend(inst.notes)
    all_notes.sort(key=lambda n: n.start)

    # ハット集合
    HAT = set(hat_pitches or {42, 44, 46, 51, 53, 59, 82})

    # 小節境界の余白
    eps = float(max(0.0, epsilon_sec))

    # 小節分割
    if bars_from_downbeats:
        try:
            downbeats = midi.get_downbeats()
        except Exception:
            downbeats = []

        if len(downbeats) >= 2:
            bar_ranges = [(downbeats[i], downbeats[i + 1]) for i in range(len(downbeats) - 1)]
        else:
            bars_from_downbeats = False  # フォールバック

    if not bars_from_downbeats:
        # Phase E: bars.parquetからtime_signature参照
        beats_per_bar_list = []
        if bars_parquet and bars_parquet.exists() and PANDAS_AVAILABLE:
            try:
                bars_df = pd.read_parquet(bars_parquet)
                if "time_signature" in bars_df.columns:
                    # 各小節のbeats_per_barを取得
                    for _, row in bars_df.iterrows():
                        ts = row.get("time_signature", "4/4")
                        numerator, _ = map(int, ts.split("/"))
                        beats_per_bar_list.append(float(numerator))
                    if debug:
                        print(
                            f"[DEBUG] Loaded {len(beats_per_bar_list)} bars with time_signature from bars.parquet"
                        )
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Failed to load time_signature from bars.parquet: {e}")

        # time_signature情報があれば活用、なければ4/4フォールバック
        if beats_per_bar_list:
            bar_ranges = []
            current_time = 0.0
            for beats_per_bar in beats_per_bar_list:
                bar_duration = 60.0 / tempo_bpm * beats_per_bar
                bar_ranges.append((current_time, current_time + bar_duration))
                current_time += bar_duration
        else:
            # 4/4前提フォールバック
            bar_duration = 60.0 / tempo_bpm * 4.0
            total_duration = midi.get_end_time()
            num_bars = int(np.ceil(total_duration / bar_duration))
            bar_ranges = [(i * bar_duration, (i + 1) * bar_duration) for i in range(num_bars)]

    bars_dict = {}

    for bar_idx, (bar_start, bar_end) in enumerate(bar_ranges):
        # この小節のノート抽出（境界±ε許容）
        bar_notes = [n for n in all_notes if (bar_start - eps) <= n.start < (bar_end + eps)]

        if not bar_notes:
            continue

        # 1. density: ハット密度（拡張定義）
        hat_notes = [n for n in bar_notes if n.pitch in HAT]
        density = len(hat_notes)

        # 2. notes_per_bar: 総ノート数
        notes_per_bar = len(bar_notes)

        # 3. backbeat_strength: Snare平均Velocity（参考値）
        snare_notes = [n for n in bar_notes if n.pitch in (38, 40)]
        backbeat_strength = (
            (np.mean([n.velocity for n in snare_notes]) / 127.0) if snare_notes else 0.0
        )

        # 4. swing: マイクロタイミング分散（簡易）
        swing = 0.0
        if len(hat_notes) >= 4:
            sec_per_beat = 60.0 / tempo_bpm
            eighth = sec_per_beat / 2.0
            swing_deviations = []

            for note in hat_notes:
                relative_time = note.start - bar_start
                nearest_grid = round(relative_time / eighth) * eighth
                deviation = abs(relative_time - nearest_grid)
                swing_deviations.append(deviation)

            swing = (np.mean(swing_deviations) / eighth) if swing_deviations else 0.0

        # 5. kick_downbeat_rate: 小節頭でのKick命中率
        kick_notes = [n for n in bar_notes if n.pitch in (35, 36)]
        downbeat_window = 0.045  # 45ms
        kick_on_db = [n for n in kick_notes if abs(n.start - bar_start) <= downbeat_window]
        kick_downbeat_rate = 1.0 if kick_on_db else 0.0

        # 6. snare_backbeat_acc: 2拍/4拍での命中率
        sec_per_beat = 60.0 / tempo_bpm
        beat_pos = [bar_start + i * sec_per_beat for i in range(4)]
        backbeat_window = 0.045  # 45ms

        backbeat_hits = 0
        backbeat_targets = 2  # beat 2 & 4
        for center in (beat_pos[1], beat_pos[3]):
            if any(abs(n.start - center) <= backbeat_window for n in snare_notes):
                backbeat_hits += 1
        snare_backbeat_acc = backbeat_hits / backbeat_targets if backbeat_targets > 0 else 0.0

        # パターン辞書作成
        pattern = {
            "density": float(density),
            "notes_per_bar": float(notes_per_bar),
            "backbeat_strength": float(backbeat_strength),
            "swing": float(swing),
            "tempo_bpm": float(tempo_bpm),
            "kick_downbeat_rate": float(kick_downbeat_rate),
            "snare_backbeat_acc": float(snare_backbeat_acc),
        }

        bars_dict[f"bar_{bar_idx}"] = {"bar_index": bar_idx, "pattern": pattern}

    return bars_dict


def validate_pattern_enhanced(
    pattern: dict,
    gate_config: dict,
    targets_by_bar: Optional[Dict[int, Dict[str, float]]] = None,
    bar_idx: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    """パターン検証（Enhanced版 - 相対判定+セクション別オーバーライド対応）

    Args:
        pattern: パターン辞書
        gate_config: gate設定
        targets_by_bar: bars.parquetから抽出したターゲット値（section_label含む）
        bar_idx: 小節インデックス

    Returns:
        (pass_flag, messages): 合格フラグとメッセージリスト
    """
    drums_config = gate_config.get("drums", {})
    messages = []
    all_pass = True

    # セクション情報取得
    section_label = ""
    if targets_by_bar is not None and bar_idx is not None:
        section_label = targets_by_bar.get(bar_idx, {}).get("section_label", "")

    # セクション別オーバーライド設定読み込み
    section_overrides = drums_config.get("section_overrides", {})
    epsilon_sec_overrides = section_overrides.get("epsilon_sec_override", {})
    min_rel_overrides = section_overrides.get("min_rel_override", {})
    min_notes_overrides = section_overrides.get("min_notes_per_bar_override", {})

    # 密度検証（相対運用に対応）
    if "density" in pattern:
        dens_cfg = drums_config.get("density", {})
        use_rel = bool(dens_cfg.get("use_relative", False))

        if use_rel and targets_by_bar is not None and bar_idx is not None:
            dens_target = targets_by_bar.get(bar_idx, {}).get("density_target", None)
            value = float(pattern["density"])

            if dens_target and dens_target > 0:
                ratio = value / float(dens_target)

                # セクション別オーバーライド適用
                min_rel = float(dens_cfg.get("min_rel", 0.45))
                if section_label in min_rel_overrides:
                    min_rel = float(min_rel_overrides[section_label])

                warn_rel = float(dens_cfg.get("warn_rel_low", 0.65))

                if ratio < min_rel:
                    messages.append(
                        f"density too low (relative): {ratio:.2f} < {min_rel} (target={dens_target:.1f}, actual={value:.1f})"
                    )
                    all_pass = False
                elif ratio < warn_rel:
                    messages.append(
                        f"density warning (relative low): {ratio:.2f} < {warn_rel} (target={dens_target:.1f}, actual={value:.1f})"
                    )
                else:
                    messages.append(
                        f"density OK (relative): {ratio:.2f} (target={dens_target:.1f}, actual={value:.1f})"
                    )
            else:
                # ターゲット不明時は絶対判定フォールバック
                value = float(pattern["density"])
                min_val = float(dens_cfg.get("min", 0.0))
                max_val = float(dens_cfg.get("max", 999.0))

                if value < min_val:
                    messages.append(f"density too low: {value:.2f} < {min_val}")
                    all_pass = False
                elif value > max_val:
                    messages.append(f"density too high: {value:.2f} > {max_val}")
                    all_pass = False
                else:
                    messages.append(f"density OK: {value:.2f}")
        else:
            # 絶対判定
            value = float(pattern["density"])
            min_val = float(dens_cfg.get("min", 0.0))
            max_val = float(dens_cfg.get("max", 999.0))

            if value < min_val:
                messages.append(f"density too low: {value:.2f} < {min_val}")
                all_pass = False
            elif value > max_val:
                messages.append(f"density too high: {value:.2f} > {max_val}")
                all_pass = False
            else:
                messages.append(f"density OK: {value:.2f}")

    # 他のメトリック検証（簡易版 + notes_per_barセクション別オーバーライド対応）
    for metric in [
        "swing",
        "backbeat_strength",
        "notes_per_bar",
        "kick_downbeat_rate",
        "snare_backbeat_acc",
    ]:
        if metric in pattern:
            cfg = drums_config.get(metric, {})
            value = float(pattern[metric])
            min_val = float(cfg.get("min", 0.0))
            max_val = float(cfg.get("max", 999.0))

            # notes_per_barのセクション別オーバーライド適用
            if metric == "notes_per_bar" and section_label in min_notes_overrides:
                min_val = float(min_notes_overrides[section_label])

            if value < min_val:
                messages.append(f"{metric} too low: {value:.2f} < {min_val}")
                all_pass = False
            elif value > max_val:
                messages.append(f"{metric} too high: {value:.2f} > {max_val}")
                all_pass = False
            else:
                messages.append(f"{metric} OK: {value:.2f}")

    return all_pass, messages


def kpi_gate_validate_enhanced(
    midi_path: Path,
    gate_config_path: Path,
    output_path: Path,
    tempo_bpm: Optional[float] = None,
    bars_parquet: Optional[Path] = None,
    use_downbeats: bool = True,
    epsilon_sec: Optional[float] = None,
    verbose: bool = True,
):
    """KPI Gate検証（Enhanced版）"""
    # gate_config読み込み
    gate_config = load_gate_config(gate_config_path)
    HAT = _hat_pitches_from_config(gate_config)

    if verbose:
        print(f"📖 Loading MIDI: {midi_path}")
        print(f"   Hat pitches: {sorted(HAT)}")

    # bars.parquet（相対判定用）
    targets_by_bar = None
    sections_json_path = None

    if bars_parquet and PANDAS_AVAILABLE:
        try:
            bars_df = pd.read_parquet(bars_parquet)
            section_col = "section_label" if "section_label" in bars_df.columns else "section"
            density_col = "density_target" if "density_target" in bars_df.columns else "density"

            targets_by_bar = {}
            for _, r in bars_df.iterrows():
                bar_idx = int(r["bar_index"])
                targets_by_bar[bar_idx] = {
                    "density_target": float(r.get(density_col, 0.0)),
                    "swing_target": float(r.get("swing_target", 0.0)),
                    "section_label": r.get(section_col, ""),
                }

            if verbose:
                print(f"   bars.parquet loaded: {len(targets_by_bar)} bars")

            # セクション情報がない場合、sections.jsonから取得
            if not any(targets_by_bar[i]["section_label"] for i in targets_by_bar):
                sections_json_path = midi_path.parent / "sections.json"
                if sections_json_path.exists():
                    with open(sections_json_path, "r", encoding="utf-8") as f:
                        sections_data = json.load(f)
                    section_labels = sections_data.get("section_labels", [])

                    for bar_idx in targets_by_bar:
                        if bar_idx < len(section_labels):
                            targets_by_bar[bar_idx]["section_label"] = section_labels[bar_idx]

                    if verbose:
                        print(f"   sections.json loaded: {len(section_labels)} labels")

        except Exception as e:
            if verbose:
                print(f"   ⚠️  failed to load bars.parquet: {e}")

    # εはテンポから自動推定（4% bar or 20ms）
    # Phase E: 拍子可変対応（bars.parquetからtime_signature参照）
    auto_eps = 0.0
    if tempo_bpm:
        # デフォルト4/4、bars.parquetにtime_signatureがあれば最初の小節を参照
        beats_per_bar_default = 4.0
        if bars_parquet and bars_parquet.exists() and PANDAS_AVAILABLE:
            try:
                bars_df_temp = pd.read_parquet(bars_parquet)
                if "time_signature" in bars_df_temp.columns and len(bars_df_temp) > 0:
                    ts = bars_df_temp.iloc[0].get("time_signature", "4/4")
                    numerator, _ = map(int, ts.split("/"))
                    beats_per_bar_default = float(numerator)
            except Exception:
                pass

        bar_dur = 60.0 / float(tempo_bpm) * beats_per_bar_default
        auto_eps = min(0.02, bar_dur * 0.04)
    eps = epsilon_sec if epsilon_sec is not None else auto_eps

    if verbose:
        print(f"   Epsilon: {eps*1000:.1f}ms")
        print(f"   Use downbeats: {use_downbeats}")

    # MIDI抽出
    bars_dict = extract_kpi_from_midi_enhanced(
        midi_path,
        tempo_bpm,
        hat_pitches=HAT,
        bars_from_downbeats=use_downbeats,
        epsilon_sec=eps,
        bars_parquet=bars_parquet,
        debug=verbose,
    )

    if verbose:
        print(f"   Total bars: {len(bars_dict)}")
        print(f"   Gate config: {gate_config_path}")

    # 検証
    results = {}
    pass_count = 0
    fail_count = 0
    warning_count = 0
    fail_reasons = []

    for bar_key in sorted(bars_dict.keys(), key=lambda x: int(x.split("_")[1])):
        bar_data = bars_dict[bar_key]
        bar_idx = bar_data["bar_index"]
        pattern = bar_data["pattern"]

        pass_flag, messages = validate_pattern_enhanced(
            pattern, gate_config, targets_by_bar=targets_by_bar, bar_idx=bar_idx
        )

        results[bar_key] = {
            "bar_index": bar_idx,
            "pattern_id": f"midi_bar_{bar_idx}",
            "kpi_pass": pass_flag,
            "messages": messages,
            "safe_kit_fallback_recommended": not pass_flag,
        }

        if pass_flag:
            pass_count += 1
        else:
            fail_count += 1
            for msg in messages:
                if "too low" in msg or "too high" in msg:
                    fail_reasons.append(msg.split(":")[0] + ":")

    # Fail原因集計
    fail_reason_counter = Counter(fail_reasons)
    fail_reason_top = fail_reason_counter.most_common(10)

    # サマリー
    total = len(bars_dict)
    pass_rate = (pass_count / total * 100) if total > 0 else 0.0

    summary = {
        "total_bars": total,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "warning_count": warning_count,
        "pass_rate": pass_rate,
        "fail_reason_top": fail_reason_top,
    }

    # 出力
    report = {"summary": summary, "results": results}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n📊 Validation Statistics:")
        print(f"   Total bars: {total}")
        print(f"   Pass: {pass_count} ({pass_rate:.1f}%)")
        print(f"   Fail: {fail_count} ({100-pass_rate:.1f}%)")
        print(f"   Warning: {warning_count}")
        print()
        print(f"🔍 Fail原因Top10:")
        for reason, count in fail_reason_top[:10]:
            pct = count / total * 100 if total > 0 else 0.0
            print(f"   {count:3d} ({pct:5.1f}%): {reason}")
        print()
        print(f"✅ Saved validation report: {output_path}")

        if fail_count > 0:
            print(f"\n⚠️  {fail_count} bars failed KPI Gate")
            print(f"   Recommend Safe-Kit fallback for failed bars")


def main():
    parser = argparse.ArgumentParser(description="KPI Gate validation (Enhanced)")
    parser.add_argument("--midi", type=Path, required=True, help="Path to drums.mid")
    parser.add_argument("--gate-config", type=Path, required=True, help="Path to gate_prod.yaml")
    parser.add_argument("--output", type=Path, required=True, help="Path to output JSON")
    parser.add_argument("--tempo-bpm", type=float, default=None, help="Tempo in BPM")
    parser.add_argument("--bars", type=Path, default=None, help="Path to bars.parquet")
    parser.add_argument("--downbeats", action="store_true", help="Use real downbeats")
    parser.add_argument("--epsilon-sec", type=float, default=None, help="Boundary epsilon (sec)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    kpi_gate_validate_enhanced(
        args.midi,
        args.gate_config,
        args.output,
        tempo_bpm=args.tempo_bpm,
        bars_parquet=args.bars,
        use_downbeats=args.downbeats,
        epsilon_sec=args.epsilon_sec,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
