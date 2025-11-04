#!/usr/bin/env python3
"""
KPI Gate検証スクリプト

drums_recommendations.jsonのパターンをKPI基準で検証:
1. gate_prod.yaml読み込み（品質しきい値）
2. 各小節のパターンをKPI検証
3. 失敗 → Safe-Kit fallback推奨
4. 検証結果JSON出力

MIDI実体検証:
    python3 scripts/kpi_gate.py \
        --midi song_packages/sample_project/sample_song/drums.mid \
        --gate-config configs/gate_prod.yaml \
        --output song_packages/sample_project/sample_song/kpi_gate_report_postgen.json

JSON検証（既存）:
    python3 scripts/kpi_gate.py \
        --recommendations song_packages/sample_project/sample_song/drums_recommendations.json \
        --gate-config configs/gate_prod.yaml \
        --output song_packages/sample_project/sample_song/kpi_gate_report.json
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def load_gate_config(yaml_path: Path) -> dict:
    """gate_prod.yaml読み込み"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_recommendations(json_path: Path) -> dict:
    """drums_recommendations.json読み込み"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tempo_at_time(change_times, tempi, t_sec: float, fallback: float = 120.0) -> float:
    """tempo change列から、時刻t_secで有効なテンポを返す"""
    if len(tempi) == 0:
        return fallback
    # 最新のchange_time <= t_sec のtempoを選ぶ
    idx = np.searchsorted(change_times, t_sec, side="right") - 1
    idx = int(np.clip(idx, 0, len(tempi) - 1))
    return float(tempi[idx])


def extract_kpi_from_midi(
    midi_path: Path, tempo_bpm: Optional[float] = None, midi_validation_cfg: Optional[dict] = None
) -> Dict[str, dict]:
    """MIDIファイルからKPI抽出

    Args:
        midi_path: MIDIファイルパス
        tempo_bpm: テンポ（Noneの場合はMIDIから抽出）
        midi_validation_cfg: gate_prod.yaml の drums.midi_validation 設定

    Returns:
        bars_dict: {bar_0: {pattern: {density: ..., kick_downbeat_rate: ...}}}
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi required for MIDI parsing. Install: pip install pretty_midi")

    midi = pretty_midi.PrettyMIDI(str(midi_path))

    # midi_validation 設定（デフォルト値）
    mv = midi_validation_cfg or {}
    downbeat_window_ms = float(mv.get("downbeat_window_ms", 45.0))
    backbeat_window_ms = float(mv.get("backbeat_window_ms", 45.0))
    ghost_velocity_min = int(mv.get("ghost_velocity_min", 30))
    hat_pedal_weight = float(mv.get("hat_pedal_weight", 0.7))
    ride_in_density = bool(mv.get("ride_in_density", False))
    swing_triplet_hint = bool(mv.get("swing_triplet_hint", True))

    change_times, tempi = midi.get_tempo_changes()
    if tempo_bpm is None:
        # 全体テンポは冒頭の有効テンポ（barごとにローカル再評価）
        tempo_bpm = _tempo_at_time(change_times, tempi, 0.0, 120.0)

    # ドラム楽器抽出
    drum_instruments = [inst for inst in midi.instruments if inst.is_drum]

    if not drum_instruments:
        raise ValueError(f"No drum track found in {midi_path}")

    # 全ドラムノート統合
    all_notes = []
    for inst in drum_instruments:
        all_notes.extend(inst.notes)

    # 時間でソート
    all_notes.sort(key=lambda n: n.start)

    # 小節分割（4/4拍子前提）
    bar_duration = 60.0 / tempo_bpm * 4
    total_duration = midi.get_end_time()
    num_bars = int(np.ceil(total_duration / bar_duration))

    def _within(t, center, win_ms):
        """時間tがcenter±win_ms以内か判定"""
        return abs((t - center) * 1000.0) <= win_ms

    bars_dict = {}

    for bar_idx in range(num_bars):
        bar_start = bar_idx * bar_duration
        bar_end = (bar_idx + 1) * bar_duration
        bar_mid = 0.5 * (bar_start + bar_end)
        bar_tempo = (
            tempo_bpm
            if tempo_bpm is not None
            else _tempo_at_time(change_times, tempi, bar_mid, tempo_bpm or 120.0)
        )
        sec_per_beat = 60.0 / bar_tempo
        beat_pos = [bar_start + i * sec_per_beat for i in range(4)]

        bar_notes = [n for n in all_notes if bar_start <= n.start < bar_end]

        if not bar_notes:
            continue

        # 1. density: ハイハット密度（ペダル重み対応）
        hat_notes = [n for n in bar_notes if n.pitch in (42, 44, 46)]
        density = 0.0
        for n in hat_notes:
            if n.pitch == 44:  # Pedal Hat
                density += hat_pedal_weight
            else:
                density += 1.0

        # 2. notes_per_bar: 総ノート数
        notes_per_bar = len(bar_notes)

        # 3. backbeat_strength: Snare平均Velocity（参考値）
        snare_notes = [n for n in bar_notes if n.pitch in (38, 40)]
        backbeat_strength = (
            (np.mean([n.velocity for n in snare_notes]) / 127.0) if snare_notes else 0.0
        )

        # 3b. snare_backbeat_acc: 2拍/4拍での命中率（ghost除外）
        backbeat_hits = 0
        backbeat_targets = 0
        for center in (beat_pos[1], beat_pos[3]):
            notes = [
                n
                for n in snare_notes
                if n.velocity >= ghost_velocity_min and _within(n.start, center, backbeat_window_ms)
            ]
            backbeat_targets += 1
            if len(notes) > 0:
                backbeat_hits += 1
        snare_backbeat_acc = (backbeat_hits / backbeat_targets) if backbeat_targets > 0 else 0.0

        # 3c. kick_downbeat_rate: 小節頭でのKick命中率
        kick_notes = [n for n in bar_notes if n.pitch in (35, 36)]
        kick_on_db = [n for n in kick_notes if _within(n.start, bar_start, downbeat_window_ms)]
        kick_downbeat_rate = 1.0 if kick_on_db else 0.0

        # 4. swing: マイクロタイミング分散（正規化）
        swing = 0.0
        if len(hat_notes) >= 4:
            eighth = sec_per_beat / 2.0
            triplet = sec_per_beat / 3.0
            grid = triplet if swing_triplet_hint else eighth
            swing_deviations = []

            for note in hat_notes:
                relative_time = note.start - bar_start
                nearest_grid = round(relative_time / grid) * grid
                deviation = abs(relative_time - nearest_grid)
                swing_deviations.append(deviation)

            swing = (np.mean(swing_deviations) / grid) if swing_deviations else 0.0

        # パターン辞書作成
        pattern = {
            "density": float(density),
            "notes_per_bar": float(notes_per_bar),
            "backbeat_strength": float(backbeat_strength),
            "swing": float(swing),
            "tempo_bpm": float(bar_tempo),
            "kick_downbeat_rate": float(kick_downbeat_rate),
            "snare_backbeat_acc": float(snare_backbeat_acc),
        }

        bars_dict[f"bar_{bar_idx}"] = {"bar_index": bar_idx, "pattern": pattern}

    return bars_dict


def validate_metric(value: float, metric_config: dict, metric_name: str) -> Tuple[bool, str]:
    """単一メトリック検証

    Args:
        value: 検証値
        metric_config: メトリック設定（min, max, warn_min, warn_max）
        metric_name: メトリック名

    Returns:
        (pass_flag, message): 合格フラグとメッセージ
    """
    min_val = metric_config.get("min", -float("inf"))
    max_val = metric_config.get("max", float("inf"))
    warn_min = metric_config.get("warn_min", min_val)
    warn_max = metric_config.get("warn_max", max_val)

    # ハード制約チェック
    if value < min_val:
        return False, f"{metric_name} too low: {value:.2f} < {min_val}"
    if value > max_val:
        return False, f"{metric_name} too high: {value:.2f} > {max_val}"

    # ソフト警告チェック
    if value < warn_min:
        return True, f"{metric_name} warning (low): {value:.2f} < {warn_min}"
    if value > warn_max:
        return True, f"{metric_name} warning (high): {value:.2f} > {warn_max}"

    return True, f"{metric_name} OK: {value:.2f}"


def validate_pattern(
    pattern: dict,
    gate_config: dict,
    section_label: Optional[str] = None,
    targets: Optional[dict] = None,
) -> Tuple[bool, List[str]]:
    """パターンKPI検証

    Args:
        pattern: パターン辞書（drums_recommendations.jsonのpattern）
        gate_config: gate_prod.yamlの設定
        section_label: セクションラベル（FILL/BREAK/INTRO/OUTRO等）
        targets: bars.parquetからのターゲット値（swing_target, density_target等）

    Returns:
        (pass_flag, messages): 合格フラグとメッセージリスト
    """
    drums_config = gate_config.get("drums", {})
    messages = []
    all_pass = True

    # 密度検証
    if "density" in pattern:
        pass_flag, msg = validate_metric(
            pattern["density"], drums_config.get("density", {}), "density"
        )
        # ★ 相対Warning: bars.parquet の density_target があれば「偏差」で警告
        if targets and "density_target" in targets:
            target = float(targets["density_target"])
            tol = float(drums_config.get("density", {}).get("warn_tol", 1.0))
            delta = abs(pattern["density"] - target)
            # ハードFail（min/max）は validate_metric に委譲。ここはWarningのみ上書き。
            if pass_flag and delta > tol:
                msg = f"density warning (dev): measured {pattern['density']:.2f} vs target {target:.2f} (Δ={delta:.2f} > {tol})"
        messages.append(msg)
        if not pass_flag:
            all_pass = False

    # スウィング検証
    if "swing" in pattern:
        pass_flag, msg = validate_metric(pattern["swing"], drums_config.get("swing", {}), "swing")
        # ★ 相対Warning: bars.parquet の swing_target があれば「偏差」で警告
        if targets and "swing_target" in targets:
            target = float(targets["swing_target"])
            tol = float(drums_config.get("swing", {}).get("warn_tol", 0.03))

            # ★ ストレート～軽いシャッフル（target≈0.0～0.15）の場合はtolを大幅に緩める
            # 実運用では±0.3程度の誤差は正常（人間の演奏バラつき、量子化誤差等）
            if target < 0.15:
                tol = 0.30

            delta = abs(pattern["swing"] - target)
            if pass_flag and delta > tol:
                msg = f"swing warning (dev): measured {pattern['swing']:.3f} vs target {target:.3f} (Δ={delta:.3f} > {tol})"
        messages.append(msg)
        if not pass_flag:
            all_pass = False

    # バックビート強度検証（セクション文脈対応）
    if "backbeat_strength" in pattern:
        pass_flag, msg = validate_metric(
            pattern["backbeat_strength"],
            drums_config.get("backbeat_strength", {}),
            "backbeat_strength",
        )

        # セクション例外: FILL/BREAK/INTRO/OUTRO は Fail→Warn に格下げ
        if (
            (not pass_flag)
            and section_label
            and str(section_label).upper() in {"FILL", "BREAK", "INTRO", "OUTRO"}
        ):
            # Fail を Warn として通す（観測は残す）
            pass_flag = True
            msg = msg.replace("too", "warning (section_override): too", 1)

        messages.append(msg)
        if not pass_flag:
            all_pass = False

    # 総ノート密度検証（過密防止）
    if "notes_per_bar" in pattern:
        pass_flag, msg = validate_metric(
            pattern["notes_per_bar"], drums_config.get("notes_per_bar", {}), "notes_per_bar"
        )
        messages.append(msg)
        if not pass_flag:
            all_pass = False

    # テンポ検証
    if "tempo_bpm" in pattern:
        pass_flag, msg = validate_metric(
            pattern["tempo_bpm"], drums_config.get("tempo_bpm", {}), "tempo_bpm"
        )
        messages.append(msg)
        if not pass_flag:
            all_pass = False

    # キック・ダウンビート命中率検証
    if "kick_downbeat_rate" in pattern and "kick_downbeat_rate" in drums_config:
        pass_flag, msg = validate_metric(
            pattern["kick_downbeat_rate"],
            drums_config.get("kick_downbeat_rate", {}),
            "kick_downbeat_rate",
        )
        messages.append(msg)
        if not pass_flag:
            all_pass = False

    # スネア・バックビート整合率検証
    if "snare_backbeat_acc" in pattern and "snare_backbeat_acc" in drums_config:
        pass_flag, msg = validate_metric(
            pattern["snare_backbeat_acc"],
            drums_config.get("snare_backbeat_acc", {}),
            "snare_backbeat_acc",
        )
        messages.append(msg)
        if not pass_flag:
            all_pass = False

    return all_pass, messages


def kpi_gate_validate(
    recommendations_path: Optional[Path] = None,
    midi_path: Optional[Path] = None,
    gate_config_path: Path = None,
    output_path: Path = None,
    tempo_bpm: Optional[float] = None,
    verbose: bool = True,
    bars_parquet_path: Optional[Path] = None,
):
    """KPI Gate検証メイン処理

    Args:
        recommendations_path: drums_recommendations.jsonパス（JSONモード）
        midi_path: drums.midパス（MIDIモード）
        gate_config_path: gate_prod.yamlパス
        output_path: kpi_gate_report.json出力パス
        tempo_bpm: テンポ（MIDIモード時）
        verbose: 詳細出力
        bars_parquet_path: bars.parquetパス（セクション情報取得用）
    """
    # 入力モード判定
    if midi_path:
        # MIDIモード
        if verbose:
            print(f"📖 Loading MIDI: {midi_path}")

        # gateのmidi_validationブロックを渡す
        tmp_gate = load_gate_config(gate_config_path)
        midi_val = (
            (tmp_gate.get("drums", {}) or {}).get("midi_validation", {})
            if isinstance(tmp_gate, dict)
            else {}
        )
        bars_dict = extract_kpi_from_midi(midi_path, tempo_bpm, midi_validation_cfg=midi_val)

        if verbose:
            print(f"   Total bars: {len(bars_dict)}")
            print(f"   Tempo: {tempo_bpm or 'auto-detected'} BPM")

    elif recommendations_path:
        # JSONモード（既存）
        if verbose:
            print(f"📖 Loading recommendations: {recommendations_path}")

        recommendations = load_recommendations(recommendations_path)

        # metadataを除外してbar_*のみカウント
        bars_dict = {k: v for k, v in recommendations.items() if k.startswith("bar_")}

        if verbose:
            print(f"   Total bars: {len(bars_dict)}")

    else:
        raise ValueError("Either --recommendations or --midi must be specified")

    # gate_config読み込み（上で読んだtmp_gateを再利用可）
    gate_config = tmp_gate if midi_path else load_gate_config(gate_config_path)

    if verbose:
        print(f"   Gate config: {gate_config_path}")

    # セクション情報読み込み（任意）
    section_by_bar = {}
    targets_by_bar = {}  # ★ ターゲット値マップ（swing_target, density_target等）

    if bars_parquet_path and Path(bars_parquet_path).exists() and PANDAS_AVAILABLE:
        try:
            df = pd.read_parquet(bars_parquet_path)
            if {"bar_index", "section_label"} <= set(df.columns):
                section_by_bar = dict(
                    zip(df["bar_index"].astype(int), df["section_label"].astype(str))
                )
                if verbose:
                    print(f"   Section info loaded: {len(section_by_bar)} bars")

            # ★ ターゲット値も辞書化（swing_target, density_target等）
            target_cols = [c for c in df.columns if c.endswith("_target")]
            if target_cols:
                for _, row in df.iterrows():
                    bar_idx = int(row["bar_index"])
                    targets_by_bar[bar_idx] = {
                        col: row[col] for col in target_cols if pd.notna(row[col])
                    }
                if verbose:
                    print(
                        f"   Target values loaded: {len(targets_by_bar)} bars, {len(target_cols)} metrics"
                    )
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Failed to load bars.parquet: {e}")
            section_by_bar = {}
            targets_by_bar = {}

    # 検証処理
    validation_results = {}
    pass_count = 0
    fail_count = 0
    warning_count = 0
    fail_reasons = Counter()  # Fail原因集計

    for bar_key, bar_data in bars_dict.items():
        bar_idx = bar_data["bar_index"]
        pattern = bar_data["pattern"]

        # セクション情報取得
        sec = section_by_bar.get(int(bar_idx))

        # ★ ターゲット値取得
        tgt = targets_by_bar.get(int(bar_idx), {})

        # KPI検証（セクション文脈 + ターゲット値を渡す）
        pass_flag, messages = validate_pattern(pattern, gate_config, section_label=sec, targets=tgt)

        # 統計
        if pass_flag:
            pass_count += 1
            # 警告チェック
            if any("warning" in msg for msg in messages):
                warning_count += 1
        else:
            fail_count += 1
            # Fail原因を集計（"density too high: 15.0 > 11.0" → "density too high"）
            for msg in messages:
                if "too low" in msg or "too high" in msg or "exceeds" in msg:
                    reason = " ".join(msg.split()[:3])  # 最初の3単語（KPI名 + 状態）
                    fail_reasons[reason] += 1

        # pattern_idの取得（MIDIモードでは存在しない）
        pattern_id = pattern.get("pattern_id", f"midi_bar_{bar_idx}")

        validation_results[bar_key] = {
            "bar_index": bar_idx,
            "pattern_id": pattern_id,
            "kpi_pass": pass_flag,
            "messages": messages,
            "safe_kit_fallback_recommended": not pass_flag,
        }

    # 統計
    if verbose:
        print(f"\n📊 Validation Statistics:")
        print(f"   Total bars: {len(bars_dict)}")
        print(f"   Pass: {pass_count} ({pass_count/len(bars_dict)*100:.1f}%)")
        print(f"   Fail: {fail_count} ({fail_count/len(bars_dict)*100:.1f}%)")
        print(f"   Warning: {warning_count} ({warning_count/len(bars_dict)*100:.1f}%)")

        # Fail原因Top10を表示
        if fail_reasons:
            print(f"\n🔍 Fail原因Top10:")
            for reason, count in fail_reasons.most_common(10):
                print(f"   {count:3d} ({count/fail_count*100:5.1f}%): {reason}")

    # レポート保存
    report = {
        "summary": {
            "total_bars": len(bars_dict),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "warning_count": warning_count,
            "pass_rate": pass_count / len(bars_dict) if len(bars_dict) > 0 else 0.0,
            "fail_reason_top": fail_reasons.most_common(12),  # JSON出力はTop12
        },
        "results": validation_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n✅ Saved validation report: {output_path}")

        if fail_count > 0:
            print(f"\n⚠️  {fail_count} bars failed KPI Gate")
            print(f"   Recommend Safe-Kit fallback for failed bars")


def main():
    parser = argparse.ArgumentParser(description="Validate drums patterns with KPI Gate")

    # 入力ソース（排他的）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--recommendations", type=Path, help="Path to drums_recommendations.json (JSON mode)"
    )
    input_group.add_argument("--midi", type=Path, help="Path to drums.mid (MIDI mode)")

    # 共通引数
    parser.add_argument("--gate-config", type=Path, required=True, help="Path to gate_prod.yaml")
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to output kpi_gate_report.json"
    )
    parser.add_argument(
        "--tempo-bpm",
        type=float,
        default=None,
        help="Tempo in BPM (MIDI mode only, auto-detect if not specified)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument(
        "--bars-parquet",
        type=Path,
        default=None,
        help="Optional bars.parquet to enable section-aware overrides (FILL/BREAK/INTRO/OUTRO)",
    )

    args = parser.parse_args()

    kpi_gate_validate(
        recommendations_path=args.recommendations,
        midi_path=args.midi,
        gate_config_path=args.gate_config,
        output_path=args.output,
        tempo_bpm=args.tempo_bpm,
        verbose=not args.quiet,
        bars_parquet_path=args.bars_parquet,
    )


if __name__ == "__main__":
    main()
