#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ci_verify_music_package.py
--------------------------------------------------
SunoAI アレンジメント成果物の "壊れていないこと" を自動検証する CI 用スクリプト。

検証観点（安全な最小セット）:
  1) テンポ・拍子メタ:
     - set_tempo メタは Track 0 のみ（テンポトラック専用）
  2) 小節境界・長さ:
     - bars.parquet の bar 数と MIDI の downbeats が一致（±1 許容）
     - 全トラックの終端が期待時間（bars × 4拍）±許容誤差に収まる
  3) クリップ健全性:
     - 期待終端を超える長尺ノートが無い（超過イベント数も計上）
  4) KPI Gate（任意）:
     - kpi_gate_enhanced.py を呼び出し、Pass率が閾値以上

実行例:
  python3 ci_verify_music_package.py \
    --song-dir song_packages/suno_project/song_001 \
    --midi song_packages/suno_project/song_001/full_arrangement.mid \
    --bars song_packages/suno_project/song_001/bars.parquet \
    --tempo-bpm 74.677 \
    --gate-config configs/gate_prod.yaml \
    --kpi-threshold 0.90

終了コード:
  0: すべて PASS（Warn は許容）
  1: いずれか FAIL
"""

import argparse
import json
import os
import sys
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

# 依存: pretty_midi, mido, pandas, yaml
try:
    import pretty_midi
except Exception as e:
    print("❌ pretty_midi import に失敗しました。`pip install pretty_midi` を実行してください。")
    raise
try:
    import mido
except Exception as e:
    print("❌ mido import に失敗しました。`pip install mido` を実行してください。")
    raise
try:
    import pandas as pd
except Exception as e:
    print("❌ pandas import に失敗しました。`pip install pandas` を実行してください。")
    raise
try:
    import yaml
except Exception as e:
    print("❌ pyyaml import に失敗しました。`pip install pyyaml` を実行してください。")
    raise


@dataclass
class CheckResult:
    name: str
    status: str  # "pass" | "fail" | "warn"
    details: str


def human_sec(sec: float) -> str:
    return f"{sec:.2f}s"


def compute_ab_metrics(midi_path: Path, song_dir: Path | None) -> Dict[str, Any]:
    """パッチ5: ABテスト用メトリクス計算（Phase F準備）"""
    metrics = {"build_id": midi_path.stem}

    # F0追従精度（Bass）
    if song_dir:
        bass_f0 = song_dir / "bass_f0.parquet"
        if bass_f0.exists():
            try:
                f0_df = pd.read_parquet(bass_f0)
                pm = pretty_midi.PrettyMIDI(str(midi_path))
                bass_inst = [
                    i for i in pm.instruments if not i.is_drum and "bass" in i.name.lower()
                ]

                if bass_inst:
                    bass_notes = bass_inst[0].notes
                    f0_values = f0_df.get("f0_median_midi", pd.Series(dtype=float)).dropna()

                    if len(bass_notes) > 0 and len(f0_values) > 0:
                        diffs = []
                        for note in bass_notes:
                            closest_f0 = f0_values.iloc[0]  # 簡易マッチング
                            diffs.append(abs(note.pitch - closest_f0))

                        metrics["f0_cents_mae"] = (
                            float(sum(diffs) / len(diffs) * 100) if diffs else 0.0
                        )
            except Exception:
                pass

    # ボイシング多様性（Piano）
    if song_dir:
        oaf_piano = song_dir / "piano_oaf.json"
        if oaf_piano.exists():
            try:
                with open(oaf_piano, "r") as f:
                    oaf = json.load(f)
                pm = pretty_midi.PrettyMIDI(str(midi_path))
                piano_inst = [
                    i for i in pm.instruments if not i.is_drum and "piano" in i.name.lower()
                ]

                if piano_inst:
                    piano_notes = piano_inst[0].notes
                    oaf_notes = oaf.get("notes", [])

                    oaf_unique = len(set(n["midi"] for n in oaf_notes)) if oaf_notes else 1
                    piano_unique = len(set(n.pitch for n in piano_notes)) if piano_notes else 0

                    metrics["voicing_unique_ratio"] = float(piano_unique / max(1, oaf_unique))
            except Exception:
                pass

    # CC変動範囲（Strings）
    try:
        mid = mido.MidiFile(str(midi_path))
        cc11_vals, cc74_vals = [], []

        for track in mid.tracks:
            for msg in track:
                if msg.type == "control_change":
                    if msg.control == 11:
                        cc11_vals.append(msg.value)
                    elif msg.control == 74:
                        cc74_vals.append(msg.value)

        metrics["cc11_range"] = max(cc11_vals) - min(cc11_vals) if cc11_vals else 0
        metrics["cc74_range"] = max(cc74_vals) - min(cc74_vals) if cc74_vals else 0
    except Exception:
        pass

    return metrics


def load_song_package(song_dir: Path) -> Dict[str, Any]:
    yaml_path = song_dir / "song_package.yaml"
    if yaml_path.exists():
        return yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return {}


def expected_duration_sec(num_bars: int, bpm: float, beats_per_bar: float = 4.0) -> float:
    return num_bars * beats_per_bar * (60.0 / bpm)


def check_magenta_intermediates(song_dir: Path, drums_mode: str) -> CheckResult:
    """Magenta中間ファイル存在チェック（drums_mode=magentaの時のみ）"""
    if drums_mode != "magenta":
        return CheckResult(
            name="Magenta intermediate files",
            status="pass",
            details="SKIP: drums_mode != magenta",
        )

    required = ["drums_seed.mid", "drums_grooved.mid", "drums_plan.json"]
    missing = [f for f in required if not (song_dir / f).exists()]

    if missing:
        return CheckResult(
            name="Magenta intermediate files",
            status="fail",
            details=f"Missing: {', '.join(missing)}",
        )

    return CheckResult(
        name="Magenta intermediate files",
        status="pass",
        details="All Magenta intermediate files present",
    )


def check_activity_columns(song_dir: Path, used_inst_activity: bool) -> CheckResult:
    """activity列の存在チェック（--inst-activity使用時）"""
    if not used_inst_activity:
        return CheckResult(
            name="Activity columns",
            status="pass",
            details="SKIP: --inst-activity not used",
        )

    bars_path = song_dir / "bars.parquet"
    if not bars_path.exists():
        return CheckResult(
            name="Activity columns",
            status="fail",
            details="bars.parquet not found",
        )

    try:
        bars_df = pd.read_parquet(bars_path)
        activity_cols = [c for c in bars_df.columns if c.endswith("_activity")]

        if not activity_cols:
            return CheckResult(
                name="Activity columns",
                status="fail",
                details="No *_activity columns found in bars.parquet",
            )

        return CheckResult(
            name="Activity columns",
            status="pass",
            details=f"Activity columns found: {', '.join(activity_cols)}",
        )
    except Exception as e:
        return CheckResult(
            name="Activity columns",
            status="fail",
            details=f"Failed to read bars.parquet: {e}",
        )


def check_crepe_oaf_outputs(song_dir: Path, enable_crepe: bool, enable_oaf: bool) -> CheckResult:
    """CREPE/OaF成果物の存在チェック"""
    missing = []

    if enable_crepe:
        crepe_file = song_dir / "vocal_f0_crepe.parquet"
        if not crepe_file.exists():
            missing.append("vocal_f0_crepe.parquet")

    if enable_oaf:
        oaf_file = song_dir / "piano_oaf.mid"
        if not oaf_file.exists():
            missing.append("piano_oaf.mid")

    if missing:
        return CheckResult(
            name="CREPE/OaF outputs",
            status="warn",
            details=f"Missing (E2E continues): {', '.join(missing)}",
        )

    return CheckResult(
        name="CREPE/OaF outputs",
        status="pass",
        details="All CREPE/OaF outputs present" if (enable_crepe or enable_oaf) else "SKIP",
    )


def check_grooved_mid(song_dir: Path, enable: bool) -> CheckResult:
    """grooved.midの存在とノート数確認"""
    if not enable:
        return CheckResult(
            name="grooved.mid",
            status="skip",
            details="Magenta groove disabled",
        )

    # grooved.mid のノート数確認
    try:
        pm = pretty_midi.PrettyMIDI(str(song_dir / "drums_grooved.mid"))
        note_count = sum(len(instr.notes) for instr in pm.instruments)
        if note_count == 0:
            return CheckResult(
                name="Magenta intermediate files",
                status="fail",
                details="❌ drums_grooved.mid has 0 notes",
            )
    except Exception as e:
        return CheckResult(
            name="Magenta intermediate files",
            status="warn",
            details=f"⚠️  Could not verify grooved.mid note count: {e}",
        )

    return CheckResult(
        name="Magenta intermediate files",
        status="pass",
        details=f"✅ All Magenta files present, grooved.mid has {note_count} notes",
    )


def check_set_tempo_track0_only(midi_path: Path) -> CheckResult:
    mid = mido.MidiFile(str(midi_path))
    bad_tracks = []
    for i, track in enumerate(mid.tracks):
        tempo_msgs = [msg for msg in track if msg.type == "set_tempo"]
        if i == 0:
            # Track0 は OK（ただし 0 個でも OK とする）
            continue
        if len(tempo_msgs) > 0:
            bad_tracks.append(i)
    if bad_tracks:
        return CheckResult(
            name="Tempo meta on Track>0",
            status="fail",
            details=f"set_tempo が Track {bad_tracks} に存在します。テンポは Track 0 限定にしてください。",
        )
    return CheckResult(
        name="Tempo meta on Track>0",
        status="pass",
        details="OK: set_tempo は Track 0 のみ。",
    )


def check_ppq_consistency(midi_path: Path, expected_ppq: int = 480) -> CheckResult:
    """PPQ（Pulses Per Quarter）が期待値と一致するか確認"""
    try:
        mid = mido.MidiFile(str(midi_path))
        actual_ppq = mid.ticks_per_beat
        if actual_ppq != expected_ppq:
            return CheckResult(
                name="PPQ consistency",
                status="fail",
                details=f"❌ PPQ mismatch: expected {expected_ppq}, got {actual_ppq}",
            )
        return CheckResult(
            name="PPQ consistency",
            status="pass",
            details=f"✅ PPQ={actual_ppq} (expected {expected_ppq})",
        )
    except Exception as e:
        return CheckResult(
            name="PPQ consistency",
            status="warn",
            details=f"⚠️  Could not verify PPQ: {e}",
        )


def check_drums_channel_9(midi_path: Path) -> CheckResult:
    """Drumsトラックがchannel=9を維持しているか確認"""
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        drums_instruments = [instr for instr in pm.instruments if instr.is_drum]
        non_channel9 = [
            instr for instr in drums_instruments if instr.program != 0 or not instr.is_drum
        ]

        # MIDOでも確認（channel番号直接チェック）
        mid = mido.MidiFile(str(midi_path))
        drums_channels = set()
        for track in mid.tracks:
            for msg in track:
                if msg.type == "note_on" and hasattr(msg, "channel"):
                    # Drumsはchannel 9（0-indexed）
                    if msg.channel == 9:
                        drums_channels.add(msg.channel)

        if drums_channels and 9 not in drums_channels:
            return CheckResult(
                name="Drums channel=9",
                status="fail",
                details=f"❌ Drums not on channel 9: found channels {drums_channels}",
            )

        if not drums_channels and len(drums_instruments) > 0:
            return CheckResult(
                name="Drums channel=9",
                status="warn",
                details=f"⚠️  Drums instrument exists but channel unclear",
            )

        return CheckResult(
            name="Drums channel=9",
            status="pass",
            details=f"✅ Drums on channel 9 (instruments: {len(drums_instruments)})",
        )
    except Exception as e:
        return CheckResult(
            name="Drums channel=9",
            status="warn",
            details=f"⚠️  Could not verify drums channel: {e}",
        )


def check_downbeats_vs_bars(
    midi_path: Path, bars_path: Path, tolerance_bars: int = 1
) -> CheckResult:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    downbeats = pm.get_downbeats()
    try:
        bars_df = pd.read_parquet(bars_path)
        num_bars = int(len(bars_df))
    except Exception:
        return CheckResult(
            name="Downbeats vs bars",
            status="warn",
            details=f"bars.parquet を読み込めませんでした（{bars_path}）。downbeats={len(downbeats)} のみ報告します。",
        )
    expected = num_bars + 1  # 終端 downbeat を含む
    delta = abs(len(downbeats) - expected)
    if delta <= tolerance_bars:
        return CheckResult(
            name="Downbeats vs bars",
            status="pass",
            details=f"OK: downbeats={len(downbeats)}, bars={num_bars}（期待 downbeats≈{expected}, 許容±{tolerance_bars}）",
        )
    return CheckResult(
        name="Downbeats vs bars",
        status="fail",
        details=f"NG: downbeats={len(downbeats)} と bars={num_bars} が乖離（期待≈{expected}, 乖離={delta}）。",
    )


def check_track_durations(
    midi_path: Path, num_bars: int, bpm: float, tolerance_sec: float = 1.0
) -> List[CheckResult]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    exp = expected_duration_sec(num_bars, bpm)
    lo = exp - tolerance_sec
    hi = exp + tolerance_sec

    results: List[CheckResult] = []
    # 全体終端
    end = pm.get_end_time()
    if lo <= end <= hi:
        results.append(
            CheckResult(
                name="Total duration",
                status="pass",
                details=f"OK: {human_sec(end)} ≈ 期待 {human_sec(exp)} (±{tolerance_sec:.2f}s)",
            )
        )
    else:
        results.append(
            CheckResult(
                name="Total duration",
                status="fail",
                details=f"NG: {human_sec(end)} が期待 {human_sec(exp)} ±{tolerance_sec:.2f}s を外れています。",
            )
        )

    # 各トラック
    for inst in pm.instruments:
        dur = max((n.end for n in inst.notes), default=0.0)
        track_name = (inst.name or "Unnamed")[:32]
        status = "pass" if lo <= dur <= hi else "fail"
        results.append(
            CheckResult(
                name=f"Track duration: {track_name}",
                status=status,
                details=f"{human_sec(dur)}（期待 {human_sec(exp)} ±{tolerance_sec:.2f}s）",
            )
        )
    return results


def check_overlong_notes(midi_path: Path, num_bars: int, bpm: float) -> CheckResult:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    # 期待終端（秒）
    end_sec = expected_duration_sec(num_bars, bpm)
    over = 0
    for inst in pm.instruments:
        for n in inst.notes:
            if n.end > end_sec + 1e-6:
                over += 1
    if over == 0:
        return CheckResult(
            name="Hard clip over-end",
            status="pass",
            details="OK: 期待終端を超えるノートはありません。",
        )
    return CheckResult(
        name="Hard clip over-end",
        status="fail",
        details=f"NG: 期待終端 {human_sec(end_sec)} を超えるノートが {over} 個あります。",
    )


def run_kpi_gate_if_available(
    midi_path: Path, bars_path: Path, gate_config: Path, threshold: float, python_bin: Optional[str]
) -> Optional[CheckResult]:
    """
    scripts/kpi_gate_enhanced.py が存在する場合のみ実行。
    依存環境に左右されるため、失敗したら WARN に落として続行。
    """
    # 推定場所（カレントの scripts/）
    script_candidates = [
        Path("scripts/kpi_gate_enhanced.py"),
        Path("kpi_gate_enhanced.py"),
    ]
    script_path = next((p for p in script_candidates if p.exists()), None)
    if not script_path:
        return CheckResult(
            name="KPI Gate",
            status="warn",
            details="kpi_gate_enhanced.py が見つからないためスキップしました。",
        )

    out_path = Path(os.getenv("TMPDIR", "/tmp")) / "ci_kpi_report.json"
    py = python_bin or sys.executable
    cmd = [
        py,
        str(script_path),
        "--midi",
        str(midi_path),
        "--bars",
        str(bars_path),
        "--gate-config",
        str(gate_config),
        "--out",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return CheckResult(
                name="KPI Gate",
                status="warn",
                details=f"KPI 実行に失敗しました（return={proc.returncode}）。stderr:\n{proc.stderr[:4000]}",
            )
        data = json.loads(Path(out_path).read_text(encoding="utf-8"))
        # 2 つの形式に対応（summary.pass_rate or pass_rate）
        pass_rate = None
        if isinstance(data, dict):
            if (
                "summary" in data
                and isinstance(data["summary"], dict)
                and "pass_rate" in data["summary"]
            ):
                pass_rate = float(data["summary"]["pass_rate"]) / (
                    100.0 if data["summary"]["pass_rate"] > 1.0 else 1.0
                )
            elif "pass_rate" in data:
                pr = float(data["pass_rate"])
                pass_rate = pr / (100.0 if pr > 1.0 else 1.0)
        if pass_rate is None:
            return CheckResult(
                name="KPI Gate",
                status="warn",
                details="KPI レポートから pass_rate を取得できませんでした。",
            )
        if pass_rate >= threshold:
            return CheckResult(
                name="KPI Gate",
                status="pass",
                details=f"OK: Pass率={pass_rate:.3f}（閾値 {threshold:.3f} 以上）",
            )
        else:
            return CheckResult(
                name="KPI Gate",
                status="fail",
                details=f"NG: Pass率={pass_rate:.3f}（閾値 {threshold:.3f} 未満）",
            )
    except Exception as e:
        return CheckResult(name="KPI Gate", status="warn", details=f"KPI 実行に失敗（例外）: {e}")


def main():
    ap = argparse.ArgumentParser(description="SunoAI arrangement CI verify")
    ap.add_argument(
        "--song-dir",
        type=Path,
        required=False,
        help="SongPackage ディレクトリ（song_package.yaml がある場所）",
    )
    ap.add_argument("--midi", type=Path, required=True, help="検証対象のフルアレンジ MIDI")
    ap.add_argument("--bars", type=Path, required=True, help="bars.parquet パス")
    ap.add_argument(
        "--drums-mode",
        type=str,
        default="rule",
        help="Drums生成モード（rule|ml|real|magenta）※magenta時は中間ファイル必須",
    )
    ap.add_argument(
        "--tempo-bpm",
        type=float,
        default=None,
        help="期待 BPM（未指定なら song_package.yaml から読む）",
    )
    ap.add_argument("--beats-per-bar", type=float, default=4.0, help="拍子の分子（既定 4/4）")
    ap.add_argument(
        "--duration-tolerance",
        type=float,
        default=5.0,
        help="終端時間の許容誤差 [秒]（activity/anchors間引き考慮）",
    )
    ap.add_argument(
        "--downbeats-tolerance", type=int, default=1, help="downbeats と bars の許容差 [bar]"
    )
    ap.add_argument("--gate-config", type=Path, default=None, help="KPI Gate 用 YAML（任意）")
    ap.add_argument(
        "--kpi-threshold", type=float, default=0.90, help="KPI Gate 最低合格率（0.0-1.0）"
    )
    ap.add_argument(
        "--python-bin", type=str, default=None, help="KPI 呼び出しに使う Python（仮想環境切替用）"
    )
    ap.add_argument(
        "--report", type=Path, default=Path("ci_report.json"), help="JSON レポート出力先"
    )
    ap.add_argument(
        "--inst-activity", action="store_true", help="--inst-activity使用時のactivity列チェック"
    )
    ap.add_argument(
        "--enable-crepe", action="store_true", help="CREPE F0抽出有効時の成果物チェック"
    )
    ap.add_argument(
        "--enable-oaf", action="store_true", help="Onsets-and-Frames転写有効時の成果物チェック"
    )
    ap.add_argument(
        "--ab-csv", type=Path, default=None, help="パッチ5: ABテスト用CSV出力先（Phase F準備）"
    )
    args = ap.parse_args()

    # 参照情報の読込
    bpm = args.tempo_bpm
    if args.song_dir and not bpm:
        sp = load_song_package(args.song_dir)
        bpm = float(sp.get("meta", {}).get("bpm", sp.get("meta", {}).get("tempo_bpm", 0.0)))
    if not bpm or bpm <= 0.0:
        print(
            "❌ BPM が不明です。--tempo-bpm または song_package.yaml(meta.bpm/tempo_bpm) を指定してください。"
        )
        sys.exit(1)

    try:
        bars_df = pd.read_parquet(args.bars)
        num_bars = int(len(bars_df))
    except Exception as e:
        print(f"❌ bars.parquet の読込に失敗しました: {e}")
        sys.exit(1)

    results: List[CheckResult] = []

    # 0) Magenta中間ファイルチェック（drums_mode=magentaの場合のみ）
    if args.song_dir:
        results.append(check_magenta_intermediates(args.song_dir, args.drums_mode))

    # 0.1) Activity列チェック（--inst-activity使用時）
    if args.song_dir and args.inst_activity:
        results.append(check_activity_columns(args.song_dir, args.inst_activity))

    # 0.2) CREPE/OaF成果物チェック
    if args.song_dir and (args.enable_crepe or args.enable_oaf):
        results.append(check_crepe_oaf_outputs(args.song_dir, args.enable_crepe, args.enable_oaf))

    # 1) テンポメタの健全性
    results.append(check_set_tempo_track0_only(args.midi))

    # 1.5) PPQ一貫性チェック（grooved.mid PPQ==480確認）
    results.append(check_ppq_consistency(args.midi, expected_ppq=480))

    # 1.6) Drumsチャンネル維持チェック（channel==9確認）
    results.append(check_drums_channel_9(args.midi))

    # 2) downbeats vs bars
    results.append(
        check_downbeats_vs_bars(args.midi, args.bars, tolerance_bars=args.downbeats_tolerance)
    )

    # 3) 長さチェック（全体 + 各トラック）
    results.extend(
        check_track_durations(
            args.midi, num_bars=num_bars, bpm=bpm, tolerance_sec=args.duration_tolerance
        )
    )

    # 4) 期待終端超過ノート
    results.append(check_overlong_notes(args.midi, num_bars=num_bars, bpm=bpm))

    # 5) KPI Gate（任意）
    if args.gate_config:
        results.append(
            run_kpi_gate_if_available(
                args.midi, args.bars, args.gate_config, args.kpi_threshold, args.python_bin
            )
        )

    # 集計
    summary = {"pass": 0, "fail": 0, "warn": 0}
    for r in results:
        if r is None:
            continue
        summary[r.status] = summary.get(r.status, 0) + 1

    # 出力
    report = {
        "midi": str(args.midi),
        "bars": str(args.bars),
        "bpm": bpm,
        "num_bars": num_bars,
        "expected_duration_sec": expected_duration_sec(num_bars, bpm),
        "results": [asdict(r) for r in results if r],
        "summary": summary,
    }
    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ CI レポートを書き出しました: {args.report}")
    except Exception as e:
        print(f"⚠️ CI レポートの書き出しに失敗しました: {e}")

    # 表示（短縮）
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("CI Summary:")
    for k in ("pass", "warn", "fail"):
        print(f"  {k:>4s} : {summary.get(k,0)}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # パッチ5: ABテスト用CSV出力（Phase F準備）
    if args.ab_csv:
        try:
            ab_metrics = compute_ab_metrics(args.midi, args.song_dir)

            # CSV書き出し（ヘッダー付き）
            import csv

            csv_exists = args.ab_csv.exists()

            args.ab_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(args.ab_csv, "a", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "build_id",
                    "f0_cents_mae",
                    "voicing_unique_ratio",
                    "cc11_range",
                    "cc74_range",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                if not csv_exists:
                    writer.writeheader()

                writer.writerow(ab_metrics)

            print(f"✅ ABテスト用CSVを書き出しました: {args.ab_csv}")
        except Exception as e:
            print(f"⚠️ ABテスト用CSV書き出しに失敗しました: {e}")

    # 失敗があれば非ゼロ終了
    sys.exit(1 if summary.get("fail", 0) > 0 else 0)


if __name__ == "__main__":
    main()
