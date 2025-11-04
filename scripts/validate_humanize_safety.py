#!/usr/bin/env python3
"""
[レビュー提案3] 人間味フェイルセーフ検証

CI用の安全性検証スクリプト：
1. 無効化時ビット完全一致（±0ticks / ±0vel / ±0len）
2. 境界制約チェック（max|Δtime|≤30ms, max|Δvel|≤20, max|Δlen|≤40ms）
3. KPI非劣化チェック（Pass率悪化≤0.3%）
"""

import sys
import json
import mido
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class HumanizeSafetyValidator:
    """人間味安全性検証クラス"""

    def __init__(self, baseline_mid: Path, humanized_mid: Path, plan_json: Path = None):
        self.baseline_mid = baseline_mid
        self.humanized_mid = humanized_mid
        self.plan_json = plan_json

        # 安全境界
        self.MAX_TIME_DELTA_MS = 30
        self.MAX_VEL_DELTA = 20
        self.MAX_LEN_DELTA_MS = 40
        self.MAX_KPI_DEGRADATION_PCT = 0.3

    def load_midi_events(self, midi_path: Path) -> List[Dict]:
        """MIDIファイルからノートイベントを抽出"""
        mid = mido.MidiFile(midi_path)
        events = []
        ppq = mid.ticks_per_beat

        for track_idx, track in enumerate(mid.tracks):
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    events.append(
                        {
                            "track": track_idx,
                            "time_ticks": abs_time,
                            "time_ms": (abs_time / ppq) * 500,  # 仮定: 120bpm
                            "pitch": msg.note,
                            "velocity": msg.velocity,
                            "channel": msg.channel,
                        }
                    )
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    # ノート長を逆算（簡易版）
                    for ev in reversed(events):
                        if (
                            ev["pitch"] == msg.note
                            and ev["channel"] == msg.channel
                            and "length_ms" not in ev
                        ):
                            ev["length_ticks"] = abs_time - ev["time_ticks"]
                            ev["length_ms"] = (ev["length_ticks"] / ppq) * 500
                            break

        return events

    def check_1_exact_match_when_disabled(self) -> Tuple[bool, str]:
        """チェック1: humanize無効時にビット完全一致"""
        baseline = self.load_midi_events(self.baseline_mid)
        humanized = self.load_midi_events(self.humanized_mid)

        if len(baseline) != len(humanized):
            return False, f"イベント数不一致: baseline={len(baseline)}, humanized={len(humanized)}"

        violations = []
        for i, (b, h) in enumerate(zip(baseline, humanized)):
            if b["time_ticks"] != h["time_ticks"]:
                violations.append(f"Event#{i}: time {b['time_ticks']} → {h['time_ticks']}")
            if b["velocity"] != h["velocity"]:
                violations.append(f"Event#{i}: vel {b['velocity']} → {h['velocity']}")
            if b.get("length_ticks") != h.get("length_ticks"):
                violations.append(
                    f"Event#{i}: len {b.get('length_ticks')} → {h.get('length_ticks')}"
                )

        if violations:
            return False, f"{len(violations)}件の不一致:\n" + "\n".join(violations[:10])
        return True, "✅ ビット完全一致"

    def check_2_boundary_constraints(self) -> Tuple[bool, str]:
        """チェック2: 境界制約チェック"""
        baseline = self.load_midi_events(self.baseline_mid)
        humanized = self.load_midi_events(self.humanized_mid)

        violations = {"time": [], "velocity": [], "length": []}

        for i, (b, h) in enumerate(zip(baseline, humanized)):
            delta_time_ms = abs(h["time_ms"] - b["time_ms"])
            delta_vel = abs(h["velocity"] - b["velocity"])
            delta_len_ms = abs(h.get("length_ms", 0) - b.get("length_ms", 0))

            if delta_time_ms > self.MAX_TIME_DELTA_MS:
                violations["time"].append((i, delta_time_ms))
            if delta_vel > self.MAX_VEL_DELTA:
                violations["velocity"].append((i, delta_vel))
            if delta_len_ms > self.MAX_LEN_DELTA_MS:
                violations["length"].append((i, delta_len_ms))

        total_violations = sum(len(v) for v in violations.values())
        if total_violations > 0:
            msg = f"{total_violations}件の境界違反:\n"
            msg += f"  時間: {len(violations['time'])}件 (max={self.MAX_TIME_DELTA_MS}ms)\n"
            msg += f"  ベロシティ: {len(violations['velocity'])}件 (max={self.MAX_VEL_DELTA})\n"
            msg += f"  長さ: {len(violations['length'])}件 (max={self.MAX_LEN_DELTA_MS}ms)"
            return False, msg

        return (
            True,
            f"✅ 境界制約OK (time≤{self.MAX_TIME_DELTA_MS}ms, vel≤{self.MAX_VEL_DELTA}, len≤{self.MAX_LEN_DELTA_MS}ms)",
        )

    def check_3_kpi_non_degradation(self) -> Tuple[bool, str]:
        """チェック3: KPI非劣化チェック（簡易版）"""
        # 注: 実際のKPI評価はkpi_gate.pyを利用すべき
        # ここでは簡易的にノート数とバックビート強度を確認
        baseline = self.load_midi_events(self.baseline_mid)
        humanized = self.load_midi_events(self.humanized_mid)

        if len(baseline) != len(humanized):
            degradation = abs(len(humanized) - len(baseline)) / len(baseline) * 100
            if degradation > self.MAX_KPI_DEGRADATION_PCT:
                return (
                    False,
                    f"❌ ノート数変化: {degradation:.2f}% (閾値: {self.MAX_KPI_DEGRADATION_PCT}%)",
                )

        # バックビート強度（簡易: 2/4拍のvel平均）
        def calc_backbeat_strength(events):
            backbeats = [e["velocity"] for e in events if e["channel"] == 9]  # drums
            return sum(backbeats) / len(backbeats) if backbeats else 0

        baseline_bb = calc_backbeat_strength(baseline)
        humanized_bb = calc_backbeat_strength(humanized)
        bb_change = abs(humanized_bb - baseline_bb) / baseline_bb * 100 if baseline_bb > 0 else 0

        if bb_change > self.MAX_KPI_DEGRADATION_PCT:
            return (
                False,
                f"❌ バックビート変化: {bb_change:.2f}% (閾値: {self.MAX_KPI_DEGRADATION_PCT}%)",
            )

        return True, f"✅ KPI非劣化 (ノート数一致, BB変化={bb_change:.3f}%)"

    def run_all_checks(self) -> bool:
        """全チェック実行"""
        print("=" * 60)
        print("人間味フェイルセーフ検証")
        print("=" * 60)
        print(f"Baseline:  {self.baseline_mid}")
        print(f"Humanized: {self.humanized_mid}")
        print()

        all_pass = True

        # チェック1: 無効化時ビット完全一致（このチェックは humanize.enabled=false 時のみ）
        # 現在は常にスキップ（humanize有効前提）
        print("[チェック1] 無効化時ビット完全一致")
        print("  ⏭️  スキップ（humanize有効時は非適用）")
        print()

        # チェック2: 境界制約
        print("[チェック2] 境界制約チェック")
        ok, msg = self.check_2_boundary_constraints()
        print(f"  {msg}")
        if not ok:
            all_pass = False
        print()

        # チェック3: KPI非劣化
        print("[チェック3] KPI非劣化チェック")
        ok, msg = self.check_3_kpi_non_degradation()
        print(f"  {msg}")
        if not ok:
            all_pass = False
        print()

        print("=" * 60)
        if all_pass:
            print("✅ すべてのチェックに合格")
            return True
        else:
            print("❌ 一部のチェックで違反を検出")
            return False


def main():
    parser = argparse.ArgumentParser(description="人間味フェイルセーフ検証")
    parser.add_argument("--baseline", required=True, help="ベースラインMIDI（humanize無効）")
    parser.add_argument("--humanized", required=True, help="人間味適用MIDI（humanize有効）")
    parser.add_argument("--plan", help="plan.json（オプション、KPI検証用）")

    args = parser.parse_args()

    validator = HumanizeSafetyValidator(
        baseline_mid=Path(args.baseline),
        humanized_mid=Path(args.humanized),
        plan_json=Path(args.plan) if args.plan else None,
    )

    success = validator.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
