#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harmony_pipeline_phaseB.py - Harmony Beat専用パイプライン（chordmap完全排除版）

Architecture Philosophy:
- Source of Truth: harmony_beat.json ONLY
- No chordmap/manual/LAMDA dependencies
- Sections/Emotion: Derived from harmony_beat.json (minimal cache)
- Policy: scripts/instrument_harmonybeat/policy (chordmap policy完全廃止)
- Bars: bars_with_slots.parquet (スロット情報必須)
- Two-Layer System:
  - Expression Layer: Generators (note placement)
  - Performance Layer: Playfulness/Dynamics (humanization)

Created: 2025-12-06
Purpose: "全5楽器が同じ呼吸（ダイナミクス・感情）で演奏する"新アーキテクチャ
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    """JSON読み込み"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    """JSON保存"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_sections_from_harmony_beat(harmony_beat: Dict[str, Any]) -> Dict[str, Any]:
    """
    harmony_beat.jsonからsections.json互換形式を生成（最小キャッシュ）

    Args:
        harmony_beat: harmony_beat.json内容

    Returns:
        sections_from_harmony_beat.json形式
    """
    chords = harmony_beat.get("chords", [])
    total_bars = harmony_beat.get("meta", {}).get("total_bars")

    if total_bars is None:
        total_bars = max([c.get("bar", 0) for c in chords] + [0]) + 1

    # 各小節のセクション決定（コードのsectionフィールドから）
    section_by_bar = {}
    for chord in chords:
        bar = chord.get("bar")
        section = chord.get("section")
        if bar is not None and section:
            section_by_bar[bar] = section

    bars = []
    last_section = "verse"
    for bar_idx in range(total_bars):
        last_section = section_by_bar.get(bar_idx, last_section)
        bars.append({"bar_index": bar_idx, "section": last_section})

    return {
        "bars": bars,
        "source": "harmony_beat.json",
        "generated_by": "harmony_pipeline_phaseB.py",
    }


def build_emotion_profile_from_harmony_beat(harmony_beat: Dict[str, Any]) -> Dict[str, Any]:
    """
    harmony_beat.jsonからemotion_profile.json互換形式を生成（最小キャッシュ）

    Args:
        harmony_beat: harmony_beat.json内容

    Returns:
        emotion_profile_from_harmony_beat.json形式
    """
    chords = harmony_beat.get("chords", [])
    total_bars = harmony_beat.get("meta", {}).get("total_bars")

    if total_bars is None:
        total_bars = max([c.get("bar", 0) for c in chords] + [0]) + 1

    # 小節ごとの感情情報集約
    accumulator = {
        i: {"energy": [], "valence": [], "tension": [], "tags": []} for i in range(total_bars)
    }

    for chord in chords:
        bar = chord.get("bar")
        if bar is None or bar not in accumulator:
            continue

        # Emotion数値取得
        valence = chord.get("valence")
        tension = chord.get("tension")
        energy = chord.get("energy")

        if isinstance(valence, (int, float)):
            accumulator[bar]["valence"].append(float(valence))
        if isinstance(tension, (int, float)):
            accumulator[bar]["tension"].append(float(tension))
        if isinstance(energy, (int, float)):
            accumulator[bar]["energy"].append(float(energy))

        # Emotion tag取得
        tag = chord.get("xmusic_emotion") or chord.get("emotion")
        if tag:
            accumulator[bar]["tags"].append(str(tag))

    # 小節ごとの平均値計算
    bars = []
    for bar_idx in range(total_bars):
        data = accumulator[bar_idx]

        energy = sum(data["energy"]) / len(data["energy"]) if data["energy"] else 0.5
        valence = sum(data["valence"]) / len(data["valence"]) if data["valence"] else 0.5
        tension = sum(data["tension"]) / len(data["tension"]) if data["tension"] else 0.0
        emotion_tag = data["tags"][-1] if data["tags"] else "neutral"

        bars.append(
            {
                "bar_index": bar_idx,
                "energy": energy,
                "valence": valence,
                "tension": tension,
                "emotion_tag": emotion_tag,
            }
        )

    return {
        "bars": bars,
        "source": "harmony_beat.json",
        "generated_by": "harmony_pipeline_phaseB.py",
    }


class HarmonyPipelinePhaseB:
    """Phase B: Harmony Beat専用パイプライン（完全改修版）"""

    def __init__(
        self,
        song_dir: Path,
        harmony_beat_path: Path,
        skip_humanize: bool = False,
        skip_midi: bool = False,
    ):
        self.song_dir = song_dir
        self.harmony_beat_path = harmony_beat_path
        self.skip_humanize = skip_humanize
        self.skip_midi = skip_midi
        # プロジェクトルート（scripts配下からの相対参照を安定化）
        self.project_root = Path(__file__).resolve().parent.parent

        # ディレクトリ構造
        self.analysis_dir = song_dir / "analysis"
        self.plans_dir = song_dir / "plans"
        self.midi_dir = song_dir / "midi"

        # Policy（harmony_beat専用）
        self.policy_dir = Path("scripts/instrument_harmonybeat/policy")

        # 必須ファイル（bars_with_slots.parquet使用）
        self.bars_with_slots = self.analysis_dir / "bars_with_slots.parquet"
        self.lyric_anchors = self.analysis_dir / "lyric_anchors.json"

        # 派生ファイル（harmony_beatから生成）
        self.sections_cache = self.analysis_dir / "sections_from_harmony_beat.json"
        self.emotion_cache = self.analysis_dir / "emotion_profile_from_harmony_beat.json"

        self.log: List[str] = []

    def _run_cmd(self, cmd: List[str], desc: str) -> None:
        """コマンド実行"""
        print(f"\n{'='*60}")
        print(f"{desc}")
        print(f"{'='*60}")
        print(" ".join(str(c) for c in cmd))
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            self.log.append(f"  ❌ Failed: {desc}")
            raise RuntimeError(f"❌ Command failed: {desc}")
        self.log.append(f"  ✅ Success: {desc}")

    def phase_b0_prepare_caches(self) -> None:
        """Phase B-0: 最小派生キャッシュ生成（sections/emotion）"""
        print("\n" + "=" * 60)
        print("Phase B-0: Minimal Derived Caches（harmony_beat専用）")
        print("=" * 60)

        harmony_beat = load_json(self.harmony_beat_path)

        # sections_from_harmony_beat.json生成
        sections = build_sections_from_harmony_beat(harmony_beat)
        save_json(sections, self.sections_cache)
        print(f"✅ Created: {self.sections_cache}")

        # emotion_profile_from_harmony_beat.json生成
        emotion_profile = build_emotion_profile_from_harmony_beat(harmony_beat)
        save_json(emotion_profile, self.emotion_cache)
        print(f"✅ Created: {self.emotion_cache}")

    def phase_b1_generate_plans(self) -> None:
        """Phase B-1: Plan生成（Expression Layer）"""
        print("\n" + "=" * 60)
        print("Phase B-1: Plan生成（Expression Layer - harmony_beat専用）")
        print("=" * 60)

        # Bass Plan
        self._run_cmd(
            [
                "python3",
                "scripts/generate_bass_plan_from_harmony.py",
                "--bars",
                str(self.bars_with_slots),
                "--harmony-beat",
                str(self.harmony_beat_path),
                "--policy",
                str(self.policy_dir / "bass.yaml"),
                "--out",
                str(self.plans_dir / "bass_plan.json"),
                "--lyric-anchors",
                str(self.lyric_anchors),
                "--emotion-profile",
                str(self.emotion_cache),
                "--rulebook",
                str(self.project_root / "tools" / "bass_rulebook_v0.yaml"),
            ],
            "Bass Plan生成",
        )

        # Guitar Plan
        self._run_cmd(
            [
                "python3",
                "scripts/generate_guitar_plan_from_harmony.py",
                "--bars",
                str(self.bars_with_slots),
                "--harmony-beat",
                str(self.harmony_beat_path),
                "--policy",
                str(self.policy_dir / "guitar.yaml"),
                "--out",
                str(self.plans_dir / "guitar_plan.json"),
                "--lyric-anchors",
                str(self.lyric_anchors),
                "--emotion-profile",
                str(self.emotion_cache),
                "--rulebook",
                str(self.project_root / "config" / "guitar_rulebook_v1.json"),
            ],
            "Guitar Plan生成",
        )

        # Piano Plan
        self._run_cmd(
            [
                "python3",
                "scripts/generate_piano_plan_from_harmony.py",
                "--bars",
                str(self.bars_with_slots),
                "--harmony-beat",
                str(self.harmony_beat_path),
                "--policy",
                str(self.policy_dir / "piano.yaml"),
                "--out",
                str(self.plans_dir / "piano_plan.json"),
                "--lyric-anchors",
                str(self.lyric_anchors),
                "--emotion-profile",
                str(self.emotion_cache),
                "--rulebook",
                str(self.project_root / "config" / "keys_rulebook_v1.json"),
            ],
            "Piano Plan生成",
        )

        # Strings Plan
        self._run_cmd(
            [
                "python3",
                "scripts/generate_strings_plan_from_harmony.py",
                "--bars",
                str(self.bars_with_slots),
                "--harmony-beat",
                str(self.harmony_beat_path),
                "--policy",
                str(self.policy_dir / "strings.yaml"),
                "--out",
                str(self.plans_dir / "strings_plan.json"),
                "--lyric-anchors",
                str(self.lyric_anchors),
                "--emotion-profile",
                str(self.emotion_cache),
                "--rulebook",
                str(self.project_root / "config" / "strings_rulebook_v1.json"),
            ],
            "Strings Plan生成",
        )

        # Drums Plan（Expression Layer）
        self._run_cmd(
            [
                "python3",
                "scripts/generate_drums_plan_from_harmony.py",
                "--bars",
                str(self.bars_with_slots),
                "--harmony-beat",
                str(self.harmony_beat_path),
                "--policy",
                str(self.policy_dir / "drums.yaml"),
                "--out",
                str(self.plans_dir / "drums_plan.json"),
                "--lyric-anchors",
                str(self.lyric_anchors),
                "--emotion-profile",
                str(self.emotion_cache),
                "--rulebook",
                str(self.project_root / "config" / "drum_rulebook_v1.json"),
            ],
            "Drums Plan生成（Expression Layer）",
        )

    def phase_b2_dynamics(self) -> None:
        """Phase B-2: Dynamics Plan生成（Expression層の補完）"""
        print("\n" + "=" * 60)
        print("Phase B-2: Dynamics Plan生成")
        print("=" * 60)

        # ライブラリ関数として直接呼び出し（CLI wrapper不要）
        sys.path.insert(0, str(self.project_root / "scripts"))
        from apply_bass_dynamics_policy_v1_1 import apply_bass_dynamics_policy_v1_1
        from apply_guitar_dynamics_policy_v1_1 import apply_guitar_dynamics_policy_v1_1
        from apply_piano_dynamics_policy_v1_1 import apply_piano_dynamics_policy_v1_1
        from apply_strings_dynamics_policy_v1_1 import apply_strings_dynamics_policy_v1_1
        from drum_dynamics_policy_v1_1 import apply_drum_dynamics_policy_v1_1

        # Bass Dynamics
        print("  🎸 Bass Dynamics適用中...")
        apply_bass_dynamics_policy_v1_1(
            contexts=[],  # 空リスト（sections/emotionから自動構築）
            policy_yaml_path=str(self.project_root / "config" / "bass_dynamics_policy_v1_1.yaml"),
            sections_json_path=str(self.sections_cache),
            emotion_profile_json_path=str(self.emotion_cache),
            out_plan_path=str(self.plans_dir / "bass_dynamics_plan.json"),
        )
        print("  ✅ Bass Dynamics完了")

        # Guitar Dynamics
        print("  🎸 Guitar Dynamics適用中...")
        apply_guitar_dynamics_policy_v1_1(
            contexts=[],
            policy_yaml_path=str(self.project_root / "config" / "guitar_dynamics_policy_v1_1.yaml"),
            sections_json_path=str(self.sections_cache),
            emotion_profile_json_path=str(self.emotion_cache),
            out_plan_path=str(self.plans_dir / "guitar_dynamics_plan.json"),
        )
        print("  ✅ Guitar Dynamics完了")

        # Piano Dynamics
        print("  🎹 Piano Dynamics適用中...")
        apply_piano_dynamics_policy_v1_1(
            contexts=[],
            policy_yaml_path=str(self.project_root / "config" / "piano_dynamics_policy_v1_1.yaml"),
            sections_json_path=str(self.sections_cache),
            emotion_profile_json_path=str(self.emotion_cache),
            out_plan_path=str(self.plans_dir / "piano_dynamics_plan.json"),
        )
        print("  ✅ Piano Dynamics完了")

        # Strings Dynamics
        print("  🎻 Strings Dynamics適用中...")
        apply_strings_dynamics_policy_v1_1(
            contexts=[],
            policy_yaml_path=str(
                self.project_root / "config" / "strings_dynamics_policy_v1_1.yaml"
            ),
            sections_json_path=str(self.sections_cache),
            emotion_profile_json_path=str(self.emotion_cache),
            out_plan_path=str(self.plans_dir / "strings_dynamics_plan.json"),
        )
        print("  ✅ Strings Dynamics完了")

        # Drums Dynamics（Expression Layer専用）
        print("  🥁 Drums Dynamics適用中...")
        apply_drum_dynamics_policy_v1_1(
            contexts=[],
            policy_yaml_path=str(self.project_root / "config" / "drum_dynamics_policy_v1_1.yaml"),
            sections_json_path=str(self.sections_cache),
            emotion_profile_json_path=str(self.emotion_cache),
            out_plan_path=str(self.plans_dir / "drums_dynamics_plan.json"),
        )
        print("  ✅ Drums Dynamics完了")

    def phase_b3_humanize(self) -> None:
        """Phase B-3: Humanization（Performance Layer）"""
        if self.skip_humanize:
            print("\n⏭️  Phase B-3: Humanization スキップ")
            return

        print("\n" + "=" * 60)
        print("Phase B-3: Humanization（Performance Layer）")
        print("=" * 60)

        # Bass Humanize
        if (self.plans_dir / "bass_dynamics_plan.json").exists():
            self._run_cmd(
                [
                    "python3",
                    "scripts/apply_bass_playfulness_v0_1.py",
                    "--bass-plan",
                    str(self.plans_dir / "bass_plan.json"),
                    "--dynamics-plan",
                    str(self.plans_dir / "bass_dynamics_plan.json"),
                    "--sections-json",
                    str(self.sections_cache),
                    "--out",
                    str(self.plans_dir / "bass_plan_humanized.json"),
                ],
                "Bass Humanization",
            )

        # Guitar Humanize
        if (self.plans_dir / "guitar_dynamics_plan.json").exists():
            self._run_cmd(
                [
                    "python3",
                    "scripts/apply_guitar_playfulness_v0_1.py",
                    "--guitar-plan",
                    str(self.plans_dir / "guitar_plan.json"),
                    "--dynamics-plan",
                    str(self.plans_dir / "guitar_dynamics_plan.json"),
                    "--sections-json",
                    str(self.sections_cache),
                    "--out",
                    str(self.plans_dir / "guitar_plan_humanized.json"),
                ],
                "Guitar Humanization",
            )

        # Piano Humanize
        if (self.plans_dir / "piano_dynamics_plan.json").exists():
            self._run_cmd(
                [
                    "python3",
                    "scripts/apply_piano_playfulness_v0_1.py",
                    "--piano-plan",
                    str(self.plans_dir / "piano_plan.json"),
                    "--dynamics-plan",
                    str(self.plans_dir / "piano_dynamics_plan.json"),
                    "--sections-json",
                    str(self.sections_cache),
                    "--out",
                    str(self.plans_dir / "piano_plan_humanized.json"),
                ],
                "Piano Humanization",
            )

        # Strings Humanize
        if (self.plans_dir / "strings_dynamics_plan.json").exists():
            self._run_cmd(
                [
                    "python3",
                    "scripts/apply_strings_playfulness_v0_1.py",
                    "--strings-plan",
                    str(self.plans_dir / "strings_plan.json"),
                    "--dynamics-plan",
                    str(self.plans_dir / "strings_dynamics_plan.json"),
                    "--sections-json",
                    str(self.sections_cache),
                    "--out",
                    str(self.plans_dir / "strings_plan_humanized.json"),
                ],
                "Strings Humanization",
            )

        # Drums Humanize（Performance Layer）
        if (self.plans_dir / "drums_dynamics_plan.json").exists():
            self._run_cmd(
                [
                    "python3",
                    "scripts/run_drums_humanization.py",
                    "--drums-plan",
                    str(self.plans_dir / "drums_plan.json"),
                    "--dynamics-plan",
                    str(self.plans_dir / "drums_dynamics_plan.json"),
                    "--out",
                    str(self.plans_dir / "drums_plan_humanized.json"),
                ],
                "Drums Humanization（Performance Layer）",
            )

    def phase_b3_5_check_lock(self) -> None:
        """Phase B-3.5: Kick×Bass×Keys 同期チェック（QA/レポート）"""
        print("\n" + "=" * 60)
        print("Phase B-3.5: Kick×Bass×Keys Lock Metrics")
        print("=" * 60)

        # humanized優先で入力プランを選択
        drums_plan = self.plans_dir / "drums_plan_humanized.json"
        if not drums_plan.exists():
            drums_plan = self.plans_dir / "drums_plan.json"

        bass_plan = self.plans_dir / "bass_plan_humanized.json"
        if not bass_plan.exists():
            bass_plan = self.plans_dir / "bass_plan.json"

        keys_plan = self.plans_dir / "piano_plan_humanized.json"  # PianoはKeys扱い
        if not keys_plan.exists():
            keys_plan = self.plans_dir / "piano_plan.json"

        out_path = self.plans_dir / "lock_metrics.json"

        if drums_plan.exists() and bass_plan.exists() and keys_plan.exists():
            self._run_cmd(
                [
                    "python3",
                    "scripts/lock_metrics_kick_bass_keys_v1.py",
                    "--drums-plan",
                    str(drums_plan),
                    "--bass-plan",
                    str(bass_plan),
                    "--keys-plan",
                    str(keys_plan),
                    "--out",
                    str(out_path),
                ],
                "Calc Kick×Bass×Keys Lock Metrics",
            )
            print(f"✅ Lock metrics saved: {out_path}")
        else:
            print("⏭️  Skipping Lock Metrics (Missing drums/bass/keys plan)")

    def phase_b4_to_midi(self) -> None:
        """Phase B-4: MIDI変換"""
        if self.skip_midi:
            print("\n⏭️  Phase B-4: MIDI変換 スキップ")
            return

        print("\n" + "=" * 60)
        print("Phase B-4: MIDI変換")
        print("=" * 60)

        self.midi_dir.mkdir(parents=True, exist_ok=True)

        # Bass
        bass_plan = self.plans_dir / "bass_plan_humanized.json"
        if not bass_plan.exists():
            bass_plan = self.plans_dir / "bass_plan.json"

        if bass_plan.exists():
            self._run_cmd(
                [
                    "python3",
                    "scripts/bass_plan_to_midi.py",
                    str(bass_plan),
                    str(self.midi_dir / "bass.mid"),
                ],
                "Bass to MIDI",
            )

        # Guitar
        guitar_plan = self.plans_dir / "guitar_plan_humanized.json"
        if not guitar_plan.exists():
            guitar_plan = self.plans_dir / "guitar_plan.json"

        if guitar_plan.exists():
            self._run_cmd(
                [
                    "python3",
                    "scripts/guitar_plan_to_midi.py",
                    str(guitar_plan),
                    str(self.midi_dir / "guitar.mid"),
                ],
                "Guitar to MIDI",
            )

        # Piano
        piano_plan = self.plans_dir / "piano_plan_humanized.json"
        if not piano_plan.exists():
            piano_plan = self.plans_dir / "piano_plan.json"

        if piano_plan.exists():
            self._run_cmd(
                [
                    "python3",
                    "scripts/keys_plan_to_midi.py",
                    str(piano_plan),
                    str(self.midi_dir / "piano.mid"),
                ],
                "Piano to MIDI",
            )

        # Strings
        strings_plan = self.plans_dir / "strings_plan_humanized.json"
        if not strings_plan.exists():
            strings_plan = self.plans_dir / "strings_plan.json"

        if strings_plan.exists():
            self._run_cmd(
                [
                    "python3",
                    "scripts/strings_plan_to_midi.py",
                    str(strings_plan),
                    str(self.midi_dir / "strings.mid"),
                ],
                "Strings to MIDI",
            )

        # Drums
        drums_plan = self.plans_dir / "drums_plan_humanized.json"
        if not drums_plan.exists():
            drums_plan = self.plans_dir / "drums_plan.json"

        if drums_plan.exists():
            self._run_cmd(
                [
                    "python3",
                    "scripts/drums_plan_to_midi.py",
                    str(drums_plan),
                    str(self.midi_dir / "drums.mid"),
                ],
                "Drums to MIDI",
            )

    def run(self) -> None:
        """全フェーズ実行"""
        print("\n" + "=" * 60)
        print("🎵 Harmony Pipeline Phase B（完全改修版）")
        print("=" * 60)
        print(f"Song Dir: {self.song_dir}")
        print(f"Harmony Beat: {self.harmony_beat_path}")
        print(f"Policy Dir: {self.policy_dir}")
        print("=" * 60)

        self.phase_b0_prepare_caches()
        self.phase_b1_generate_plans()
        self.phase_b2_dynamics()
        self.phase_b3_humanize()
        self.phase_b3_5_check_lock()
        self.phase_b4_to_midi()

        print("\n" + "=" * 60)
        print("✅ Phase B完了")
        print("=" * 60)
        for entry in self.log:
            print(entry)


def main():
    parser = argparse.ArgumentParser(
        description="Harmony Pipeline Phase B（完全改修版 - chordmap完全排除）"
    )
    parser.add_argument(
        "--song-dir",
        type=Path,
        required=True,
        help="Song directory (e.g., data/suno_ai/suno_themesong/song_004)",
    )
    parser.add_argument(
        "--harmony-beat", type=Path, required=True, help="Path to harmony_beat.json (唯一の真理値)"
    )
    parser.add_argument("--skip-humanize", action="store_true", help="Skip humanization phase")
    parser.add_argument("--skip-midi", action="store_true", help="Skip MIDI conversion phase")
    args = parser.parse_args()

    pipeline = HarmonyPipelinePhaseB(
        song_dir=args.song_dir,
        harmony_beat_path=args.harmony_beat,
        skip_humanize=args.skip_humanize,
        skip_midi=args.skip_midi,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
