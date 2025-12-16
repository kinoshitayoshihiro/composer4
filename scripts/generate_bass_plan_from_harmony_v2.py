#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_bass_plan_from_harmony_v2.py

Harmony Beat → BarContext → BassPhraseGenerator v2 + DynamicsRouter
AIレイヤー封印版（感情ラベル + dynamics_routing_bass.yaml のみ）

目的:
- harmony_beat.jsonを唯一のソースとして
- emotion_tags.yaml + dynamics_profiles.yaml + dynamics_routing_bass.yaml で動作
- AI (GuideToneAI, ModeScale, Magenta, DUV) は全て無効化
- v2 系の「純粋な policy + playfulness + dynamics_policy」だけを評価

入力:
- harmony_beat.json
- config/emotion_tags.yaml
- config/dynamics_profiles.yaml
- config/dynamics_routing_bass.yaml
- config/bass_rulebook_v2.yaml (optional)

出力:
- bass_plan_harmony_v2.json
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# プロジェクトルートをsys.pathに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "scripts"))

from bass_phrase_generator_models_v2 import BarContext, BassEvent, bass_events_to_plan_json
from bass_rulebook_v2 import BassRulebook
from bass_phrase_generator_v2 import BassPhraseGenerator
from harmony_pipeline_phaseB import (
    build_sections_from_harmony_beat,
    build_emotion_profile_from_harmony_beat,
)
from emotion_resolver import EmotionResolver


def load_harmony_beat(path: Path) -> Dict[str, Any]:
    """harmony_beat.jsonを読み込む"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_chord_symbol_for_bar(bar_index: int, harmony_beat: Dict[str, Any]) -> Optional[str]:
    """指定小節の最初のコードシンボルを取得"""
    chords = harmony_beat.get("chords", [])
    for chord in chords:
        if chord.get("bar") == bar_index:
            return chord.get("symbol")
    return None


def get_chord_function_for_bar(bar_index: int, harmony_beat: Dict[str, Any]) -> str:
    """指定小節のコード機能を取得"""
    chords = harmony_beat.get("chords", [])
    for chord in chords:
        if chord.get("bar") == bar_index:
            func = chord.get("function")
            if func:
                return func.upper()
    return "TONIC"


def infer_bass_role_from_emotion(emotion_tag: str, section: str) -> str:
    """
    emotion_tagとsectionからbass_roleを推測

    シンプルなマッピング:
    - BRIGHT/ENERGETIC → WALKING
    - SAD/DARK → MELODIC
    - NEUTRAL → ROOT_FIFTH
    - chorus → RHYTHMIC_ACCENT
    - intro/outro → ROOT_FIFTH
    """
    section_lower = section.lower()
    emotion_upper = emotion_tag.upper()

    # Sectionベース
    if "chorus" in section_lower or "climax" in section_lower:
        return "RHYTHMIC_ACCENT"
    if "intro" in section_lower or "outro" in section_lower:
        return "ROOT_FIFTH"

    # Emotion ベース
    if "BRIGHT" in emotion_upper or "ENERGETIC" in emotion_upper or "EPIC" in emotion_upper:
        return "WALKING"
    if "SAD" in emotion_upper or "DARK" in emotion_upper or "MELANCHOLIC" in emotion_upper:
        return "MELODIC"
    if "AGGRESSIVE" in emotion_upper or "ANGRY" in emotion_upper:
        return "SYNCOPATED"

    # デフォルト
    return "ROOT_FIFTH"


def infer_kick_pattern_from_emotion(emotion_tag: str, section: str) -> str:
    """
    emotion_tagとsectionからkick_pattern_tagを推測

    シンプルなマッピング:
    - ENERGETIC → 4_ON_FLOOR
    - NEUTRAL → TWO_FOUR
    - SAD → SPARSE
    """
    emotion_upper = emotion_tag.upper()
    section_lower = section.lower()

    if "chorus" in section_lower or "climax" in section_lower:
        return "4_ON_FLOOR"
    if "ENERGETIC" in emotion_upper or "EPIC" in emotion_upper or "BRIGHT" in emotion_upper:
        return "4_ON_FLOOR"
    if "SAD" in emotion_upper or "DARK" in emotion_upper or "MELANCHOLIC" in emotion_upper:
        return "SPARSE"

    return "TWO_FOUR"


def infer_vocal_density_from_section(section: str) -> str:
    """
    sectionからvocal_densityを推測

    簡易版:
    - chorus → high
    - verse → medium
    - intro/bridge → low
    """
    section_lower = section.lower()

    if "chorus" in section_lower or "climax" in section_lower:
        return "high"
    if "verse" in section_lower:
        return "medium"
    if "intro" in section_lower or "bridge" in section_lower or "outro" in section_lower:
        return "low"

    return "medium"


def create_bar_contexts_from_harmony(
    harmony_beat: Dict[str, Any],
    sections_data: Dict[str, Any],
    emotion_profile: Dict[str, Any],
    use_emotion_resolver: bool = True,
) -> List[BarContext]:
    """
    harmony_beat.jsonからBarContextのリストを生成

    Args:
        harmony_beat: harmony_beat.json
        sections_data: build_sections_from_harmony_beat()の結果
        emotion_profile: build_emotion_profile_from_harmony_beat()の結果
        use_emotion_resolver: EmotionResolverを使用するか (デフォルト: True)

    Returns:
        BarContextのリスト
    """
    meta = harmony_beat.get("meta", {})
    beats_per_bar = meta.get("beats_per_bar", 4)
    tempo_bpm = meta.get("bpm", 89.3)

    # key_root を tonic_key から抽出 (例: "C# minor" → "C#")
    tonic_key = meta.get("tonic_key", "C# minor")
    key_root = tonic_key.split()[0] if tonic_key else "C#"

    total_bars = harmony_beat.get("meta", {}).get("total_bars")
    if total_bars is None:
        chords = harmony_beat.get("chords", [])
        total_bars = max([c.get("bar", 0) for c in chords] + [0]) + 1

    # sectionsをbar_indexでマッピング
    section_by_bar = {}
    for section in sections_data.get("sections", []):
        for bar_idx in range(section["start_bar"], section["end_bar"] + 1):
            section_by_bar[bar_idx] = section["label"]

    # emotion_profileをbar_indexでマッピング
    emotion_by_bar = {}
    for bar_data in emotion_profile.get("bars", []):
        bar_idx = bar_data.get("bar_index")
        if bar_idx is not None:
            emotion_by_bar[bar_idx] = bar_data

    # EmotionResolver初期化（オプション）
    resolver = EmotionResolver() if use_emotion_resolver else None

    # BarContext生成
    bar_contexts = []
    start_beat = 0.0

    for bar_idx in range(total_bars):
        section = section_by_bar.get(bar_idx, "verse")
        emotion_data = emotion_by_bar.get(bar_idx, {})

        emotion_tag = emotion_data.get("emotion_tag", "NEUTRAL_COOL")
        energy = emotion_data.get("energy", 0.5)
        valence = emotion_data.get("valence", 0.0)
        tension = emotion_data.get("tension", 0.0)

        chord_symbol = get_chord_symbol_for_bar(bar_idx, harmony_beat)
        chord_function = get_chord_function_for_bar(bar_idx, harmony_beat)

        # EmotionResolverがあれば正式解決、なければ簡易フォールバック
        resolved = {}
        if resolver:
            try:
                resolved = resolver.resolve_all(
                    valence=valence,
                    arousal=energy,
                    section_name=section,
                )
                # resolverは section を正規化するので上書き
                section = resolved.get("section_type", section)
                emotion_tag = resolved.get("emotion_tag", emotion_tag)
            except Exception as e:
                print(f"⚠️  EmotionResolver failed for bar {bar_idx}: {e}")
                resolved = {}

        bass_role = resolved.get("bass_role") or infer_bass_role_from_emotion(emotion_tag, section)
        # kick_pattern_tag は専用関数で推定（rhythm_density とは別）
        kick_pattern_tag = infer_kick_pattern_from_emotion(emotion_tag, section)
        vocal_density = resolved.get("vocal_density") or infer_vocal_density_from_section(section)
        rhythm_density = resolved.get("rhythm_density", "MEDIUM")

        # arousal を energy に対応
        arousal = energy

        ctx = BarContext(
            bar_index=bar_idx,
            start_beat=start_beat,
            section=section,
            style_id="ROCK_8BEAT_ROOT",  # デフォルト
            chord_symbol=chord_symbol or "C",
            chord_function=chord_function,
            key_root=key_root,  # meta から取得
            meter=f"{beats_per_bar}/4",  # meta から取得
            tempo_bpm=tempo_bpm,  # meta から取得
            groove_type="STANDARD_8BEAT_ROCK",  # デフォルト
            kick_pattern_tag=kick_pattern_tag,
            vocal_density=vocal_density,
            vocal_activity_score=0.5,  # デフォルト
            vocal_long_tone=False,
            prev_bass_pitch=None,  # TODO: 前小節の最後の音高を保持
            prev_bar_tension=tension,
            valence=valence,
            arousal=arousal,
            emotion_tag=emotion_tag,
            bass_role=bass_role,
            rhythm_density=rhythm_density,
        )

        bar_contexts.append(ctx)
        start_beat += beats_per_bar

    return bar_contexts


def main():
    parser = argparse.ArgumentParser(
        description="Generate bass plan from harmony_beat.json using v2 system (AI封印版)"
    )
    parser.add_argument(
        "--harmony-beat",
        required=True,
        type=Path,
        help="Path to harmony_beat.json",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output bass plan JSON path",
    )
    parser.add_argument(
        "--rulebook",
        type=Path,
        default=Path("config/bass_rulebook_v2.yaml"),
        help="Bass rulebook YAML",
    )
    parser.add_argument(
        "--default-velocity",
        type=int,
        default=90,
        help="Default velocity",
    )
    args = parser.parse_args()

    print("🎸 Generating Bass Plan from Harmony Beat (v2 + DynamicsRouter)")
    print(f"   Harmony Beat: {args.harmony_beat}")
    print(f"   Rulebook: {args.rulebook}")
    print(f"   Output: {args.out}")
    print("   🚫 AI Layers: DISABLED (GuideToneAI, ModeScale, Magenta, DUV)\n")

    # 1. harmony_beat.json読み込み
    print("📖 Loading harmony_beat...")
    harmony_beat = load_harmony_beat(args.harmony_beat)
    print(f"   ✅ Loaded {len(harmony_beat.get('chords', []))} chords\n")

    # 2. sections & emotion_profile 生成
    print("🔧 Building sections & emotion_profile from harmony_beat...")
    sections_data = build_sections_from_harmony_beat(harmony_beat)
    emotion_profile = build_emotion_profile_from_harmony_beat(harmony_beat)
    print(f"   ✅ Sections: {len(sections_data.get('sections', []))}")
    print(f"   ✅ Emotion bars: {len(emotion_profile.get('bars', []))}\n")

    # 3. BarContext生成
    print("📊 Creating BarContexts...")
    bar_contexts = create_bar_contexts_from_harmony(harmony_beat, sections_data, emotion_profile)
    print(f"   ✅ Created {len(bar_contexts)} BarContexts\n")

    # セクション統計
    section_counts = {}
    for ctx in bar_contexts:
        section_counts[ctx.section] = section_counts.get(ctx.section, 0) + 1

    print("📈 Section distribution:")
    for sec, count in sorted(section_counts.items()):
        print(f"   {sec}: {count} bars")
    print()

    # 4. BassRulebook生成
    print("📖 Loading BassRulebook...")
    try:
        rulebook = BassRulebook.load_from_yaml(args.rulebook)
        stats = rulebook.stats()
        print(
            f"   ✅ Loaded rulebook: {stats.get('total_rules', 0)} rules / "
            f"{stats.get('style_ids', 0)} styles"
        )
    except Exception as exc:
        print(f"   ⚠️  Rulebook load failed ({exc}), falling back to empty rulebook")
        rulebook = BassRulebook(rules=[])

    # 5. BassPhraseGenerator v2生成（DynamicsRouter有効化）
    print("🎼 Generating bass plan (with DynamicsRouter)...")
    generator = BassPhraseGenerator(
        rulebook=rulebook,
        enable_dynamics=True,  # DynamicsRouter有効化
        enable_bass_dyn_policy=True,  # Bass Dynamics Policy v1.1有効化
    )
    events = generator.generate_song(bar_contexts)  # generate_song()が正しいメソッド名
    print(f"   ✅ Generated {len(events)} bass events\n")

    # イベント統計
    if events:
        velocities = [e.velocity for e in events]
        techniques = [e.technique for e in events if e.technique]

        print("📈 Event statistics:")
        print(f"   Velocity range: {min(velocities)} - {max(velocities)}")
        print(f"   Avg velocity: {sum(velocities) / len(velocities):.1f}")

        if techniques:
            tech_counts = {}
            for tech in techniques:
                tech_counts[tech] = tech_counts.get(tech, 0) + 1

            print("   Techniques:")
            for tech, count in sorted(tech_counts.items()):
                print(f"     {tech}: {count}")
        print()

    # 6. JSON保存
    print(f"💾 Saving plan to {args.out}...")
    plan_data = bass_events_to_plan_json(events)
    plan_data.setdefault("meta", {})
    plan_data["meta"]["source"] = "harmony_beat.json"
    plan_data["meta"]["generator"] = "generate_bass_plan_from_harmony_v2.py"
    plan_data["meta"]["ai_enabled"] = False
    plan_data["meta"]["dynamics_enabled"] = True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(plan_data, f, indent=2, ensure_ascii=False)

    print(f"   ✅ Saved: {args.out}")
    print(f"   File size: {args.out.stat().st_size / 1024:.1f} KB\n")

    # 7. サンプルイベント表示
    print("🎵 Sample events (first 5):")
    for i, e in enumerate(events[:5]):
        bar = e.bar
        beat = e.beat
        role = e.role
        technique = e.technique or "N/A"

        print(f"   [{i}] Bar {bar}, Beat {beat:.2f} | {role}")
        print(f"       Pitch: {e.pitch}, Vel: {e.velocity}, Tech: {technique}")

    print("\n✅ Bass plan generation complete!")
    print("\n💡 Next step: Convert to MIDI with:")
    print("   python3 scripts/bass_plan_to_midi.py \\")
    print(f"     --plan {args.out} \\")
    print("     --out data/suno_ai/suno_themesong/song_004/midi/bass_harmony_v2.mid \\")
    print("     --tempo-map data/suno_ai/suno_themesong/song_004/analysis/tempo_map.json \\")
    print("     --program 33")


if __name__ == "__main__":
    main()
