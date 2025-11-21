#!/usr/bin/env python3
"""
Stage3 Step3: Batch Articulation Renderer (VioPTT/MOSA-VPT)

technique_map.yamlに基づいて、既存MIDIに奏法情報を合成。
Stage3生成モデルの学習データを増強。

Usage:
    python scripts/batch_articulation_renderer.py \
        --input-dir output/drumloops_cleaned \
        --technique-map configs/labels/technique_map.yaml \
        --output-dir outputs/stage3/technique_synth \
        --sample 10  # テスト用
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import copy

import yaml
import mido


class ArticulationRenderer:
    """奏法合成エンジン"""

    def __init__(self, technique_map_path: Path):
        with technique_map_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.instruments = self.config.get("instruments", {})

    def render_drums_articulation(
        self, midi: mido.MidiFile, technique: str
    ) -> Tuple[mido.MidiFile, Dict[str, Any]]:
        """ドラム奏法を適用"""

        drum_config = self.instruments.get("drums", {})
        articulations = drum_config.get("articulations", {})

        if technique not in articulations:
            return midi, {"technique": "none", "modified": False}

        artic_spec = articulations[technique]
        new_midi = copy.deepcopy(midi)
        modified_notes = 0

        # 例: ghost note処理
        if technique == "ghost":
            target_velocity = artic_spec.get("velocity", 34)
            for track in new_midi.tracks:
                for msg in track:
                    if msg.type == "note_on" and msg.channel == 9:  # Drums
                        if msg.note == artic_spec.get("pitch", 38):  # snare
                            # velocityをghost levelに下げる
                            original_vel = msg.velocity
                            if original_vel > 50:  # ghostに変換
                                msg.velocity = target_velocity
                                modified_notes += 1

        # 例: flam処理
        elif technique == "flam":
            grace_pitch = artic_spec.get("grace_pitch", 37)
            grace_velocity = artic_spec.get("grace_velocity", 72)

            for track in new_midi.tracks:
                new_messages = []
                for i, msg in enumerate(track):
                    new_messages.append(msg)

                    if msg.type == "note_on" and msg.channel == 9:
                        if msg.note == artic_spec.get("primary_pitch", 38):
                            # grace noteを挿入 (簡略化: 直前に追加)
                            grace_msg = mido.Message(
                                "note_on",
                                channel=9,
                                note=grace_pitch,
                                velocity=grace_velocity,
                                time=msg.time,
                            )
                            new_messages.insert(-1, grace_msg)
                            modified_notes += 1

                track[:] = new_messages

        metadata = {
            "technique": technique,
            "modified": modified_notes > 0,
            "modified_notes": modified_notes,
            "description": artic_spec.get("description", ""),
        }

        return new_midi, metadata

    def render(
        self, input_path: Path, technique: str, instrument_type: str = "drums"
    ) -> Tuple[Optional[mido.MidiFile], Dict[str, Any]]:
        """MIDIファイルに奏法を適用"""

        try:
            midi = mido.MidiFile(input_path)
        except Exception as e:
            return None, {"error": str(e)}

        if instrument_type == "drums":
            return self.render_drums_articulation(midi, technique)
        else:
            # 他の楽器はTODO
            return midi, {"technique": "none", "instrument_type": instrument_type}


def main():
    parser = argparse.ArgumentParser(description="Batch render articulations for MIDI files")
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Directory containing input MIDI files"
    )
    parser.add_argument("--technique-map", type=Path, required=True, help="Technique map YAML")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for synthesized MIDI"
    )
    parser.add_argument(
        "--techniques", nargs="+", default=["ghost", "flam"], help="Techniques to synthesize"
    )
    parser.add_argument("--instrument", default="drums", help="Instrument type")
    parser.add_argument("--sample", type=int, help="Process only N files (for testing)")

    args = parser.parse_args()

    print(f"Loading technique map from {args.technique_map}")
    renderer = ArticulationRenderer(args.technique_map)

    # 入力ファイル収集 (.mid/.midi両方をサポート)
    input_files = sorted({*args.input_dir.rglob("*.mid"), *args.input_dir.rglob("*.midi")})
    if args.sample:
        input_files = input_files[: args.sample]

    print(f"Found {len(input_files)} MIDI files")
    print(f"Techniques to render: {args.techniques}")

    # 出力ディレクトリ作成
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # メタデータ収集
    all_metadata = []

    for technique in args.techniques:
        tech_dir = args.output_dir / technique
        tech_dir.mkdir(exist_ok=True)

        print(f"\n🎵 Rendering technique: {technique}")

        for i, input_path in enumerate(input_files):
            # 奏法適用
            rendered_midi, metadata = renderer.render(input_path, technique, args.instrument)

            if rendered_midi is None:
                print(f"  ⚠️  Skipped {input_path.name}: {metadata.get('error')}")
                continue

            # 出力パス
            output_path = tech_dir / input_path.name
            rendered_midi.save(output_path)

            # メタデータ記録
            metadata_entry = {
                "input_file": str(input_path.relative_to(args.input_dir)),
                "output_file": str(output_path.relative_to(args.output_dir)),
                "technique": technique,
                **metadata,
            }
            all_metadata.append(metadata_entry)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(input_files)}...")

        print(f"  ✅ Rendered {len(input_files)} files with '{technique}'")

    # メタデータ保存
    metadata_path = args.output_dir / "technique_metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for entry in all_metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n📊 Saved metadata to {metadata_path}")
    print(f"✅ Total synthesized files: {len(all_metadata)}")


if __name__ == "__main__":
    main()
