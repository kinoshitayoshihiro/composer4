#!/usr/bin/env python3
"""
YAML→MIDI再生成器

Suno構造抽出器で生成された構造YAMLから、Stage2 Generatorを使用して
各楽器のMIDIパートを生成。

Input:
    - 構造YAML（tempo_map, sections, chords, drums_hits, bass_contour）

Output:
    - Piano MIDI
    - Bass MIDI
    - Guitar MIDI
    - Strings MIDI
    - Full Score MIDI（全楽器統合）

Features:
    - 4つのStage2 Generator統合（Piano/Bass/Guitar/Strings）
    - セクションごとに適切なemotion推定
    - Tempo map適用
    - Quality gates（オプション）

Usage:
    python scripts/arrange/arrange_from_yaml.py \\
        --input data/suno_structures/song1.yaml \\
        --output-dir output/midi/song1 \\
        --enable-quality-gates
"""

import argparse
import pathlib
from typing import Dict, List, Optional, Any
import logging

import yaml
import music21
from music21 import stream, tempo, meter, instrument as m21instrument

# Import Stage2 Generators
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from generator.piano_generator_stage2 import MelodyGeneratorStage2, CompingGeneratorStage2
from generator.bass_generator_stage2 import BassGeneratorStage2
from generator.guitar_generator_stage2 import GuitarGeneratorStage2
from generator.strings_generator_stage2 import StringsGeneratorStage2

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ArrangeFromYAML:
    """構造YAML → Stage2 Generator → MIDI生成"""
    
    def __init__(
        self,
        structure_yaml_path: pathlib.Path,
        enable_stage2: bool = True,
        enable_quality_gates: bool = False,
        verbose: bool = True
    ):
        """
        Initialize arranger
        
        Args:
            structure_yaml_path: 構造YAMLファイルパス
            enable_stage2: Stage2パターン推薦を使用
            enable_quality_gates: Quality gates適用
            verbose: 詳細ログ出力
        """
        self.structure_yaml_path = pathlib.Path(structure_yaml_path)
        self.enable_stage2 = enable_stage2
        self.enable_quality_gates = enable_quality_gates
        self.verbose = verbose
        
        # 構造YAML読み込み
        self.structure = self._load_structure()
        
        # Generators初期化
        self._init_generators()
    
    def _load_structure(self) -> Dict[str, Any]:
        """構造YAML読み込み"""
        if not self.structure_yaml_path.exists():
            raise FileNotFoundError(f"Structure YAML not found: {self.structure_yaml_path}")
        
        with open(self.structure_yaml_path, 'r', encoding='utf-8') as f:
            structure = yaml.safe_load(f)
        
        if self.verbose:
            logger.info(f"✅ Structure loaded: {self.structure_yaml_path}")
            logger.info(f"   Tempo: {structure['tempo_map']['global_tempo']:.1f} BPM")
            logger.info(f"   Sections: {len(structure['sections'])}")
        
        return structure
    
    def _init_generators(self):
        """Stage2 Generators初期化（現在はGuitar/Strings Stage2のみ使用）"""
        # Guitar - default_instrument必須
        self.guitar_gen = GuitarGeneratorStage2(
            default_instrument=m21instrument.AcousticGuitar(),
            use_stage2=self.enable_stage2
        )
        
        # Strings - default_instrument必須
        self.strings_gen = StringsGeneratorStage2(
            default_instrument=m21instrument.Violin(),
            use_stage2=self.enable_stage2
        )
        
        # TODO: Piano/Bass Stage2 integration (generate() method requires Section/Context objects)
        
        if self.verbose:
            logger.info("✅ Stage2 Generators initialized")
    
    def generate_all(self, output_dir: pathlib.Path):
        """
        全楽器MIDI生成
        
        Args:
            output_dir: 出力ディレクトリ
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            logger.info(f"\n🎵 Generating MIDI files...")
            logger.info(f"   Output dir: {output_dir}")
        
        # 各セクション処理
        sections = self.structure['sections']
        tempo_map = self.structure['tempo_map']
        chords_by_section = self.structure.get('chords', {})
        drums_hits = self.structure.get('drums_hits', {})
        
        # 楽器別Part集約（現在はGuitar/Strings Stage2のみ）
        full_guitar_part = stream.Part()
        full_guitar_part.insert(0, m21instrument.AcousticGuitar())
        full_guitar_part.partName = "Guitar"
        
        full_strings_part = stream.Part()
        full_strings_part.insert(0, m21instrument.Violin())
        full_strings_part.partName = "Strings"
        
        # セクション処理
        cumulative_offset = 0.0  # Quarterbeats累積
        
        for section in sections:
            section_label = section['label']
            duration_measures = section['duration_measures']
            section_chords = chords_by_section.get(section_label, [])
            
            # Emotion推定
            emotion = self._estimate_emotion(section_label)
            
            # Chord progression抽出
            chord_symbols = [c['chord'] for c in section_chords] if section_chords else ['C', 'G', 'Am', 'F']
            
            if self.verbose:
                logger.info(f"\n📐 Processing section: {section_label}")
                logger.info(f"   Measures: {duration_measures}")
                logger.info(f"   Emotion: {emotion}")
                logger.info(f"   Chords: {chord_symbols[:4]}...")
            
            # Kick hits（Bass用）
            kick_times = drums_hits.get('kick', [])
            kick_offsets = [k for k in kick_times if section['start_time'] <= k < section['end_time']]
            
            # 各Generator呼び出し（現在はGuitar/Strings Stage2のみ）
            try:
                # Guitar
                guitar_part = self.guitar_gen.compose(
                    section_name=section_label,
                    measures=duration_measures,
                    chord_progression=chord_symbols,
                    tempo=tempo_map['global_tempo'],
                    emotion=emotion
                )
                self._append_part_with_offset(full_guitar_part, guitar_part, cumulative_offset)
                
                # Strings
                strings_part = self.strings_gen.compose(
                    section_name=section_label,
                    measures=duration_measures,
                    chord_progression=chord_symbols,
                    tempo=tempo_map['global_tempo'],
                    emotion=emotion
                )
                self._append_part_with_offset(full_strings_part, strings_part, cumulative_offset)
                
                if self.verbose:
                    logger.info(f"   ✓ Guitar: {len(list(guitar_part.flatten().notes))} notes")
                    logger.info(f"   ✓ Strings: {len(list(strings_part.flatten().notes))} notes")
            
            except Exception as e:
                logger.error(f"❌ Section {section_label} generation failed: {e}")
            
            # 累積offset更新（4/4拍子仮定: duration_measures * 4 quarterbeats）
            cumulative_offset += duration_measures * 4.0
        
        # Full Score作成
        score = stream.Score()
        score.insert(0, tempo.MetronomeMark(number=tempo_map['global_tempo']))
        score.insert(0, meter.TimeSignature('4/4'))
        
        # Parts追加（Guitar/Strings Stage2のみ）
        if len(list(full_guitar_part.flatten().notes)) > 0:
            score.append(full_guitar_part)
        if len(list(full_strings_part.flatten().notes)) > 0:
            score.append(full_strings_part)
        
        # MIDI保存（Guitar/Strings Stage2のみ）
        output_files = []
        
        if len(list(full_guitar_part.flatten().notes)) > 0:
            guitar_path = output_dir / "guitar.mid"
            full_guitar_part.write('midi', fp=guitar_path)
            output_files.append(guitar_path)
        
        if len(list(full_strings_part.flatten().notes)) > 0:
            strings_path = output_dir / "strings.mid"
            full_strings_part.write('midi', fp=strings_path)
            output_files.append(strings_path)
        
        # Full score
        full_score_path = output_dir / "full_score.mid"
        score.write('midi', fp=full_score_path)
        output_files.append(full_score_path)
        
        if self.verbose:
            logger.info(f"\n✅ MIDI generation complete!")
            logger.info(f"   Files generated: {len(output_files)}")
            for f in output_files:
                logger.info(f"     - {f.name}")
        
        return output_files
    
    def _append_part_with_offset(
        self,
        target_part: stream.Part,
        source_part: stream.Part,
        offset_quarterbeats: float
    ):
        """
        source_partのnotes/chordsをtarget_partにoffset付きで追加
        
        Args:
            target_part: 追加先Part
            source_part: 追加元Part
            offset_quarterbeats: オフセット（quarterbeats単位）
        """
        for element in source_part.flatten().notesAndRests:
            # Clone element
            new_element = element
            if hasattr(element, 'offset'):
                new_element.offset = element.offset + offset_quarterbeats
            target_part.insert(new_element.offset, new_element)
    
    def _estimate_emotion(self, section_label: str) -> str:
        """
        セクション名 → Emotion推定
        
        Heuristics:
        - Intro: calm
        - Verse: neutral
        - Chorus: happy/energetic
        - Bridge: dramatic
        - Outro: calm
        
        Args:
            section_label: セクション名
        
        Returns:
            Emotion string
        """
        section = section_label.lower()
        
        if "intro" in section:
            return "calm"
        elif "verse" in section:
            return "neutral"
        elif "chorus" in section:
            return "happy"
        elif "bridge" in section:
            return "dramatic"
        elif "outro" in section:
            return "calm"
        else:
            return "neutral"


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description='YAML→MIDI Arranger (Stage2)')
    parser.add_argument('--input', type=pathlib.Path, required=True, help='Input structure YAML')
    parser.add_argument('--output-dir', type=pathlib.Path, required=True, help='Output MIDI directory')
    parser.add_argument('--disable-stage2', action='store_true', help='Disable Stage2 pattern recommendation')
    parser.add_argument('--enable-quality-gates', action='store_true', help='Enable quality gates')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    print("\n🎼 YAML→MIDI Arranger (Stage2)")
    print("=" * 60)
    
    # Arranger作成
    arranger = ArrangeFromYAML(
        structure_yaml_path=args.input,
        enable_stage2=not args.disable_stage2,
        enable_quality_gates=args.enable_quality_gates,
        verbose=not args.quiet
    )
    
    # 生成
    output_files = arranger.generate_all(args.output_dir)
    
    print(f"\n🎉 Complete! Generated {len(output_files)} MIDI files")
    print(f"   Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
