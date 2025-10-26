#!/usr/bin/env python3
"""
Stage2 Suno Stem 実戦適用スクリプト

SunoのstemwavデータにStage2を適用してメトリクス収集

Usage:
    python scripts/stage2_suno_stem_test.py \
        --input data/suno_ai/suno_themesong/song_001/stemswav_001 \
        --output data/stage2_suno_output
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

# パス設定
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from music21 import stream, converter
except ImportError:
    raise ImportError("music21 required: pip install music21")

# Stage2 imports
from generator.bass_params_stage2 import BassParamsStage2
from generator.piano_params_stage2 import PianoParamsStage2
from generator.strings_params_stage2 import StringsParamsStage2
from generator.guitar_params_stage2 import GuitarParamsStage2
from generator.instrument_stage2_base import load_yaml_presets

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


class SunoStemStage2Applier:
    """Suno Stem WAVにStage2適用"""
    
    def __init__(self):
        # Stage2インスタンス
        self.stage2_map = {
            "bass": self._init_bass_stage2(),
            "piano": self._init_piano_stage2(),
            "keyboard": self._init_piano_stage2(),  # KeyboardはPiano扱い
            "strings": self._init_strings_stage2(),
            "guitar": self._init_guitar_stage2(),
        }
        
        self.metrics_history = []
    
    def _init_bass_stage2(self) -> Optional[BassParamsStage2]:
        try:
            preset_path = PROJECT_ROOT / "data/presets/bass_style_presets.yaml"
            presets = load_yaml_presets(preset_path)
            return BassParamsStage2(style_presets=presets)
        except Exception as e:
            logger.warning(f"Bass Stage2 init failed: {e}")
            return None
    
    def _init_piano_stage2(self) -> Optional[PianoParamsStage2]:
        try:
            preset_path = PROJECT_ROOT / "data/presets/piano_style_presets.yaml"
            presets = load_yaml_presets(preset_path)
            return PianoParamsStage2(style_presets=presets)
        except Exception as e:
            logger.warning(f"Piano Stage2 init failed: {e}")
            return None
    
    def _init_strings_stage2(self) -> Optional[StringsParamsStage2]:
        try:
            preset_path = PROJECT_ROOT / "data/presets/strings_style_presets.yaml"
            presets = load_yaml_presets(preset_path)
            return StringsParamsStage2(style_presets=presets)
        except Exception as e:
            logger.warning(f"Strings Stage2 init failed: {e}")
            return None
    
    def _init_guitar_stage2(self) -> Optional[GuitarParamsStage2]:
        try:
            preset_path = PROJECT_ROOT / "data/presets/guitar_style_presets.yaml"
            presets = load_yaml_presets(preset_path)
            return GuitarParamsStage2(style_presets=presets)
        except Exception as e:
            logger.warning(f"Guitar Stage2 init failed: {e}")
            return None
    
    def detect_instrument_from_path(self, wav_path: Path) -> Optional[str]:
        """WAVパスから楽器名を検出"""
        name = wav_path.stem.lower()
        
        if "(bass)" in name or "_bass" in name:
            return "bass"
        elif "(keyboard)" in name or "_keyboard" in name:
            return "keyboard"
        elif "(piano)" in name or "_piano" in name:
            return "piano"
        elif "(strings)" in name or "_strings" in name:
            return "strings"
        elif "(guitar)" in name or "_guitar" in name:
            return "guitar"
        else:
            return None
    
    def process_stem_directory(
        self,
        stem_dir: Path,
        emotion: str = "energetic",
        tempo: float = 120.0,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Stem WAVディレクトリ全体を処理"""
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Processing Suno Stems: {stem_dir.name}")
        logger.info(f"# Emotion: {emotion}, Tempo: {tempo}")
        logger.info(f"{'#'*60}\n")
        
        results = {}
        wav_files = list(stem_dir.glob("*.wav"))
        
        logger.info(f"Found {len(wav_files)} WAV files")
        
        for wav_file in sorted(wav_files):
            instrument = self.detect_instrument_from_path(wav_file)
            
            if not instrument:
                logger.info(f"⏭️  Skipping {wav_file.name} (unknown instrument)")
                continue
            
            stage2 = self.stage2_map.get(instrument)
            
            if not stage2:
                logger.info(f"⏭️  Skipping {wav_file.name} ({instrument}: no Stage2)")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {wav_file.name} → {instrument}")
            logger.info(f"{'='*60}")
            
            try:
                result = self.process_single_stem(
                    wav_file, instrument, stage2, emotion, tempo, seed
                )
                results[wav_file.stem] = result
                
                if result.get("success"):
                    self.metrics_history.append(result["metrics"])
            
            except Exception as e:
                logger.exception(f"Failed to process {wav_file.name}: {e}")
                results[wav_file.stem] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def process_single_stem(
        self,
        wav_path: Path,
        instrument: str,
        stage2_instance: Any,
        emotion: str,
        tempo: float,
        seed: int
    ) -> Dict[str, Any]:
        """単一Stem処理"""
        
        # NOTE: 現実のWAV→MIDI変換は別ツール必要
        # ここではモックパート生成（実データテストはフェーズ2）
        logger.info(f"⚠️  WAV→MIDI変換は未実装。モックパート使用。")
        
        part = self._create_mock_from_wav_metadata(wav_path, instrument, tempo)
        
        original_notes = len(list(part.flatten().notes))
        
        # スタイル選択（emotion based）
        style = self._select_style_for_emotion(instrument, emotion)
        
        logger.info(f"Style: {style}")
        logger.info(f"Original notes: {original_notes}")
        
        # section_meta
        section_meta = {
            "label": "SunoStem",
            "bar": 0,
            "emotion": emotion,
            "tempo": tempo,
            f"{instrument}_style": style,
        }
        
        mix_context = {}
        
        # Stage2適用
        start_time = time.time()
        
        try:
            stage2_instance.apply(
                part=part,
                section_meta=section_meta,
                mix_context=mix_context,
                overrides={"style": style},
                seed=seed
            )
            
            elapsed = time.time() - start_time
            
            metrics = stage2_instance.metrics.copy()
            metrics.update({
                "instrument": instrument,
                "style": style,
                "emotion": emotion,
                "wav_file": wav_path.name,
                "elapsed_sec": round(elapsed, 3),
                "original_note_count": original_notes
            })
            
            logger.info(f"✅ Stage2 applied in {elapsed:.3f}s")
            logger.info(f"Metrics: {json.dumps({k: v for k, v in metrics.items() if k != 'wav_file'}, indent=2)}")
            
            return {
                "success": True,
                "metrics": metrics,
                "part": part
            }
        
        except Exception as e:
            logger.exception(f"Stage2 apply failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "instrument": instrument,
                "wav_file": wav_path.name
            }
    
    def _create_mock_from_wav_metadata(
        self,
        wav_path: Path,
        instrument: str,
        tempo: float
    ) -> stream.Part:
        """WAVメタデータからモックパート生成"""
        from music21 import note, meter, tempo as m21tempo, instrument as m21instr
        
        part = stream.Part()
        part.id = instrument
        part.partName = instrument.capitalize()
        
        # 楽器設定
        if instrument == "bass":
            part.insert(0, m21instr.ElectricBass())
            pitches = [40, 43, 45, 47, 48, 50, 52, 55]  # E-G range
        elif instrument in ["piano", "keyboard"]:
            part.insert(0, m21instr.Piano())
            pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C4-C5
        elif instrument == "strings":
            part.insert(0, m21instr.StringInstrument())
            pitches = [55, 57, 59, 60, 62, 64, 65, 67]  # G3-G4
        elif instrument == "guitar":
            part.insert(0, m21instr.ElectricGuitar())
            pitches = [40, 45, 48, 52, 55, 57, 60, 64]  # E2-E4
        else:
            pitches = [60, 62, 64, 65, 67]
        
        part.insert(0, m21tempo.MetronomeMark(number=tempo))
        part.insert(0, meter.TimeSignature('4/4'))
        
        # 8小節分のダミーノート
        offset = 0.0
        for i in range(32):
            midi = pitches[i % len(pitches)]
            n = note.Note(midi=midi)
            n.quarterLength = 0.5
            n.volume.velocity = 75 + (i % 20)
            part.insert(offset, n)
            offset += 0.5
        
        return part
    
    def _select_style_for_emotion(self, instrument: str, emotion: str) -> str:
        """Emotionに応じたスタイル選択"""
        style_map = {
            "bass": {
                "energetic": "funk_groove",
                "melancholic": "jazz_walking",
                "calm": "loose_indie",
                "aggressive": "tight_pop",
                "romantic": "jazz_walking"
            },
            "piano": {
                "energetic": "pop_comp",
                "melancholic": "ballad_drop2",
                "calm": "ballad_drop2",
                "aggressive": "edm_stabs",
                "romantic": "jazz_rootless"
            },
            "keyboard": {
                "energetic": "pop_comp",
                "melancholic": "ballad_drop2",
                "calm": "ballad_drop2",
                "aggressive": "edm_stabs",
                "romantic": "jazz_rootless"
            },
            "strings": {
                "energetic": "ostinato_rhythmic",
                "melancholic": "pad_cinematic",
                "calm": "minimalist",
                "aggressive": "ostinato_rhythmic",
                "romantic": "divisi_rich"
            },
            "guitar": {
                "energetic": "power_chord_rock",
                "melancholic": "fingerstyle_folk",
                "calm": "fingerstyle_folk",
                "aggressive": "power_chord_rock",
                "romantic": "jazz_comp"
            }
        }
        
        return style_map.get(instrument, {}).get(emotion, "tight_pop")
    
    def save_metrics_report(self, output_dir: Path):
        """メトリクスレポート保存"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # メトリクスJSON
        metrics_json = output_dir / "suno_stem_metrics.json"
        with open(metrics_json, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Metrics saved: {metrics_json}")
        
        # サマリー
        summary_md = output_dir / "suno_stem_summary.md"
        with open(summary_md, 'w', encoding='utf-8') as f:
            f.write("# Suno Stem Stage2 Application Summary\n\n")
            f.write(f"**Total Stems Processed**: {len(self.metrics_history)}\n\n")
            
            f.write("## Results\n\n")
            f.write("| WAV File | Instrument | Style | Notes | Vel Mean | Elapsed |\n")
            f.write("|----------|------------|-------|-------|----------|----------|\n")
            
            for m in self.metrics_history:
                f.write(f"| {m['wav_file']} | {m['instrument']} | {m['style']} | ")
                f.write(f"{m.get('note_count', 0)} | {m.get('vel_mean', 0):.1f} | ")
                f.write(f"{m['elapsed_sec']:.3f}s |\n")
        
        logger.info(f"✅ Summary saved: {summary_md}")


def main():
    parser = argparse.ArgumentParser(description="Stage2 Suno Stem Test")
    parser.add_argument("--input", type=str, 
                        default="data/suno_ai/suno_themesong/song_001/stemswav_001",
                        help="Suno stem WAV directory")
    parser.add_argument("--output", type=str, default="data/stage2_suno_output",
                        help="Output directory")
    parser.add_argument("--emotion", type=str, default="energetic",
                        help="Emotion preset")
    parser.add_argument("--tempo", type=float, default=120.0,
                        help="Tempo (BPM)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    stem_dir = Path(args.input)
    
    if not stem_dir.exists():
        logger.error(f"Stem directory not found: {stem_dir}")
        sys.exit(1)
    
    applier = SunoStemStage2Applier()
    
    results = applier.process_stem_directory(
        stem_dir=stem_dir,
        emotion=args.emotion,
        tempo=args.tempo,
        seed=args.seed
    )
    
    output_dir = Path(args.output)
    applier.save_metrics_report(output_dir)
    
    success_count = sum(1 for r in results.values() if r.get("success"))
    total_count = len(results)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Processing Complete: {success_count}/{total_count} stems")
    logger.info(f"📊 Metrics: {output_dir / 'suno_stem_metrics.json'}")
    logger.info(f"📄 Summary: {output_dir / 'suno_stem_summary.md'}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
