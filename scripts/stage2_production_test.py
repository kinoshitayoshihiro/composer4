#!/usr/bin/env python3
"""
Stage2 実戦投入スクリプト

Suno AI stemデータで4楽器Stage2の動作確認とメトリクス収集

Usage:
    python scripts/stage2_production_test.py \
        --input data/suno_ai/suno_themesong/song_001/stemswav_001 \
        --output data/stage2_test_output \
        --tempo 120 \
        --emotion energetic
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
    from music21 import stream, note, tempo as m21tempo, meter, instrument
except ImportError:
    raise ImportError("music21 required: pip install music21")

# Stage2 imports
from generator.bass_params_stage2 import BassParamsStage2
from generator.piano_params_stage2 import PianoParamsStage2
from generator.strings_params_stage2 import StringsParamsStage2
from generator.guitar_params_stage2 import GuitarParamsStage2
from generator.drums_params_stage2 import DrumsParamsStage2
from generator.instrument_stage2_base import load_yaml_presets

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


class Stage2ProductionTester:
    """Stage2実戦テスター"""
    
    def __init__(self):
        # Stage2インスタンス初期化
        self.bass_stage2 = self._init_bass_stage2()
        self.piano_stage2 = self._init_piano_stage2()
        self.strings_stage2 = self._init_strings_stage2()
        self.guitar_stage2 = self._init_guitar_stage2()
        self.drums_stage2 = self._init_drums_stage2()
        
        self.metrics_history: List[Dict[str, Any]] = []
    
    def _init_bass_stage2(self) -> Optional[BassParamsStage2]:
        """Bass Stage2初期化"""
        try:
            preset_path = PROJECT_ROOT / "data/presets/bass_style_presets.yaml"
            presets = load_yaml_presets(preset_path)
            logger.info(f"Bass presets loaded: {list(presets.keys())}")
            return BassParamsStage2(style_presets=presets)
        except Exception as e:
            logger.warning(f"Bass Stage2 init failed: {e}")
            return None
    
    def _init_piano_stage2(self) -> Optional[PianoParamsStage2]:
        """Piano Stage2初期化"""
        try:
            preset_path = PROJECT_ROOT / "data/presets/piano_style_presets.yaml"
            presets = load_yaml_presets(preset_path)
            logger.info(f"Piano presets loaded: {list(presets.keys())}")
            return PianoParamsStage2(style_presets=presets)
        except Exception as e:
            logger.warning(f"Piano Stage2 init failed: {e}")
            return None
    
    def _init_strings_stage2(self) -> Optional[StringsParamsStage2]:
        """Strings Stage2初期化"""
        try:
            preset_path = PROJECT_ROOT / "data/presets/strings_style_presets.yaml"
            presets = load_yaml_presets(preset_path)
            logger.info(f"Strings presets loaded: {list(presets.keys())}")
            return StringsParamsStage2(style_presets=presets)
        except Exception as e:
            logger.warning(f"Strings Stage2 init failed: {e}")
            return None
    
    def _init_guitar_stage2(self) -> Optional[GuitarParamsStage2]:
        """Guitar Stage2初期化"""
        try:
            preset_path = PROJECT_ROOT / "data/presets/guitar_style_presets.yaml"
            presets = load_yaml_presets(preset_path)
            logger.info(f"Guitar presets loaded: {list(presets.keys())}")
            return GuitarParamsStage2(style_presets=presets)
        except Exception as e:
            logger.warning(f"Guitar Stage2 init failed: {e}")
            return None
    
    def _init_drums_stage2(self) -> Optional[DrumsParamsStage2]:
        """Drums Stage2初期化"""
        try:
            preset_path = PROJECT_ROOT / "data/presets/drums_style_presets.yaml"
            presets = load_yaml_presets(preset_path)
            logger.info(f"Drums presets loaded: {list(presets.keys())}")
            return DrumsParamsStage2(style_presets=presets)
        except Exception as e:
            logger.warning(f"Drums Stage2 init failed: {e}")
            return None
    
    def create_mock_part(
        self,
        instrument_name: str,
        bars: int = 8,
        tempo_bpm: float = 120.0
    ) -> stream.Part:
        """モックパート生成（テスト用）"""
        part = stream.Part()
        part.id = instrument_name
        part.partName = instrument_name.capitalize()
        
        # 楽器設定
        if instrument_name == "bass":
            part.insert(0, instrument.ElectricBass())
            pitch_range = range(36, 64)  # E1-E3
        elif instrument_name == "piano":
            part.insert(0, instrument.Piano())
            pitch_range = range(48, 84)  # C3-C6
        elif instrument_name == "strings":
            part.insert(0, instrument.StringInstrument())
            pitch_range = range(55, 88)  # G3-E6
        elif instrument_name == "guitar":
            part.insert(0, instrument.ElectricGuitar())
            pitch_range = range(40, 76)  # E2-E5
        elif instrument_name == "drums":
            part.insert(0, instrument.Percussion())
            pitch_range = range(35, 60)  # GMドラム範囲
        else:
            pitch_range = range(48, 72)
        
        # テンポ・拍子
        part.insert(0, m21tempo.MetronomeMark(number=tempo_bpm))
        part.insert(0, meter.TimeSignature('4/4'))
        
        # ダミーノート生成（8分音符）
        offset = 0.0
        note_idx = 0
        for _ in range(bars * 8):
            midi = pitch_range[note_idx % len(pitch_range)]
            n = note.Note(midi=midi)
            n.quarterLength = 0.5
            n.volume.velocity = 80
            part.insert(offset, n)
            offset += 0.5
            note_idx += 1
        
        return part
    
    def test_instrument(
        self,
        stage2_instance: Any,
        instrument_name: str,
        style: str,
        emotion: str = "energetic",
        bars: int = 8,
        tempo: float = 120.0,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """単一楽器テスト"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {instrument_name.upper()} with style='{style}'")
        logger.info(f"{'='*60}")
        
        # モックパート生成
        part = self.create_mock_part(instrument_name, bars, tempo)
        original_note_count = len(list(part.flatten().notes))
        
        logger.info(f"Original notes: {original_note_count}")
        
        # section_meta
        section_meta = {
            "label": "TestSection",
            "bar": 0,
            "emotion": emotion,
            "tempo": tempo,
            f"{instrument_name}_style": style,
        }
        
        # mix_context（空でOK）
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
            
            # メトリクス取得
            metrics = stage2_instance.metrics.copy()
            metrics["instrument"] = instrument_name
            metrics["style"] = style
            metrics["emotion"] = emotion
            metrics["elapsed_sec"] = round(elapsed, 3)
            metrics["original_note_count"] = original_note_count
            
            logger.info(f"✅ Stage2 applied in {elapsed:.3f}s")
            logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
            
            return {
                "success": True,
                "metrics": metrics,
                "part": part
            }
        
        except Exception as e:
            logger.exception(f"❌ Stage2 apply failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "instrument": instrument_name,
                "style": style
            }
    
    def run_full_test(
        self,
        emotion: str = "energetic",
        bars: int = 8,
        tempo: float = 120.0,
        seed: int = 42
    ) -> Dict[str, Any]:
        """全楽器フルテスト"""
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Stage2 Production Test")
        logger.info(f"# Emotion: {emotion}, Bars: {bars}, Tempo: {tempo}")
        logger.info(f"{'#'*60}\n")
        
        results = {}
        
        # Bass
        if self.bass_stage2:
            for style in ["tight_pop", "loose_indie", "funk_groove", "jazz_walking"]:
                result = self.test_instrument(
                    self.bass_stage2, "bass", style, emotion, bars, tempo, seed
                )
                results[f"bass_{style}"] = result
                if result["success"]:
                    self.metrics_history.append(result["metrics"])
        
        # Piano
        if self.piano_stage2:
            for style in ["ballad_drop2", "pop_comp", "jazz_rootless", "edm_stabs"]:
                result = self.test_instrument(
                    self.piano_stage2, "piano", style, emotion, bars, tempo, seed
                )
                results[f"piano_{style}"] = result
                if result["success"]:
                    self.metrics_history.append(result["metrics"])
        
        # Strings
        if self.strings_stage2:
            for style in ["pad_cinematic", "ostinato_rhythmic", "divisi_rich", "minimalist"]:
                result = self.test_instrument(
                    self.strings_stage2, "strings", style, emotion, bars, tempo, seed
                )
                results[f"strings_{style}"] = result
                if result["success"]:
                    self.metrics_history.append(result["metrics"])
        
        # Guitar
        if self.guitar_stage2:
            for style in ["strum_pop_clean", "fingerstyle_folk", "power_chord_rock", "jazz_comp"]:
                result = self.test_instrument(
                    self.guitar_stage2, "guitar", style, emotion, bars, tempo, seed
                )
                results[f"guitar_{style}"] = result
                if result["success"]:
                    self.metrics_history.append(result["metrics"])
        
        # Drums
        if self.drums_stage2:
            for style in ["simple", "moderate", "complex", "intense"]:
                result = self.test_instrument(
                    self.drums_stage2, "drums", style, emotion, bars, tempo, seed
                )
                results[f"drums_{style}"] = result
                if result["success"]:
                    self.metrics_history.append(result["metrics"])
        
        return results
    
    def save_metrics_report(self, output_dir: Path):
        """メトリクスレポート保存"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1) メトリクス履歴JSON
        metrics_json = output_dir / "stage2_metrics.json"
        with open(metrics_json, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Metrics saved: {metrics_json}")
        
        # 2) サマリーレポート
        summary_md = output_dir / "stage2_summary.md"
        with open(summary_md, 'w', encoding='utf-8') as f:
            f.write("# Stage2 Production Test Summary\n\n")
            f.write(f"**Total Tests**: {len(self.metrics_history)}\n\n")
            
            # 楽器別集計
            by_instrument = {}
            for m in self.metrics_history:
                inst = m.get("instrument", "unknown")
                by_instrument.setdefault(inst, []).append(m)
            
            f.write("## By Instrument\n\n")
            for inst, metrics in sorted(by_instrument.items()):
                f.write(f"### {inst.capitalize()}\n\n")
                f.write(f"- Tests: {len(metrics)}\n")
                
                avg_elapsed = sum(m["elapsed_sec"] for m in metrics) / len(metrics)
                f.write(f"- Avg Elapsed: {avg_elapsed:.3f}s\n")
                
                # 共通メトリクス
                if "note_count" in metrics[0]:
                    avg_notes = sum(m.get("note_count", 0) for m in metrics) / len(metrics)
                    f.write(f"- Avg Note Count: {avg_notes:.1f}\n")
                
                if "vel_mean" in metrics[0]:
                    avg_vel = sum(m.get("vel_mean", 0) for m in metrics) / len(metrics)
                    f.write(f"- Avg Velocity: {avg_vel:.1f}\n")
                
                # 楽器固有メトリクス
                if inst == "bass" and "lock_ratio_with_kick" in metrics[0]:
                    avg_lock = sum(m.get("lock_ratio_with_kick", 0) for m in metrics) / len(metrics)
                    f.write(f"- Avg Kick Lock Ratio: {avg_lock:.3f}\n")
                
                f.write("\n")
            
            f.write("## Metrics Detail\n\n")
            f.write("| Instrument | Style | Notes | Vel Mean | Vel Std | Elapsed |\n")
            f.write("|------------|-------|-------|----------|---------|----------|\n")
            
            for m in self.metrics_history:
                f.write(f"| {m['instrument']} | {m['style']} | ")
                f.write(f"{m.get('note_count', 0)} | ")
                f.write(f"{m.get('vel_mean', 0):.1f} | ")
                f.write(f"{m.get('vel_std', 0):.1f} | ")
                f.write(f"{m['elapsed_sec']:.3f}s |\n")
        
        logger.info(f"✅ Summary saved: {summary_md}")


def main():
    parser = argparse.ArgumentParser(description="Stage2 Production Test")
    parser.add_argument("--output", type=str, default="data/stage2_test_output",
                        help="Output directory for test results")
    parser.add_argument("--emotion", type=str, default="energetic",
                        help="Emotion (energetic/melancholic/calm/aggressive/romantic)")
    parser.add_argument("--bars", type=int, default=8,
                        help="Number of bars to test")
    parser.add_argument("--tempo", type=float, default=120.0,
                        help="Tempo (BPM)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # テスター初期化
    tester = Stage2ProductionTester()
    
    # フルテスト実行
    results = tester.run_full_test(
        emotion=args.emotion,
        bars=args.bars,
        tempo=args.tempo,
        seed=args.seed
    )
    
    # メトリクス保存
    output_dir = Path(args.output)
    tester.save_metrics_report(output_dir)
    
    # サマリー表示
    success_count = sum(1 for r in results.values() if r.get("success"))
    total_count = len(results)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Test Complete: {success_count}/{total_count} passed")
    logger.info(f"📊 Metrics: {output_dir / 'stage2_metrics.json'}")
    logger.info(f"📄 Summary: {output_dir / 'stage2_summary.md'}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
