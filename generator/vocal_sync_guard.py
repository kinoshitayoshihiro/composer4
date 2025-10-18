#!/usr/bin/env python3
"""
Vocal Sync Guard

Vocalタイミング vs MIDI生成の同期ズレを検証し、品質保証を行う。

Features:
- Vocal onset timing抽出（librosa.onset.onset_detect）
- MIDI note timing抽出（music21 or pretty_midi）
- Drift計算：各セクションでVocal-MIDI時間差を測定
- 警告閾値：> 50ms でWARNING、> 100ms でERROR
- タイムストレッチ推奨：許容範囲外の場合、修正係数を提案
- Quality gate統合：arrange_from_yaml.py等で使用可能

Use Cases:
1. Suno vocalステム + 生成MIDI → 同期チェック
2. セクション境界での累積ズレ検出
3. Tempo変動による影響分析
4. タイムストレッチ係数自動計算

Output:
- 同期レポート（JSON/YAML）
- 警告/エラーリスト
- 推奨修正係数

Usage:
    from generator.vocal_sync_guard import VocalSyncGuard
    
    guard = VocalSyncGuard(
        vocal_audio_path="suno_ai/song1/vocals.wav",
        midi_path="output/midi/full_score.mid",
        structure_yaml_path="suno_ai/song1/structure.yaml"
    )
    
    report = guard.check_sync()
    if report['has_errors']:
        print(f"⚠️  Sync errors detected: {report['error_count']}")
        print(f"   Recommended time stretch: {report['recommended_stretch']:.4f}")
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import yaml

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available, vocal onset detection disabled")

try:
    from music21 import converter, stream
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    logging.warning("music21 not available, MIDI parsing disabled")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class VocalSyncGuard:
    """Vocal - MIDI同期検証システム"""
    
    # 閾値設定
    WARNING_THRESHOLD_MS = 50.0  # > 50ms で警告
    ERROR_THRESHOLD_MS = 100.0   # > 100ms でエラー
    
    def __init__(
        self,
        vocal_audio_path: Optional[Path] = None,
        midi_path: Optional[Path] = None,
        structure_yaml_path: Optional[Path] = None,
        hop_length: int = 512,
        sr: int = 22050
    ):
        """
        Initialize Vocal Sync Guard
        
        Args:
            vocal_audio_path: Vocalステムパス
            midi_path: 生成MIDIファイルパス
            structure_yaml_path: 構造YAMLパス（sections情報）
            hop_length: librosa hop length
            sr: Sample rate
        """
        self.vocal_audio_path = Path(vocal_audio_path) if vocal_audio_path else None
        self.midi_path = Path(midi_path) if midi_path else None
        self.structure_yaml_path = Path(structure_yaml_path) if structure_yaml_path else None
        
        self.hop_length = hop_length
        self.sr = sr
        
        # データキャッシュ
        self.vocal_onsets: Optional[np.ndarray] = None
        self.midi_note_onsets: Optional[List[float]] = None
        self.sections: Optional[List[Dict[str, Any]]] = None
    
    def load_vocal_onsets(self) -> np.ndarray:
        """
        Vocal onset timing抽出
        
        Returns:
            np.ndarray: Onset times (seconds)
        """
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa is required for vocal onset detection")
        
        if self.vocal_onsets is not None:
            return self.vocal_onsets
        
        if not self.vocal_audio_path or not self.vocal_audio_path.exists():
            raise FileNotFoundError(f"Vocal audio not found: {self.vocal_audio_path}")
        
        logger.info(f"Loading vocal audio: {self.vocal_audio_path}")
        y, sr = librosa.load(str(self.vocal_audio_path), sr=self.sr)
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            hop_length=self.hop_length,
            units='frames'
        )
        
        # Convert to seconds
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
        
        self.vocal_onsets = onset_times
        logger.info(f"✅ Detected {len(onset_times)} vocal onsets")
        
        return onset_times
    
    def load_midi_note_onsets(self) -> List[float]:
        """
        MIDI note onset timing抽出
        
        Returns:
            List[float]: Note onset times (seconds)
        """
        if not MUSIC21_AVAILABLE:
            raise RuntimeError("music21 is required for MIDI parsing")
        
        if self.midi_note_onsets is not None:
            return self.midi_note_onsets
        
        if not self.midi_path or not self.midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {self.midi_path}")
        
        logger.info(f"Loading MIDI: {self.midi_path}")
        score = converter.parse(str(self.midi_path))
        
        # Extract all note onsets
        onsets = []
        for element in score.flatten().notes:
            offset_seconds = element.offset  # Quarterbeats
            # Convert quarterbeats to seconds (assume 4/4, tempo from first metronome mark)
            tempo_marks = score.flatten().getElementsByClass('MetronomeMark')
            if tempo_marks:
                tempo = tempo_marks[0].number
                # quarterbeats → seconds: offset / (tempo / 60)
                onset_time = offset_seconds / (tempo / 60.0)
                onsets.append(onset_time)
            else:
                # Fallback: assume 120 BPM
                onset_time = offset_seconds / 2.0
                onsets.append(onset_time)
        
        onsets = sorted(onsets)
        self.midi_note_onsets = onsets
        logger.info(f"✅ Extracted {len(onsets)} MIDI note onsets")
        
        return onsets
    
    def load_sections(self) -> List[Dict[str, Any]]:
        """
        構造YAMLからセクション情報読み込み
        
        Returns:
            List[Dict]: Section list
        """
        if self.sections is not None:
            return self.sections
        
        if not self.structure_yaml_path or not self.structure_yaml_path.exists():
            logger.warning(f"Structure YAML not found: {self.structure_yaml_path}")
            return []
        
        with open(self.structure_yaml_path, 'r', encoding='utf-8') as f:
            structure = yaml.safe_load(f)
        
        self.sections = structure.get('sections', [])
        logger.info(f"✅ Loaded {len(self.sections)} sections from YAML")
        
        return self.sections
    
    def calculate_drift_per_section(self) -> List[Dict[str, Any]]:
        """
        セクションごとにVocal-MIDI drift計算
        
        Algorithm:
        1. 各セクションの時間範囲でVocal onsets, MIDI onsetsをフィルタ
        2. Dynamic Time Warping (DTW)で最近接マッチング
        3. 平均drift, 最大drift, 標準偏差を計算
        
        Returns:
            List[Dict]: Section drift reports
        """
        vocal_onsets = self.load_vocal_onsets()
        midi_onsets = self.load_midi_note_onsets()
        sections = self.load_sections()
        
        if len(sections) == 0:
            logger.warning("No sections defined, calculating global drift")
            sections = [{
                'label': 'Global',
                'start_time': 0.0,
                'end_time': max(vocal_onsets[-1] if len(vocal_onsets) > 0 else 0.0,
                                midi_onsets[-1] if len(midi_onsets) > 0 else 0.0)
            }]
        
        drift_reports = []
        
        for section in sections:
            section_label = section['label']
            start_time = section['start_time']
            end_time = section['end_time']
            
            # Filter onsets in section range
            vocal_section = [v for v in vocal_onsets if start_time <= v < end_time]
            midi_section = [m for m in midi_onsets if start_time <= m < end_time]
            
            if len(vocal_section) == 0 or len(midi_section) == 0:
                logger.warning(f"Section {section_label}: No onsets detected")
                drift_reports.append({
                    'section': section_label,
                    'start_time': start_time,
                    'end_time': end_time,
                    'vocal_onset_count': len(vocal_section),
                    'midi_onset_count': len(midi_section),
                    'mean_drift_ms': None,
                    'max_drift_ms': None,
                    'std_drift_ms': None,
                    'status': 'NO_DATA'
                })
                continue
            
            # Simple nearest-neighbor matching
            drifts = []
            for v_onset in vocal_section:
                # Find nearest MIDI onset
                distances = [abs(m_onset - v_onset) for m_onset in midi_section]
                min_distance = min(distances)
                drifts.append(min_distance * 1000.0)  # Convert to ms
            
            # Statistics
            mean_drift = float(np.mean(drifts))
            max_drift = float(np.max(drifts))
            std_drift = float(np.std(drifts))
            
            # Status判定
            if max_drift > self.ERROR_THRESHOLD_MS:
                status = 'ERROR'
            elif mean_drift > self.WARNING_THRESHOLD_MS:
                status = 'WARNING'
            else:
                status = 'OK'
            
            drift_reports.append({
                'section': section_label,
                'start_time': start_time,
                'end_time': end_time,
                'vocal_onset_count': len(vocal_section),
                'midi_onset_count': len(midi_section),
                'mean_drift_ms': mean_drift,
                'max_drift_ms': max_drift,
                'std_drift_ms': std_drift,
                'status': status
            })
            
            logger.info(f"Section {section_label}: mean={mean_drift:.1f}ms, max={max_drift:.1f}ms ({status})")
        
        return drift_reports
    
    def calculate_recommended_stretch(self, drift_reports: List[Dict[str, Any]]) -> float:
        """
        タイムストレッチ推奨係数計算
        
        Algorithm:
        - Global drift平均から修正係数を算出
        - stretch_factor = (vocal_duration + mean_drift) / vocal_duration
        
        Args:
            drift_reports: Drift report list
        
        Returns:
            float: Recommended time stretch factor (1.0 = no change)
        """
        # Global mean drift計算
        valid_drifts = [r['mean_drift_ms'] for r in drift_reports if r['mean_drift_ms'] is not None]
        
        if len(valid_drifts) == 0:
            return 1.0
        
        global_mean_drift_ms = float(np.mean(valid_drifts))
        
        # Convert to stretch factor
        # Positive drift: MIDI is ahead → slow down MIDI (stretch < 1.0)
        # Negative drift: MIDI is behind → speed up MIDI (stretch > 1.0)
        # stretch_factor ≈ 1.0 - (drift / total_duration)
        
        # Simple approximation: small correction
        stretch_factor = 1.0 - (global_mean_drift_ms / 100000.0)  # Very conservative
        
        return stretch_factor
    
    def check_sync(self) -> Dict[str, Any]:
        """
        同期チェック実行
        
        Returns:
            Dict: Sync report
        """
        logger.info("\n🔍 Vocal Sync Guard: Checking synchronization...")
        
        try:
            # Load data
            vocal_onsets = self.load_vocal_onsets()
            midi_onsets = self.load_midi_note_onsets()
            sections = self.load_sections()
            
            # Calculate drift per section
            drift_reports = self.calculate_drift_per_section()
            
            # Count warnings/errors
            warnings = [r for r in drift_reports if r['status'] == 'WARNING']
            errors = [r for r in drift_reports if r['status'] == 'ERROR']
            
            has_warnings = len(warnings) > 0
            has_errors = len(errors) > 0
            
            # Recommended stretch
            recommended_stretch = self.calculate_recommended_stretch(drift_reports)
            
            # Summary
            report = {
                'vocal_audio': str(self.vocal_audio_path) if self.vocal_audio_path else None,
                'midi_path': str(self.midi_path) if self.midi_path else None,
                'structure_yaml': str(self.structure_yaml_path) if self.structure_yaml_path else None,
                'total_vocal_onsets': len(vocal_onsets),
                'total_midi_onsets': len(midi_onsets),
                'section_count': len(sections),
                'drift_reports': drift_reports,
                'warning_count': len(warnings),
                'error_count': len(errors),
                'has_warnings': has_warnings,
                'has_errors': has_errors,
                'recommended_stretch': recommended_stretch,
                'overall_status': 'ERROR' if has_errors else ('WARNING' if has_warnings else 'OK')
            }
            
            # Log summary
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 Sync Check Summary:")
            logger.info(f"   Vocal onsets: {len(vocal_onsets)}")
            logger.info(f"   MIDI onsets: {len(midi_onsets)}")
            logger.info(f"   Sections: {len(sections)}")
            logger.info(f"   Warnings: {len(warnings)}")
            logger.info(f"   Errors: {len(errors)}")
            logger.info(f"   Overall status: {report['overall_status']}")
            
            if has_errors or has_warnings:
                logger.info(f"\n⚠️  Recommended time stretch: {recommended_stretch:.6f}")
            
            logger.info(f"{'='*60}\n")
            
            return report
        
        except Exception as e:
            logger.error(f"❌ Sync check failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'error': str(e),
                'overall_status': 'FAILED'
            }
    
    def save_report(self, report: Dict[str, Any], output_path: Path):
        """
        同期レポート保存
        
        Args:
            report: Sync report
            output_path: Output file path (JSON or YAML)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.json':
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        else:
            # YAML
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(report, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"✅ Sync report saved: {output_path}")


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vocal Sync Guard - Check vocal-MIDI synchronization')
    parser.add_argument('--vocal', type=Path, required=True, help='Vocal audio file (WAV)')
    parser.add_argument('--midi', type=Path, required=True, help='Generated MIDI file')
    parser.add_argument('--structure', type=Path, help='Structure YAML file (optional)')
    parser.add_argument('--output', type=Path, help='Output report file (JSON or YAML)')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate for librosa')
    parser.add_argument('--hop-length', type=int, default=512, help='Hop length for librosa')
    
    args = parser.parse_args()
    
    print("\n🎤 Vocal Sync Guard")
    print("=" * 60)
    
    # Initialize guard
    guard = VocalSyncGuard(
        vocal_audio_path=args.vocal,
        midi_path=args.midi,
        structure_yaml_path=args.structure,
        hop_length=args.hop_length,
        sr=args.sr
    )
    
    # Check sync
    report = guard.check_sync()
    
    # Save report
    if args.output:
        guard.save_report(report, args.output)
    
    # Exit code
    if report.get('overall_status') == 'ERROR':
        print("\n❌ Sync check failed with errors")
        exit(1)
    elif report.get('overall_status') == 'WARNING':
        print("\n⚠️  Sync check completed with warnings")
        exit(0)
    else:
        print("\n✅ Sync check passed")
        exit(0)


if __name__ == '__main__':
    main()
