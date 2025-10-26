#!/usr/bin/env python3
"""
MoisesDB Quality Filter - MIDI変換品質スコアリング

Features:
- MIDI変換品質の自動評価（0-1スコア）
- 低品質データの自動フィルタリング
- 品質メトリクス可視化

Quality Metrics:
1. Note Density (ノート密度)
   - 適切な範囲: 0.5-5.0 notes/sec
   - sparse (<0.5) or dense (>10) → 低品質

2. Pitch Range (ピッチ範囲)
   - 適切な範囲: 24-96 (C1-C7)
   - narrow (<24) or extreme (>96) → 低品質

3. Harmonic Ratio (和音率)
   - 複数音が同時発音: 15-70%
   - too monophonic or polyphonic → 低品質

4. Velocity Variance (ベロシティ分散)
   - 適切な分散: std > 10
   - flat dynamics → 低品質

5. Duration Distribution (音長分布)
   - 多様性: エントロピー > 1.5
   - monotonous → 低品質

Usage:
    # 単一ファイル評価
    python scripts/moisesdb_quality_filter.py \\
        --midi-file data/moisesdb_midi/song_001.mid \\
        --verbose
    
    # バッチ評価
    python scripts/moisesdb_quality_filter.py \\
        --midi-dir data/moisesdb_midi \\
        --output-csv data/quality_scores.csv \\
        --threshold 0.6
    
    # データベースフィルタリング
    python scripts/moisesdb_quality_filter.py \\
        --db-path data/moisesdb_unified.db \\
        --filter-low-quality \\
        --threshold 0.6
"""

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pretty_midi import PrettyMIDI


# ========== Quality Metrics ==========

class MIDIQualityMetrics:
    """MIDI品質メトリクス計算"""
    
    def __init__(self):
        # 閾値設定
        self.note_density_range = (0.5, 5.0)  # notes/sec
        self.pitch_range_range = (24, 96)     # MIDI note numbers
        self.harmonic_ratio_range = (0.15, 0.70)  # 15-70%
        self.velocity_std_min = 10.0
        self.duration_entropy_min = 1.5
    
    def calculate_all_metrics(
        self,
        midi_path: Path
    ) -> Dict[str, Any]:
        """
        全品質メトリクスを計算
        
        Returns:
            {
                'note_density': float,
                'note_density_score': float (0-1),
                'pitch_range': int,
                'pitch_range_score': float (0-1),
                'harmonic_ratio': float,
                'harmonic_ratio_score': float (0-1),
                'velocity_variance': float,
                'velocity_score': float (0-1),
                'duration_entropy': float,
                'duration_score': float (0-1),
                'overall_score': float (0-1),
                'quality_grade': str (A/B/C/D/F)
            }
        """
        try:
            midi = PrettyMIDI(str(midi_path))
            
            # メトリクス計算
            note_density, note_density_score = self._calc_note_density(midi)
            pitch_range, pitch_range_score = self._calc_pitch_range(midi)
            harmonic_ratio, harmonic_ratio_score = self._calc_harmonic_ratio(midi)
            velocity_var, velocity_score = self._calc_velocity_variance(midi)
            duration_entropy, duration_score = self._calc_duration_entropy(midi)
            
            # 総合スコア（重み付き平均）
            overall_score = (
                note_density_score * 0.25 +
                pitch_range_score * 0.20 +
                harmonic_ratio_score * 0.25 +
                velocity_score * 0.15 +
                duration_score * 0.15
            )
            
            # グレード判定
            quality_grade = self._score_to_grade(overall_score)
            
            return {
                'note_density': note_density,
                'note_density_score': note_density_score,
                'pitch_range': pitch_range,
                'pitch_range_score': pitch_range_score,
                'harmonic_ratio': harmonic_ratio,
                'harmonic_ratio_score': harmonic_ratio_score,
                'velocity_variance': velocity_var,
                'velocity_score': velocity_score,
                'duration_entropy': duration_entropy,
                'duration_score': duration_score,
                'overall_score': overall_score,
                'quality_grade': quality_grade
            }
        
        except Exception as e:
            print(f"❌ Failed to analyze {midi_path}: {e}")
            return self._empty_metrics()
    
    def _calc_note_density(
        self,
        midi: PrettyMIDI
    ) -> Tuple[float, float]:
        """
        ノート密度計算
        
        Returns:
            (density, score)
        """
        total_notes = sum(len(inst.notes) for inst in midi.instruments)
        duration = midi.get_end_time()
        
        if duration == 0:
            return 0.0, 0.0
        
        density = total_notes / duration
        
        # スコア計算
        min_d, max_d = self.note_density_range
        
        if density < min_d:
            # sparse: 線形減衰
            score = max(0.0, density / min_d)
        elif density > max_d:
            # dense: 線形減衰
            score = max(0.0, 1.0 - (density - max_d) / 5.0)
        else:
            # 適切な範囲
            score = 1.0
        
        return density, score
    
    def _calc_pitch_range(
        self,
        midi: PrettyMIDI
    ) -> Tuple[int, float]:
        """
        ピッチ範囲計算
        
        Returns:
            (range_semitones, score)
        """
        all_pitches = []
        for inst in midi.instruments:
            if not inst.is_drum:
                all_pitches.extend([note.pitch for note in inst.notes])
        
        if not all_pitches:
            return 0, 0.0
        
        pitch_range = max(all_pitches) - min(all_pitches)
        
        # スコア計算
        min_r, max_r = self.pitch_range_range
        
        if pitch_range < min_r:
            # narrow: 線形減衰
            score = pitch_range / min_r
        elif pitch_range > max_r:
            # extreme: 線形減衰
            score = max(0.0, 1.0 - (pitch_range - max_r) / 24)
        else:
            # 適切な範囲
            score = 1.0
        
        return pitch_range, score
    
    def _calc_harmonic_ratio(
        self,
        midi: PrettyMIDI
    ) -> Tuple[float, float]:
        """
        和音率計算（同時発音比率）
        
        Returns:
            (harmonic_ratio, score)
        """
        # 時間軸を100msごとにサンプリング
        dt = 0.1  # 100ms
        duration = midi.get_end_time()
        
        if duration == 0:
            return 0.0, 0.0
        
        time_steps = int(duration / dt)
        polyphony_counts = []
        
        for t in np.linspace(0, duration, time_steps):
            active_notes = 0
            for inst in midi.instruments:
                if inst.is_drum:
                    continue
                for note in inst.notes:
                    if note.start <= t < note.end:
                        active_notes += 1
            polyphony_counts.append(active_notes)
        
        # 和音率（2音以上が同時発音）
        harmonic_frames = sum(1 for count in polyphony_counts if count >= 2)
        harmonic_ratio = harmonic_frames / max(1, len(polyphony_counts))
        
        # スコア計算
        min_h, max_h = self.harmonic_ratio_range
        
        if harmonic_ratio < min_h:
            # too monophonic
            score = harmonic_ratio / min_h
        elif harmonic_ratio > max_h:
            # too polyphonic
            score = max(0.0, 1.0 - (harmonic_ratio - max_h) / 0.3)
        else:
            # 適切な範囲
            score = 1.0
        
        return harmonic_ratio, score
    
    def _calc_velocity_variance(
        self,
        midi: PrettyMIDI
    ) -> Tuple[float, float]:
        """
        ベロシティ分散計算
        
        Returns:
            (velocity_std, score)
        """
        all_velocities = []
        for inst in midi.instruments:
            if not inst.is_drum:
                all_velocities.extend([note.velocity for note in inst.notes])
        
        if not all_velocities:
            return 0.0, 0.0
        
        velocity_std = np.std(all_velocities)
        
        # スコア計算
        score = min(1.0, velocity_std / self.velocity_std_min)
        
        return float(velocity_std), score
    
    def _calc_duration_entropy(
        self,
        midi: PrettyMIDI
    ) -> Tuple[float, float]:
        """
        音長分布のエントロピー計算（多様性指標）
        
        Returns:
            (entropy, score)
        """
        all_durations = []
        for inst in midi.instruments:
            if not inst.is_drum:
                durations = [note.end - note.start for note in inst.notes]
                all_durations.extend(durations)
        
        if not all_durations:
            return 0.0, 0.0
        
        # 音長をビンに分類（16分音符刻み想定）
        bins = [0.125 * i for i in range(1, 17)]  # 0.125s刻み
        hist, _ = np.histogram(all_durations, bins=bins)
        
        # エントロピー計算
        prob = hist / (hist.sum() + 1e-10)
        prob = prob[prob > 0]  # 0除外
        entropy = -np.sum(prob * np.log2(prob))
        
        # スコア計算
        score = min(1.0, entropy / self.duration_entropy_min)
        
        return float(entropy), score
    
    def _score_to_grade(self, score: float) -> str:
        """スコアをグレード変換"""
        if score >= 0.8:
            return 'A'
        elif score >= 0.6:
            return 'B'
        elif score >= 0.4:
            return 'C'
        elif score >= 0.2:
            return 'D'
        else:
            return 'F'
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """エラー時の空メトリクス"""
        return {
            'note_density': 0.0,
            'note_density_score': 0.0,
            'pitch_range': 0,
            'pitch_range_score': 0.0,
            'harmonic_ratio': 0.0,
            'harmonic_ratio_score': 0.0,
            'velocity_variance': 0.0,
            'velocity_score': 0.0,
            'duration_entropy': 0.0,
            'duration_score': 0.0,
            'overall_score': 0.0,
            'quality_grade': 'F'
        }


# ========== Quality Filter ==========

class MoisesDBQualityFilter:
    """MoisesDB品質フィルタ"""
    
    def __init__(
        self,
        threshold: float = 0.6,
        verbose: bool = False
    ):
        self.threshold = threshold
        self.verbose = verbose
        self.metrics_calculator = MIDIQualityMetrics()
    
    def evaluate_midi_file(
        self,
        midi_path: Path
    ) -> Dict[str, Any]:
        """
        単一MIDIファイルを評価
        
        Returns:
            {
                'file': str,
                'metrics': {...},
                'passed': bool
            }
        """
        metrics = self.metrics_calculator.calculate_all_metrics(midi_path)
        passed = metrics['overall_score'] >= self.threshold
        
        result = {
            'file': str(midi_path),
            'metrics': metrics,
            'passed': passed
        }
        
        if self.verbose:
            self._print_evaluation(midi_path.name, metrics, passed)
        
        return result
    
    def evaluate_batch(
        self,
        midi_dir: Path,
        max_files: int = -1
    ) -> Dict[str, Any]:
        """
        バッチ評価
        
        Returns:
            {
                'total': int,
                'passed': int,
                'failed': int,
                'results': List[Dict]
            }
        """
        midi_files = sorted(midi_dir.glob('*.mid'))
        
        if max_files > 0:
            midi_files = midi_files[:max_files]
        
        print(f"\n{'='*70}")
        print(f"Batch Quality Evaluation")
        print(f"{'='*70}")
        print(f"Total files: {len(midi_files)}")
        print(f"Threshold: {self.threshold}")
        print(f"{'='*70}")
        
        results = []
        passed_count = 0
        
        for midi_file in midi_files:
            result = self.evaluate_midi_file(midi_file)
            results.append(result)
            
            if result['passed']:
                passed_count += 1
        
        summary = {
            'total': len(midi_files),
            'passed': passed_count,
            'failed': len(midi_files) - passed_count,
            'pass_rate': passed_count / max(1, len(midi_files)),
            'results': results
        }
        
        return summary
    
    def filter_database(
        self,
        db_path: Path,
        midi_dir: Path
    ) -> Dict[str, Any]:
        """
        データベースの低品質データをフィルタリング
        
        Returns:
            {
                'total': int,
                'kept': int,
                'removed': int
            }
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # moisesdb_meta から全曲取得
        cursor.execute("""
            SELECT song_id, midi_path FROM moisesdb_meta
            WHERE midi_path IS NOT NULL
        """)
        
        rows = cursor.fetchall()
        
        print(f"\n{'='*70}")
        print(f"Database Quality Filtering")
        print(f"{'='*70}")
        print(f"Database: {db_path}")
        print(f"Threshold: {self.threshold}")
        print(f"Total songs: {len(rows)}")
        print(f"{'='*70}")
        
        kept_count = 0
        removed_count = 0
        
        # quality_scores テーブル追加
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_scores (
                song_id TEXT PRIMARY KEY,
                overall_score REAL,
                quality_grade TEXT,
                note_density REAL,
                pitch_range INTEGER,
                harmonic_ratio REAL,
                velocity_variance REAL,
                duration_entropy REAL,
                passed BOOLEAN
            )
        """)
        
        for song_id, midi_path in rows:
            midi_file = Path(midi_path)
            
            if not midi_file.exists():
                print(f"⚠️  MIDI not found: {song_id}")
                continue
            
            # 評価
            result = self.evaluate_midi_file(midi_file)
            metrics = result['metrics']
            passed = result['passed']
            
            # quality_scores に保存
            cursor.execute("""
                INSERT OR REPLACE INTO quality_scores
                (song_id, overall_score, quality_grade, note_density, pitch_range,
                 harmonic_ratio, velocity_variance, duration_entropy, passed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                song_id,
                metrics['overall_score'],
                metrics['quality_grade'],
                metrics['note_density'],
                metrics['pitch_range'],
                metrics['harmonic_ratio'],
                metrics['velocity_variance'],
                metrics['duration_entropy'],
                passed
            ))
            
            if passed:
                kept_count += 1
            else:
                removed_count += 1
                
                if self.verbose:
                    print(f"❌ Removed: {song_id} (score: {metrics['overall_score']:.3f})")
        
        conn.commit()
        conn.close()
        
        summary = {
            'total': len(rows),
            'kept': kept_count,
            'removed': removed_count,
            'pass_rate': kept_count / max(1, len(rows))
        }
        
        return summary
    
    def _print_evaluation(
        self,
        filename: str,
        metrics: Dict[str, Any],
        passed: bool
    ):
        """評価結果を表示"""
        status = "✅ PASS" if passed else "❌ FAIL"
        grade = metrics['quality_grade']
        score = metrics['overall_score']
        
        print(f"\n{status} [{grade}] {filename} (score: {score:.3f})")
        print(f"  Note density: {metrics['note_density']:.2f} notes/sec (score: {metrics['note_density_score']:.2f})")
        print(f"  Pitch range: {metrics['pitch_range']} semitones (score: {metrics['pitch_range_score']:.2f})")
        print(f"  Harmonic ratio: {metrics['harmonic_ratio']:.2%} (score: {metrics['harmonic_ratio_score']:.2f})")
        print(f"  Velocity std: {metrics['velocity_variance']:.1f} (score: {metrics['velocity_score']:.2f})")
        print(f"  Duration entropy: {metrics['duration_entropy']:.2f} (score: {metrics['duration_score']:.2f})")


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(
        description="MoisesDB MIDI品質フィルタ"
    )
    
    # モード選択
    parser.add_argument(
        '--midi-file',
        type=Path,
        help='単一MIDIファイル評価'
    )
    parser.add_argument(
        '--midi-dir',
        type=Path,
        help='MIDIディレクトリ（バッチ評価）'
    )
    parser.add_argument(
        '--db-path',
        type=Path,
        help='データベースパス（フィルタリング）'
    )
    
    # オプション
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.6,
        help='品質スコア閾値（0-1）'
    )
    parser.add_argument(
        '--output-csv',
        type=Path,
        help='結果CSV出力パス'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=-1,
        help='最大処理ファイル数'
    )
    parser.add_argument(
        '--filter-low-quality',
        action='store_true',
        help='低品質データをDBから除外'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='詳細ログ'
    )
    
    args = parser.parse_args()
    
    # Quality Filter インスタンス
    quality_filter = MoisesDBQualityFilter(
        threshold=args.threshold,
        verbose=args.verbose
    )
    
    # 単一ファイル評価
    if args.midi_file:
        result = quality_filter.evaluate_midi_file(args.midi_file)
        print(json.dumps(result['metrics'], indent=2))
        return
    
    # バッチ評価
    if args.midi_dir:
        summary = quality_filter.evaluate_batch(
            midi_dir=args.midi_dir,
            max_files=args.max_files
        )
        
        print(f"\n{'='*70}")
        print("Evaluation Summary")
        print(f"{'='*70}")
        print(f"Total: {summary['total']}")
        print(f"✅ Passed: {summary['passed']} ({summary['pass_rate']:.1%})")
        print(f"❌ Failed: {summary['failed']}")
        print(f"{'='*70}")
        
        # CSV出力
        if args.output_csv:
            import csv
            
            with open(args.output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'file', 'overall_score', 'quality_grade', 'passed',
                    'note_density', 'pitch_range', 'harmonic_ratio',
                    'velocity_variance', 'duration_entropy'
                ])
                writer.writeheader()
                
                for result in summary['results']:
                    row = {
                        'file': Path(result['file']).name,
                        'passed': result['passed'],
                        **result['metrics']
                    }
                    writer.writerow(row)
            
            print(f"📄 Results saved to {args.output_csv}")
        
        return
    
    # データベースフィルタリング
    if args.db_path and args.filter_low_quality:
        if not args.midi_dir:
            print("❌ --midi-dir required for database filtering")
            return
        
        summary = quality_filter.filter_database(
            db_path=args.db_path,
            midi_dir=args.midi_dir
        )
        
        print(f"\n{'='*70}")
        print("Filtering Summary")
        print(f"{'='*70}")
        print(f"Total: {summary['total']}")
        print(f"✅ Kept: {summary['kept']} ({summary['pass_rate']:.1%})")
        print(f"❌ Removed: {summary['removed']}")
        print(f"{'='*70}")
        
        return
    
    # モード未指定
    print("❌ Please specify one of: --midi-file, --midi-dir, or --db-path")
    parser.print_help()


if __name__ == '__main__':
    main()
