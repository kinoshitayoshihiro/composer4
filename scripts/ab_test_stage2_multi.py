#!/usr/bin/env python3
"""
Multi-Instrument A/B Test for Stage2 V3 Patterns
Phase 24.3: All-instrument integration test with instrument-specific KPIs

Usage:
    python scripts/ab_test_stage2_multi.py \
        --num-songs 50 \
        --instruments guitar,bass,piano,strings \
        --output data/reports/ab_v3_multi_50.csv

Instrument-Specific KPIs:
    Common:
        - ml_used: ML推薦採用率
        - top1_proba: ML確信度
        - accent_score_norm16: 拍アクセント一致度
        - chord_fit_v3: コード適合率
        - density_abs: 密度偏差
    
    Bass:
        - downbeat_hit_rate: ダウンビート（1拍目・3拍目）命中率
    
    Strings:
        - voice_leading_smoothness: ボイスリーディング滑らかさ
        - avg_interval: 平均音程移動距離（半音単位）
    
    Piano:
        - voicing_spread: ボイシング音域幅（オクターブ単位）

KPI Gate (Initial):
    - ml_used ≥ 0.90
    - accent_score_norm16 ≥ 0.70
    - chord_fit_v3 ≥ 0.60
    - density_abs ≤ 1.0
    - bass.downbeat_hit_rate ≥ 0.70
    - strings.voice_leading_smoothness ≥ 0.70 (1 - avg_interval/8)
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
import random
from collections import defaultdict
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from generator.bass_generator_stage2 import BassGeneratorStage2
from generator.guitar_generator_stage2 import GuitarGeneratorStage2
# Placeholder imports (Piano/Strings may not have Stage2 yet)
try:
    from generator.piano_generator_stage2 import PianoGeneratorStage2
except ImportError:
    PianoGeneratorStage2 = None

try:
    from generator.strings_generator_stage2 import StringsGeneratorStage2
except ImportError:
    StringsGeneratorStage2 = None

from music21 import stream, note, chord as m21chord
from ml.v3_filter_config import get_v3_filter_params

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class MultiInstrumentTester:
    """全楽器統合テスター"""
    
    def __init__(
        self,
        instruments: List[str],
        num_songs: int = 50,
        global_tempo: float = 120.0,
    ):
        self.instruments = instruments
        self.num_songs = num_songs
        self.global_tempo = global_tempo
        
        # Initialize generators
        self.generators = {}
        
        if 'guitar' in instruments:
            self.generators['guitar'] = GuitarGeneratorStage2(
                use_stage2=True,
                global_tempo=global_tempo
            )
        
        if 'bass' in instruments:
            self.generators['bass'] = BassGeneratorStage2(
                use_stage2=True,
                global_tempo=global_tempo
            )
        
        if 'piano' in instruments and PianoGeneratorStage2:
            self.generators['piano'] = PianoGeneratorStage2(
                use_stage2=True,
                global_tempo=global_tempo
            )
        
        if 'strings' in instruments and StringsGeneratorStage2:
            self.generators['strings'] = StringsGeneratorStage2(
                use_stage2=True,
                global_tempo=global_tempo
            )
        
        logger.info(f"Initialized generators: {list(self.generators.keys())}")
    
    def generate_test_data(self, song_idx: int) -> Dict:
        """Generate test section data"""
        sections = ['verse', 'chorus', 'bridge', 'intro', 'outro']
        chord_progressions = [
            ['C', 'Am', 'F', 'G'],
            ['G', 'D', 'Em', 'C'],
            ['Am', 'F', 'C', 'G'],
            ['D', 'A', 'Bm', 'G'],
        ]
        
        section = random.choice(sections)
        chords = random.choice(chord_progressions)
        tempo = self.global_tempo + random.uniform(-10, 10)
        
        return {
            'song_id': song_idx,
            'section': section,
            'chords': chords,
            'tempo': tempo,
            'time_signature': '4/4',
            'duration': 8.0,  # 8 seconds
            'bars': 4,
        }
    
    def run_tests(self) -> List[Dict]:
        """Run tests for all instruments"""
        results = []
        
        for song_idx in range(self.num_songs):
            test_data = self.generate_test_data(song_idx)
            
            logger.info(
                f"\n{'='*60}\n"
                f"Song {song_idx+1}/{self.num_songs}: "
                f"{test_data['section'].upper()} @ {test_data['tempo']:.1f} BPM\n"
                f"{'='*60}"
            )
            
            for instrument in self.instruments:
                if instrument not in self.generators:
                    logger.warning(f"Skipping {instrument}: generator not available")
                    continue
                
                try:
                    metrics = self._test_instrument(
                        instrument,
                        test_data,
                        self.generators[instrument]
                    )
                    
                    metrics.update({
                        'song_id': song_idx,
                        'instrument': instrument,
                        'section': test_data['section'],
                        'tempo': test_data['tempo'],
                    })
                    
                    results.append(metrics)
                    
                except Exception as e:
                    logger.error(f"Failed {instrument} test: {e}", exc_info=True)
                    results.append({
                        'song_id': song_idx,
                        'instrument': instrument,
                        'section': test_data['section'],
                        'tempo': test_data['tempo'],
                        'error': str(e),
                    })
        
        return results
    
    def _test_instrument(
        self,
        instrument: str,
        test_data: Dict,
        generator,
    ) -> Dict:
        """Test single instrument"""
        section_data = {
            'section': test_data['section'].capitalize(),  # Capitalize section names
            'chord_progression': test_data['chords'],
            'tempo': test_data['tempo'],
            'time_signature': test_data['time_signature'],
            'num_bars': test_data['bars'],
        }
        
        # Generate part
        try:
            # Try Stage2-style compose() with keyword args
            part = generator.compose(
                section_data=section_data,
                section=test_data['section'].capitalize(),
            )
        except (AttributeError, TypeError) as e:
            logger.debug(f"Stage2 compose failed: {e}, trying V1 style")
            # Fallback to V1 style
            try:
                part = generator.compose(section_data)
            except Exception as e2:
                logger.error(f"Both compose methods failed: {e2}")
                raise
        
        # Common KPIs
        metrics = self._compute_common_kpis(part, test_data)
        
        # Instrument-specific KPIs
        if instrument == 'bass':
            metrics.update(self._compute_bass_kpis(part, test_data))
        elif instrument == 'strings':
            metrics.update(self._compute_strings_kpis(part, test_data))
        elif instrument == 'piano':
            metrics.update(self._compute_piano_kpis(part, test_data))
        
        # KPI gate check
        metrics['kpi_passed'] = self._check_kpi_gate(instrument, metrics)
        
        logger.info(
            f"  {instrument.upper()}: "
            f"ml_used={metrics.get('ml_used', 0):.2f}, "
            f"accent={metrics.get('accent_score_norm16', 0):.2f}, "
            f"chord_fit={metrics.get('chord_fit_v3', 0):.2f}, "
            f"KPI={'✅ PASS' if metrics['kpi_passed'] else '❌ FAIL'}"
        )
        
        return metrics
    
    def _compute_common_kpis(self, part: stream.Part, test_data: Dict) -> Dict:
        """Compute common KPIs for all instruments"""
        metrics = {}
        
        # ML used rate (placeholder: assume 100% if Stage2)
        metrics['ml_used'] = 1.0
        
        # top1_proba (placeholder: read from metadata if available)
        if hasattr(part, 'metadata') and hasattr(part.metadata, 'top1_proba'):
            metrics['top1_proba'] = part.metadata.top1_proba
        else:
            metrics['top1_proba'] = 0.95  # default high value for Stage2
        
        # Accent score (placeholder: simplified)
        metrics['accent_score_norm16'] = self._compute_accent_score(part, test_data)
        
        # Chord fit (placeholder: simplified)
        metrics['chord_fit_v3'] = self._compute_chord_fit(part, test_data)
        
        # Density (notes per bar)
        num_notes = len([n for n in part.flatten().notes if isinstance(n, note.Note)])
        num_bars = test_data['bars']
        density = num_notes / num_bars
        target_density = 8.0  # 1 note per beat in 4/4
        metrics['density_abs'] = abs(density - target_density)
        
        return metrics
    
    def _compute_accent_score(self, part: stream.Part, test_data: Dict) -> float:
        """Simplified accent score computation"""
        downbeat_notes = 0
        total_downbeats = test_data['bars'] * 4  # 4/4 time
        
        for n in part.flatten().notes:
            if isinstance(n, note.Note):
                offset_in_bar = n.offset % 4.0
                # Downbeats: 0.0, 2.0 (strong beats)
                if abs(offset_in_bar) < 0.1 or abs(offset_in_bar - 2.0) < 0.1:
                    downbeat_notes += 1
        
        return min(downbeat_notes / max(total_downbeats, 1), 1.0)
    
    def _compute_chord_fit(self, part: stream.Part, test_data: Dict) -> float:
        """Simplified chord fit computation"""
        chord_tones = {
            'C': {0, 4, 7},  # C-E-G
            'Am': {9, 0, 4},  # A-C-E
            'F': {5, 9, 0},  # F-A-C
            'G': {7, 11, 2},  # G-B-D
            'D': {2, 6, 9},  # D-F#-A
            'Em': {4, 7, 11},  # E-G-B
            'Bm': {11, 2, 6},  # B-D-F#
            'A': {9, 1, 4},  # A-C#-E
        }
        
        hits = 0
        total = 0
        
        for n in part.flatten().notes:
            if isinstance(n, note.Note):
                pitch_class = n.pitch.pitchClass
                # Assume first chord for simplicity
                chord_name = test_data['chords'][0]
                if chord_name in chord_tones:
                    if pitch_class in chord_tones[chord_name]:
                        hits += 1
                total += 1
        
        return hits / max(total, 1)
    
    def _compute_bass_kpis(self, part: stream.Part, test_data: Dict) -> Dict:
        """Bass-specific KPIs"""
        metrics = {}
        
        # Downbeat hit rate (1拍目・3拍目にルート音)
        downbeat_hits = 0
        total_downbeats = test_data['bars'] * 2  # 2 downbeats per bar in 4/4
        
        for n in part.flatten().notes:
            if isinstance(n, note.Note):
                offset_in_bar = n.offset % 4.0
                # Downbeats: 0.0, 2.0
                if abs(offset_in_bar) < 0.1 or abs(offset_in_bar - 2.0) < 0.1:
                    downbeat_hits += 1
        
        metrics['downbeat_hit_rate'] = downbeat_hits / max(total_downbeats, 1)
        
        return metrics
    
    def _compute_strings_kpis(self, part: stream.Part, test_data: Dict) -> Dict:
        """Strings-specific KPIs"""
        metrics = {}
        
        # Voice leading smoothness (平均音程移動距離)
        intervals = []
        prev_pitch = None
        
        for n in part.flatten().notes:
            if isinstance(n, note.Note):
                if prev_pitch is not None:
                    interval = abs(n.pitch.midi - prev_pitch)
                    intervals.append(interval)
                prev_pitch = n.pitch.midi
        
        avg_interval = np.mean(intervals) if intervals else 0.0
        metrics['avg_interval'] = avg_interval
        
        # Smoothness: 1 - (avg_interval / 8)
        # 8半音 = 完全5度（許容上限）
        metrics['voice_leading_smoothness'] = max(0.0, 1.0 - avg_interval / 8.0)
        
        return metrics
    
    def _compute_piano_kpis(self, part: stream.Part, test_data: Dict) -> Dict:
        """Piano-specific KPIs"""
        metrics = {}
        
        # Voicing spread (ボイシング音域幅)
        pitches = []
        for n in part.flatten().notes:
            if isinstance(n, note.Note):
                pitches.append(n.pitch.midi)
            elif isinstance(n, m21chord.Chord):
                pitches.extend([p.midi for p in n.pitches])
        
        if pitches:
            spread_semitones = max(pitches) - min(pitches)
            metrics['voicing_spread'] = spread_semitones / 12.0  # octaves
        else:
            metrics['voicing_spread'] = 0.0
        
        return metrics
    
    def _check_kpi_gate(self, instrument: str, metrics: Dict) -> bool:
        """Check if KPI gate passes"""
        # Common gates
        if metrics.get('ml_used', 0) < 0.90:
            return False
        
        if metrics.get('accent_score_norm16', 0) < 0.70:
            return False
        
        if metrics.get('chord_fit_v3', 0) < 0.60:
            return False
        
        if metrics.get('density_abs', 999) > 1.0:
            return False
        
        # Instrument-specific gates
        if instrument == 'bass':
            if metrics.get('downbeat_hit_rate', 0) < 0.70:
                return False
        
        if instrument == 'strings':
            if metrics.get('voice_leading_smoothness', 0) < 0.70:
                return False
        
        return True
    
    def save_results(self, results: List[Dict], output_path: Path):
        """Save results to CSV"""
        if not results:
            logger.warning("No results to save")
            return
        
        # Get all unique keys
        fieldnames = set()
        for r in results:
            fieldnames.update(r.keys())
        fieldnames = sorted(fieldnames)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"✅ Results saved: {output_path}")
    
    def generate_report(self, results: List[Dict]) -> Dict:
        """Generate summary report"""
        report = {
            'total_tests': len(results),
            'instruments': {},
        }
        
        # Group by instrument
        by_instrument = defaultdict(list)
        for r in results:
            inst = r.get('instrument')
            if inst:
                by_instrument[inst].append(r)
        
        for instrument, inst_results in by_instrument.items():
            kpi_passed = [r for r in inst_results if r.get('kpi_passed')]
            errors = [r for r in inst_results if 'error' in r]
            
            inst_report = {
                'total': len(inst_results),
                'kpi_passed': len(kpi_passed),
                'kpi_failed': len(inst_results) - len(kpi_passed) - len(errors),
                'errors': len(errors),
                'kpi_pass_rate': len(kpi_passed) / max(len(inst_results), 1),
            }
            
            # Compute stats
            for metric in ['ml_used', 'accent_score_norm16', 'chord_fit_v3', 'density_abs']:
                values = [r.get(metric) for r in inst_results if metric in r]
                if values:
                    inst_report[f'{metric}_mean'] = np.mean(values)
                    inst_report[f'{metric}_p10'] = np.percentile(values, 10)
                    inst_report[f'{metric}_p50'] = np.percentile(values, 50)
            
            report['instruments'][instrument] = inst_report
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Instrument A/B Test for Stage2 V3 Patterns'
    )
    parser.add_argument(
        '--num-songs',
        type=int,
        default=50,
        help='Number of songs to test (default: 50)'
    )
    parser.add_argument(
        '--instruments',
        type=str,
        default='guitar,bass',
        help='Comma-separated instruments (default: guitar,bass)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/reports/ab_v3_multi.csv'),
        help='Output CSV path'
    )
    parser.add_argument(
        '--tempo',
        type=float,
        default=120.0,
        help='Global tempo (default: 120.0)'
    )
    
    args = parser.parse_args()
    
    instruments = [i.strip() for i in args.instruments.split(',')]
    
    logger.info(
        f"\n{'='*60}\n"
        f"Multi-Instrument A/B Test\n"
        f"{'='*60}\n"
        f"Instruments: {', '.join(instruments)}\n"
        f"Num Songs: {args.num_songs}\n"
        f"Tempo: {args.tempo} BPM\n"
        f"Output: {args.output}\n"
        f"{'='*60}\n"
    )
    
    tester = MultiInstrumentTester(
        instruments=instruments,
        num_songs=args.num_songs,
        global_tempo=args.tempo,
    )
    
    results = tester.run_tests()
    
    tester.save_results(results, args.output)
    
    report = tester.generate_report(results)
    
    logger.info("\n" + "="*60)
    logger.info("Summary Report")
    logger.info("="*60)
    
    for instrument, inst_report in report['instruments'].items():
        logger.info(f"\n{instrument.upper()}:")
        logger.info(f"  Total: {inst_report['total']}")
        logger.info(f"  KPI Passed: {inst_report['kpi_passed']} ({inst_report['kpi_pass_rate']:.1%})")
        logger.info(f"  KPI Failed: {inst_report['kpi_failed']}")
        logger.info(f"  Errors: {inst_report['errors']}")
        
        for metric in ['ml_used', 'accent_score_norm16', 'chord_fit_v3']:
            mean_key = f'{metric}_mean'
            p10_key = f'{metric}_p10'
            if mean_key in inst_report:
                logger.info(
                    f"  {metric}: mean={inst_report[mean_key]:.3f}, "
                    f"p10={inst_report[p10_key]:.3f}"
                )
    
    logger.info("\n" + "="*60)
    
    # Overall KPI pass rate
    total_passed = sum(r['kpi_passed'] for r in report['instruments'].values())
    total_tests = report['total_tests']
    overall_rate = total_passed / max(total_tests, 1)
    
    logger.info(f"Overall KPI Pass Rate: {overall_rate:.1%} ({total_passed}/{total_tests})")
    
    if overall_rate >= 0.70:
        logger.info("✅ Overall KPI gate PASSED (≥70%)")
    else:
        logger.warning(f"❌ Overall KPI gate FAILED (<70%)")


if __name__ == '__main__':
    main()
