#!/usr/bin/env python3
"""
Stage2 Pattern Extraction - 高品質データからパターン抽出

Stage2データ（POP909/SLAKH）からパターンを抽出し、
Pattern Recommender用のデータベースを構築。

Features:
- MIDI → Note sequence extraction
- Technique推定（楽器別）
- Stage2メトリクス統合
- Pickle保存（再現性・高速ロード）

Usage:
    # 全楽器パターン抽出
    python scripts/extract_stage2_patterns.py --all
    
    # 特定楽器のみ
    python scripts/extract_stage2_patterns.py --instrument piano
    
    # Stage2スコアフィルタ
    python scripts/extract_stage2_patterns.py --all --min-score 0.5
"""

import argparse
import json
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging

import numpy as np

try:
    import pretty_midi as pm
except ImportError:
    pm = None
    print("⚠️ pretty_midi not installed. Run: pip install pretty_midi")

try:
    from music21 import converter, note, chord, stream
except ImportError:
    print("⚠️ music21 not installed. Run: pip install music21")
    converter = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class NoteEvent:
    """MIDI Note event"""
    pitch: int
    velocity: int
    start: float  # seconds
    end: float
    duration: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PatternMetadata:
    """パターンメタデータ"""
    file: str
    instrument: str
    technique: str
    tempo: float
    duration: float
    num_notes: int
    
    # Stage2メトリクス
    metrics: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    passed: bool = False
    
    # 統計情報
    pitch_range: Tuple[int, int] = (0, 127)
    avg_velocity: float = 64.0
    velocity_std: float = 0.0
    avg_ioi: float = 0.0  # Inter-onset interval
    
    # Chord progression（Piano/Chords用）
    chord_progression: Optional[List[str]] = None
    
    # Hash（重複検出用）
    content_hash: Optional[str] = None
    
    def to_dict(self) -> dict:
        d = asdict(self)
        # Tuple → List（JSON互換）
        d["pitch_range"] = list(d["pitch_range"])
        return d


@dataclass
class ExtractedPattern:
    """抽出されたパターン"""
    metadata: PatternMetadata
    notes: List[NoteEvent]
    
    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "notes": [n.to_dict() for n in self.notes],
        }


# =============================================================================
# Instrument-specific Technique Estimators
# =============================================================================

class TechniqueEstimator:
    """楽器別Technique推定"""
    
    @staticmethod
    def estimate_bass_technique(notes: List[NoteEvent]) -> str:
        """
        Bass technique推定
        
        Techniques:
        - walking: 安定したウォーキング（IOI > 0.4s）
        - pick: 短いスタッカート（duration ratio < 0.3）
        - slap: 高速＋高velocity変動（IOI < 0.3, vel_std > 15）
        - fingerstyle: デフォルト
        """
        if not notes:
            return "unknown"
        
        # Inter-onset intervals
        iois = [notes[i+1].start - notes[i].start for i in range(len(notes)-1)]
        avg_ioi = np.mean(iois) if iois else 1.0
        
        # Duration ratio
        durations = [n.duration for n in notes]
        avg_duration = np.mean(durations)
        duration_ratio = avg_duration / avg_ioi if avg_ioi > 0 else 0.5
        
        # Velocity variation
        velocities = [n.velocity for n in notes]
        vel_std = np.std(velocities) if len(velocities) > 1 else 0
        
        # Classification logic
        if avg_ioi < 0.3 and vel_std > 15:
            return "slap"
        elif avg_ioi > 0.4 and duration_ratio > 0.6:
            return "walking"
        elif duration_ratio < 0.3:
            return "pick"
        else:
            return "fingerstyle"
    
    @staticmethod
    def estimate_guitar_technique(notes: List[NoteEvent]) -> str:
        """
        Guitar technique推定
        
        Techniques:
        - strum: 多数の同時発音（simultaneous > 3）
        - arpeggio: 多数の音、ほぼ単音（notes > 20, simultaneous < 2）
        - power_chord: 少数の音、一部同時（notes < 10）
        - fingerpicking: デフォルト
        """
        if not notes:
            return "unknown"
        
        simultaneous = TechniqueEstimator._count_simultaneous_notes(notes)
        
        if simultaneous > 3:
            return "strum"
        elif len(notes) > 20 and simultaneous < 2:
            return "arpeggio"
        elif len(notes) < 10:
            return "power_chord"
        else:
            return "fingerpicking"
    
    @staticmethod
    def estimate_strings_technique(notes: List[NoteEvent]) -> str:
        """
        Strings technique推定
        
        Techniques:
        - legato: 高オーバーラップ or 長音符（overlap > 0.1 or duration > 1.0）
        - staccato: 短音符（duration < 0.2）
        - spiccato: 高velocity変動＋中間duration（vel_std > 15, 0.2 < dur < 0.6）
        - sustained: 非常に長い音（duration > 2.0）
        - mixed: その他
        """
        if not notes:
            return "unknown"
        
        # Note overlap
        overlaps = []
        for i in range(len(notes) - 1):
            if notes[i].end > notes[i+1].start:
                overlap = notes[i].end - notes[i+1].start
                overlaps.append(overlap)
        
        avg_overlap = np.mean(overlaps) if overlaps else 0
        
        # Duration
        durations = [n.duration for n in notes]
        avg_duration = np.mean(durations)
        
        # Velocity variation
        velocities = [n.velocity for n in notes]
        vel_std = np.std(velocities) if len(velocities) > 1 else 0
        
        # Classification
        if avg_overlap > 0.1 or avg_duration > 1.0:
            return "legato"
        elif avg_duration < 0.2:
            return "staccato"
        elif vel_std > 15 and 0.2 < avg_duration < 0.6:
            return "spiccato"
        elif avg_duration > 2.0:
            return "sustained"
        else:
            return "mixed"
    
    @staticmethod
    def estimate_piano_technique(notes: List[NoteEvent], instrument_name: str) -> str:
        """
        Piano technique推定
        
        Based on instrument_name (melody/chords):
        - melody: single-note dominant
        - chords: block_chords (simultaneous > 3)
        - chords: arpeggio (notes > 30, sequential)
        - chords: syncopated_chords (default)
        """
        if not notes:
            return "unknown"
        
        simultaneous = TechniqueEstimator._count_simultaneous_notes(notes)
        
        if instrument_name == "melody":
            return "melody"
        else:
            # Chords
            if simultaneous > 3:
                return "block_chords"
            elif len(notes) > 30:
                return "arpeggio"
            else:
                return "syncopated_chords"
    
    @staticmethod
    def _count_simultaneous_notes(notes: List[NoteEvent]) -> int:
        """最大同時発音数カウント"""
        if not notes:
            return 0
        
        max_simultaneous = 0
        for note in notes:
            # Count overlapping notes
            overlapping = sum(
                1 for n in notes
                if n.start < note.end and n.end > note.start
            )
            max_simultaneous = max(max_simultaneous, overlapping)
        
        return max_simultaneous


# =============================================================================
# Pattern Extractor
# =============================================================================

class Stage2PatternExtractor:
    """Stage2データからパターン抽出"""
    
    # Instrument → Dataset mapping
    INSTRUMENT_DATASET_MAP = {
        "melody": "pop909",
        "chords": "pop909",
        "bass": "slakh",
        "guitar": "slakh",
        "strings": "slakh",
        "drums": "slakh",
    }
    
    # Instrument → Directory mapping (直接clean配下)
    INSTRUMENT_DIR_MAP = {
        "melody": "melody",
        "chords": "chords",
        "bass": "bass",
        "guitar": "guitar",
        "strings": "strings",
        "drums": "drums",
    }
    
    def __init__(self, instrument: str, base_dir: Path = None):
        self.instrument = instrument
        self.base_dir = base_dir or Path(__file__).parent.parent
        
        # Paths
        self.dataset = self.INSTRUMENT_DATASET_MAP.get(instrument, "slakh")
        self.stage2_dir = self.base_dir / "output" / self.dataset / "clean" / self.INSTRUMENT_DIR_MAP.get(instrument, instrument)
        
        # Stage2 results
        self.stage2_results = self._load_stage2_results()
        
        logger.info(f"Initialized extractor for {instrument}")
        logger.info(f"  Stage2 dir: {self.stage2_dir}")
        logger.info(f"  Stage2 results: {len(self.stage2_results)} files")
    
    def _load_stage2_results(self) -> Dict[str, dict]:
        """Stage2スコアリング結果を読み込み"""
        # Try multiple possible filenames
        possible_names = [
            f"{self.instrument}_full.json",
            f"piano_{self.instrument}_full.json",  # For melody/chords
        ]
        
        results_file = None
        for name in possible_names:
            candidate = self.base_dir / "output" / "test_results" / name
            if candidate.exists():
                results_file = candidate
                break
        
        if not results_file:
            logger.warning(f"Stage2 results not found for {self.instrument}")
            logger.debug(f"  Tried: {possible_names}")
            return {}
        
        try:
            with open(results_file, "r") as f:
                data = json.load(f)
            
            # File名 → Metricsのマッピング
            results = {}
            
            # Handle both list format and dict with "results" key
            if isinstance(data, dict) and "results" in data:
                entries = data["results"]
            elif isinstance(data, list):
                entries = data
            else:
                logger.warning(f"Unknown JSON format in {results_file.name}")
                return {}
            
            for entry in entries:
                file_name = Path(entry.get("file", "")).name
                
                # Convert to expected format
                results[file_name] = {
                    "file": file_name,
                    "metrics": entry.get("scores", {}),
                    "score": entry.get("total_score", 0.0),
                    "passed": entry.get("total_score", 0.0) >= 0.4,  # Default threshold
                }
            
            logger.info(f"Loaded {len(results)} Stage2 results from {results_file.name}")
            return results
        
        except Exception as e:
            logger.error(f"Failed to load Stage2 results: {e}")
            return {}
    
    def extract_patterns(
        self,
        min_score: float = 0.0,
        max_files: Optional[int] = None,
    ) -> List[ExtractedPattern]:
        """
        全MIDIファイルからパターン抽出
        
        Args:
            min_score: 最小Stage2スコア（フィルタリング）
            max_files: 最大ファイル数（デバッグ用）
        
        Returns:
            抽出されたパターンのリスト
        """
        if not self.stage2_dir.exists():
            logger.error(f"Stage2 directory not found: {self.stage2_dir}")
            return []
        
        # MIDI files
        midi_files = sorted(self.stage2_dir.glob("*.mid"))
        
        if max_files:
            midi_files = midi_files[:max_files]
        
        logger.info(f"Extracting patterns from {len(midi_files)} files...")
        
        patterns = []
        for i, midi_file in enumerate(midi_files):
            if (i + 1) % 50 == 0:
                logger.info(f"  Progress: {i+1}/{len(midi_files)}")
            
            try:
                pattern = self._extract_single_pattern(midi_file)
                
                # Score filter
                if pattern and pattern.metadata.score >= min_score:
                    patterns.append(pattern)
            
            except Exception as e:
                logger.debug(f"Error processing {midi_file.name}: {e}")
                continue
        
        logger.info(f"Extracted {len(patterns)} patterns")
        
        # Statistics
        if patterns:
            scores = [p.metadata.score for p in patterns]
            logger.info(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")
            logger.info(f"  Score mean: {np.mean(scores):.3f}")
        
        return patterns
    
    def _extract_single_pattern(self, midi_file: Path) -> Optional[ExtractedPattern]:
        """単一MIDIファイルからパターン抽出"""
        # Load MIDI
        try:
            midi = pm.PrettyMIDI(str(midi_file))
        except Exception as e:
            logger.debug(f"Failed to load {midi_file.name}: {e}")
            return None
        
        # Extract notes
        notes = self._extract_notes(midi)
        
        if not notes:
            return None
        
        # Stage2 metrics
        stage2_data = self.stage2_results.get(midi_file.name, {})
        metrics = stage2_data.get("metrics", {})
        score = stage2_data.get("score", 0.0)
        passed = stage2_data.get("passed", False)
        
        # Technique estimation
        technique = self._estimate_technique(notes)
        
        # Statistics
        velocities = [n.velocity for n in notes]
        pitches = [n.pitch for n in notes]
        iois = [notes[i+1].start - notes[i].start for i in range(len(notes)-1)]
        
        # Metadata
        metadata = PatternMetadata(
            file=midi_file.name,
            instrument=self.instrument,
            technique=technique,
            tempo=midi.estimate_tempo(),
            duration=midi.get_end_time(),
            num_notes=len(notes),
            metrics=metrics,
            score=score,
            passed=passed,
            pitch_range=(min(pitches), max(pitches)) if pitches else (0, 127),
            avg_velocity=np.mean(velocities) if velocities else 64.0,
            velocity_std=np.std(velocities) if len(velocities) > 1 else 0.0,
            avg_ioi=np.mean(iois) if iois else 0.0,
            chord_progression=None,  # TODO: Implement chord extraction
            content_hash=self._compute_hash(notes),
        )
        
        return ExtractedPattern(metadata=metadata, notes=notes)
    
    def _extract_notes(self, midi: pm.PrettyMIDI) -> List[NoteEvent]:
        """MIDI → Note events"""
        notes = []
        
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            
            for note in inst.notes:
                notes.append(NoteEvent(
                    pitch=note.pitch,
                    velocity=note.velocity,
                    start=note.start,
                    end=note.end,
                    duration=note.end - note.start,
                ))
        
        # Sort by start time
        notes.sort(key=lambda n: n.start)
        
        return notes
    
    def _estimate_technique(self, notes: List[NoteEvent]) -> str:
        """楽器別Technique推定"""
        if self.instrument == "bass":
            return TechniqueEstimator.estimate_bass_technique(notes)
        elif self.instrument == "guitar":
            return TechniqueEstimator.estimate_guitar_technique(notes)
        elif self.instrument == "strings":
            return TechniqueEstimator.estimate_strings_technique(notes)
        elif self.instrument in ["melody", "chords"]:
            return TechniqueEstimator.estimate_piano_technique(notes, self.instrument)
        else:
            return "default"
    
    def _compute_hash(self, notes: List[NoteEvent]) -> str:
        """Note sequence hash（重複検出用）"""
        import hashlib
        
        # Simplified hash: pitches + start times
        data = "_".join([f"{n.pitch}@{n.start:.3f}" for n in notes[:100]])  # First 100 notes
        return hashlib.md5(data.encode()).hexdigest()
    
    def save_patterns(
        self,
        patterns: List[ExtractedPattern],
        output_path: Path,
        format: str = "pickle",
    ):
        """パターンを保存"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "pickle":
            with open(output_path, "wb") as f:
                pickle.dump(patterns, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved {len(patterns)} patterns to {output_path} (pickle)")
        
        elif format == "json":
            data = [p.to_dict() for p in patterns]
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(patterns)} patterns to {output_path} (JSON)")
        
        else:
            raise ValueError(f"Unknown format: {format}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract Stage2 patterns for Pattern Recommender")
    parser.add_argument(
        "--instrument",
        choices=["melody", "chords", "bass", "guitar", "strings", "drums"],
        help="Extract patterns for specific instrument",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract patterns for all instruments",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum Stage2 score (0.0-1.0)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum files to process (for testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/patterns"),
        help="Output directory",
    )
    parser.add_argument(
        "--format",
        choices=["pickle", "json"],
        default="pickle",
        help="Output format",
    )
    
    args = parser.parse_args()
    
    # Instruments to process
    if args.all:
        instruments = ["melody", "chords", "bass", "guitar", "strings"]
    elif args.instrument:
        instruments = [args.instrument]
    else:
        parser.error("Specify --instrument or --all")
    
    # Extract patterns
    for instrument in instruments:
        logger.info(f"\n{'='*60}")
        logger.info(f"Extracting patterns: {instrument}")
        logger.info('='*60)
        
        extractor = Stage2PatternExtractor(instrument)
        patterns = extractor.extract_patterns(
            min_score=args.min_score,
            max_files=args.max_files,
        )
        
        if not patterns:
            logger.warning(f"No patterns extracted for {instrument}")
            continue
        
        # Filter: Stage2合格のみ（optional）
        passed_patterns = [p for p in patterns if p.metadata.passed]
        
        logger.info(f"\nTotal patterns: {len(patterns)}")
        logger.info(f"Passed patterns: {len(passed_patterns)} ({len(passed_patterns)/len(patterns)*100:.1f}%)")
        
        # Save (use passed patterns only for high quality)
        output_path = args.output_dir / f"stage2_{instrument}.{args.format}"
        extractor.save_patterns(passed_patterns, output_path, format=args.format)
        
        # Summary
        if passed_patterns:
            techniques = {}
            for p in passed_patterns:
                tech = p.metadata.technique
                techniques[tech] = techniques.get(tech, 0) + 1
            
            logger.info(f"\nTechnique distribution:")
            for tech, count in sorted(techniques.items(), key=lambda x: -x[1]):
                logger.info(f"  {tech}: {count} ({count/len(passed_patterns)*100:.1f}%)")


if __name__ == "__main__":
    main()
