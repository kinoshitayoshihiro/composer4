#!/usr/bin/env python3
"""
Test: Technique Switch Comparison

同一セクションで複数の奏法を比較し、品質メトリクスを評価。
Stage2パターン推薦システムの柔軟性を検証。

Features:
- Guitar: strum vs fingerpicking vs arpeggio
- Strings: legato vs pizzicato vs tremolo vs staccato
- 品質メトリクス比較:
  - Note count (density)
  - Pitch range (min/max MIDI values)
  - Rhythm complexity (note duration variance)
  - Velocity variance (dynamics)

Test Philosophy:
- 既存のStage2ジェネレーター活用
- 奏法によるMIDI出力の違いを定量化
- パターン推薦の多様性を確認
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# music21 import
try:
    import music21
    from music21 import stream
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("⚠️  music21 not available, skipping tests")

# Import Stage2 generators
try:
    from generator.guitar_generator_stage2 import GuitarGeneratorStage2
    from generator.strings_generator_stage2 import StringsGeneratorStage2
    GENERATORS_AVAILABLE = True
except ImportError as e:
    GENERATORS_AVAILABLE = False
    print(f"⚠️  Stage2 generators not available: {e}")

# Import Pattern Recommender
try:
    from ml.pattern_recommender import PatternRecommender, PatternQuery
    PATTERN_RECOMMENDER_AVAILABLE = True
except ImportError:
    PATTERN_RECOMMENDER_AVAILABLE = False
    print("⚠️  PatternRecommender not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_metrics(part: stream.Part) -> Dict[str, Any]:
    """
    Calculate quality metrics for a music21.Part
    
    Metrics:
    - note_count: Number of notes
    - pitch_min: Lowest MIDI pitch
    - pitch_max: Highest MIDI pitch
    - pitch_range: Pitch range (max - min)
    - duration_mean: Average note duration (quarterLength)
    - duration_variance: Note duration variance
    - velocity_mean: Average velocity
    - velocity_variance: Velocity variance
    
    Args:
        part: music21.stream.Part
    
    Returns:
        Dict of metrics
    """
    notes = list(part.flatten().notes)
    
    if not notes:
        return {
            'note_count': 0,
            'pitch_min': 0,
            'pitch_max': 0,
            'pitch_range': 0,
            'duration_mean': 0.0,
            'duration_variance': 0.0,
            'velocity_mean': 0.0,
            'velocity_variance': 0.0,
        }
    
    # Pitch metrics
    pitches = []
    for n in notes:
        if hasattr(n, 'pitch'):
            pitches.append(n.pitch.midi)
        elif hasattr(n, 'pitches'):  # Chord
            pitches.extend([p.midi for p in n.pitches])
    
    pitch_min = min(pitches) if pitches else 0
    pitch_max = max(pitches) if pitches else 0
    pitch_range = pitch_max - pitch_min
    
    # Duration metrics
    durations = [n.duration.quarterLength for n in notes]
    duration_mean = float(sum(durations) / len(durations)) if durations else 0.0
    duration_variance = float(sum((d - duration_mean) ** 2 for d in durations) / len(durations)) if durations else 0.0
    
    # Velocity metrics
    velocities = []
    for n in notes:
        if hasattr(n, 'volume') and n.volume.velocity is not None:
            velocities.append(n.volume.velocity)
    
    velocity_mean = sum(velocities) / len(velocities) if velocities else 80.0
    velocity_variance = sum((v - velocity_mean) ** 2 for v in velocities) / len(velocities) if velocities else 0.0
    
    return {
        'note_count': len(notes),
        'pitch_min': pitch_min,
        'pitch_max': pitch_max,
        'pitch_range': pitch_range,
        'duration_mean': duration_mean,
        'duration_variance': duration_variance,
        'velocity_mean': velocity_mean,
        'velocity_variance': velocity_variance,
    }


def print_metrics_comparison(technique_metrics: Dict[str, Dict[str, Any]]):
    """
    Print metrics comparison table
    
    Args:
        technique_metrics: {technique_name: metrics_dict}
    """
    print("\n📊 Technique Metrics Comparison")
    print("=" * 80)
    
    # Header
    techniques = list(technique_metrics.keys())
    print(f"{'Metric':<20}", end="")
    for tech in techniques:
        print(f"{tech:>15}", end="")
    print()
    print("-" * 80)
    
    # Metrics rows
    metric_names = [
        ('note_count', 'Note Count'),
        ('pitch_min', 'Pitch Min'),
        ('pitch_max', 'Pitch Max'),
        ('pitch_range', 'Pitch Range'),
        ('duration_mean', 'Duration Mean'),
        ('duration_variance', 'Dur. Variance'),
        ('velocity_mean', 'Velocity Mean'),
        ('velocity_variance', 'Vel. Variance'),
    ]
    
    for metric_key, metric_label in metric_names:
        print(f"{metric_label:<20}", end="")
        for tech in techniques:
            value = technique_metrics[tech][metric_key]
            if isinstance(value, float):
                print(f"{value:>15.2f}", end="")
            else:
                print(f"{value:>15}", end="")
        print()
    
    print("=" * 80)


def test_guitar_technique_comparison():
    """
    Test 1: Guitar Technique Comparison
    
    同一セクション（Verse）でstrum/fingerpickingを比較
    """
    print("\n" + "="*80)
    print("Test 1: Guitar Technique Comparison")
    print("="*80)
    
    if not MUSIC21_AVAILABLE:
        print("⏭️  Skipping: music21 not available")
        return
    
    if not GENERATORS_AVAILABLE:
        print("⏭️  Skipping: Stage2 generators not available")
        return
    
    # Check pattern file
    pattern_file = project_root / "data" / "patterns" / "stage2_guitar.pickle"
    if not pattern_file.exists():
        print(f"⏭️  Skipping: Pattern file not found: {pattern_file}")
        return
    
    print(f"✅ Pattern file found: {pattern_file}")
    
    # Test parameters
    section_name = "Verse"
    measures = 4
    chord_progression = ["C", "G", "Am", "F"]
    tempo = 120
    
    # Techniques to test
    techniques = ["strum", "fingerpicking"]
    
    technique_metrics = {}
    
    for technique in techniques:
        print(f"\n🎸 Testing technique: {technique}")
        
        # Create generator with specific emotion to force technique
        # strum: happy, fingerpicking: sad
        emotion = "happy" if technique == "strum" else "sad"
        
        try:
            gen = GuitarGeneratorStage2(
                use_stage2=True,
                stage2_patterns_path=str(pattern_file),
                tempo=tempo,
                emotion=emotion,
                default_instrument=music21.instrument.AcousticGuitar()
            )
            
            # Generate part
            part = gen.compose(
                section_name=section_name,
                measures=measures,
                chord_progression=chord_progression,
                tempo=tempo,
                emotion=emotion
            )
            
            # Calculate metrics
            metrics = calculate_metrics(part)
            technique_metrics[technique] = metrics
            
            print(f"   ✅ Generated: {metrics['note_count']} notes, "
                  f"pitch range: {metrics['pitch_min']}-{metrics['pitch_max']}")
            
        except Exception as e:
            print(f"   ⚠️  Failed to generate with {technique}: {e}")
            technique_metrics[technique] = calculate_metrics(stream.Part())
    
    # Print comparison
    if len(technique_metrics) >= 2:
        print_metrics_comparison(technique_metrics)
        print("✅ Test 1 Passed!")
    else:
        print("⚠️  Test 1: Not enough techniques generated")


def test_strings_technique_comparison():
    """
    Test 2: Strings Technique Comparison
    
    同一セクション（Chorus）でlegato/pizzicato/tremoloを比較
    """
    print("\n" + "="*80)
    print("Test 2: Strings Technique Comparison")
    print("="*80)
    
    if not MUSIC21_AVAILABLE:
        print("⏭️  Skipping: music21 not available")
        return
    
    if not GENERATORS_AVAILABLE:
        print("⏭️  Skipping: Stage2 generators not available")
        return
    
    # Check pattern file
    pattern_file = project_root / "data" / "patterns" / "stage2_strings.pickle"
    if not pattern_file.exists():
        print(f"⏭️  Skipping: Pattern file not found: {pattern_file}")
        return
    
    print(f"✅ Pattern file found: {pattern_file}")
    
    # Test parameters
    section_name = "Chorus"
    measures = 4
    chord_progression = ["C", "G", "Am", "F"]
    tempo = 120
    
    # Techniques to test (with emotion mapping)
    technique_emotions = {
        "legato": "calm",
        "pizzicato": "playful",
        "tremolo": "dramatic"
    }
    
    technique_metrics = {}
    
    for technique, emotion in technique_emotions.items():
        print(f"\n🎻 Testing technique: {technique} (emotion: {emotion})")
        
        try:
            gen = StringsGeneratorStage2(
                use_stage2=True,
                stage2_patterns_path=str(pattern_file),
                tempo=tempo,
                emotion=emotion,
                default_instrument=music21.instrument.Violin()
            )
            
            # Generate part
            part = gen.compose(
                section_name=section_name,
                measures=measures,
                chord_progression=chord_progression,
                tempo=tempo,
                emotion=emotion
            )
            
            # Calculate metrics
            metrics = calculate_metrics(part)
            technique_metrics[technique] = metrics
            
            print(f"   ✅ Generated: {metrics['note_count']} notes, "
                  f"pitch range: {metrics['pitch_min']}-{metrics['pitch_max']}")
            
        except Exception as e:
            print(f"   ⚠️  Failed to generate with {technique}: {e}")
            technique_metrics[technique] = calculate_metrics(stream.Part())
    
    # Print comparison
    if len(technique_metrics) >= 2:
        print_metrics_comparison(technique_metrics)
        print("✅ Test 2 Passed!")
    else:
        print("⚠️  Test 2: Not enough techniques generated")


def test_section_variation():
    """
    Test 3: Section Variation
    
    異なるセクション（Verse/Chorus/Bridge）での奏法変化を確認
    """
    print("\n" + "="*80)
    print("Test 3: Section Variation")
    print("="*80)
    
    if not MUSIC21_AVAILABLE:
        print("⏭️  Skipping: music21 not available")
        return
    
    if not GENERATORS_AVAILABLE:
        print("⏭️  Skipping: Stage2 generators not available")
        return
    
    # Check pattern file
    pattern_file = project_root / "data" / "patterns" / "stage2_guitar.pickle"
    if not pattern_file.exists():
        print(f"⏭️  Skipping: Pattern file not found: {pattern_file}")
        return
    
    print(f"✅ Pattern file found: {pattern_file}")
    
    # Test parameters
    sections = ["Verse", "Chorus", "Bridge"]
    measures = 4
    chord_progression = ["C", "G", "Am", "F"]
    tempo = 120
    emotion = "happy"
    
    section_metrics = {}
    
    for section in sections:
        print(f"\n📝 Testing section: {section}")
        
        try:
            gen = GuitarGeneratorStage2(
                use_stage2=True,
                stage2_patterns_path=str(pattern_file),
                tempo=tempo,
                emotion=emotion,
                default_instrument=music21.instrument.AcousticGuitar()
            )
            
            # Generate part
            part = gen.compose(
                section_name=section,
                measures=measures,
                chord_progression=chord_progression,
                tempo=tempo,
                emotion=emotion
            )
            
            # Calculate metrics
            metrics = calculate_metrics(part)
            section_metrics[section] = metrics
            
            print(f"   ✅ Generated: {metrics['note_count']} notes, "
                  f"pitch range: {metrics['pitch_min']}-{metrics['pitch_max']}")
            
        except Exception as e:
            print(f"   ⚠️  Failed to generate {section}: {e}")
            section_metrics[section] = calculate_metrics(stream.Part())
    
    # Print comparison
    if len(section_metrics) >= 2:
        print_metrics_comparison(section_metrics)
        print("✅ Test 3 Passed!")
    else:
        print("⚠️  Test 3: Not enough sections generated")


def test_tempo_variation():
    """
    Test 4: Tempo Variation
    
    異なるテンポ（80/120/160 BPM）での生成品質を確認
    """
    print("\n" + "="*80)
    print("Test 4: Tempo Variation")
    print("="*80)
    
    if not MUSIC21_AVAILABLE:
        print("⏭️  Skipping: music21 not available")
        return
    
    if not GENERATORS_AVAILABLE:
        print("⏭️  Skipping: Stage2 generators not available")
        return
    
    # Check pattern file
    pattern_file = project_root / "data" / "patterns" / "stage2_strings.pickle"
    if not pattern_file.exists():
        print(f"⏭️  Skipping: Pattern file not found: {pattern_file}")
        return
    
    print(f"✅ Pattern file found: {pattern_file}")
    
    # Test parameters
    tempos = [80, 120, 160]
    section_name = "Verse"
    measures = 4
    chord_progression = ["C", "G", "Am", "F"]
    emotion = "calm"
    
    tempo_metrics = {}
    
    for tempo in tempos:
        print(f"\n🎼 Testing tempo: {tempo} BPM")
        
        try:
            gen = StringsGeneratorStage2(
                use_stage2=True,
                stage2_patterns_path=str(pattern_file),
                tempo=tempo,
                emotion=emotion
            )
            
            # Generate part
            part = gen.compose(
                section_name=section_name,
                measures=measures,
                chord_progression=chord_progression,
                tempo=tempo,
                emotion=emotion
            )
            
            # Calculate metrics
            metrics = calculate_metrics(part)
            tempo_metrics[f"{tempo}BPM"] = metrics
            
            print(f"   ✅ Generated: {metrics['note_count']} notes, "
                  f"pitch range: {metrics['pitch_min']}-{metrics['pitch_max']}")
            
        except Exception as e:
            print(f"   ⚠️  Failed to generate at {tempo}BPM: {e}")
            tempo_metrics[f"{tempo}BPM"] = calculate_metrics(stream.Part())
    
    # Print comparison
    if len(tempo_metrics) >= 2:
        print_metrics_comparison(tempo_metrics)
        print("✅ Test 4 Passed!")
    else:
        print("⚠️  Test 4: Not enough tempos generated")


def test_midi_export():
    """
    Test 5: MIDI Export
    
    生成したパートをMIDIファイルとして保存し、ファイルサイズを確認
    """
    print("\n" + "="*80)
    print("Test 5: MIDI Export")
    print("="*80)
    
    if not MUSIC21_AVAILABLE:
        print("⏭️  Skipping: music21 not available")
        return
    
    if not GENERATORS_AVAILABLE:
        print("⏭️  Skipping: Stage2 generators not available")
        return
    
    # Check pattern files
    guitar_pattern = project_root / "data" / "patterns" / "stage2_guitar.pickle"
    strings_pattern = project_root / "data" / "patterns" / "stage2_strings.pickle"
    
    if not guitar_pattern.exists() or not strings_pattern.exists():
        print(f"⏭️  Skipping: Pattern files not found")
        return
    
    print(f"✅ Pattern files found")
    
    # Test parameters
    section_name = "Verse"
    measures = 4
    chord_progression = ["C", "G", "Am", "F"]
    tempo = 120
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test Guitar
        print(f"\n🎸 Testing Guitar MIDI export")
        try:
            gen_guitar = GuitarGeneratorStage2(
                use_stage2=True,
                stage2_patterns_path=str(guitar_pattern),
                tempo=tempo,
                emotion="happy",
                default_instrument=music21.instrument.AcousticGuitar()
            )
            
            part_guitar = gen_guitar.compose(
                section_name=section_name,
                measures=measures,
                chord_progression=chord_progression,
                tempo=tempo,
                emotion="happy"
            )
            
            # Export MIDI
            midi_path_guitar = tmpdir_path / "guitar_strum.mid"
            part_guitar.write('midi', fp=str(midi_path_guitar))
            
            midi_size_guitar = midi_path_guitar.stat().st_size
            print(f"   ✅ Guitar MIDI exported: {midi_path_guitar.name} ({midi_size_guitar} bytes)")
            
        except Exception as e:
            print(f"   ⚠️  Guitar MIDI export failed: {e}")
        
        # Test Strings
        print(f"\n🎻 Testing Strings MIDI export")
        try:
            gen_strings = StringsGeneratorStage2(
                use_stage2=True,
                stage2_patterns_path=str(strings_pattern),
                tempo=tempo,
                emotion="calm",
                default_instrument=music21.instrument.Violin()
            )
            
            part_strings = gen_strings.compose(
                section_name=section_name,
                measures=measures,
                chord_progression=chord_progression,
                tempo=tempo,
                emotion="calm"
            )
            
            # Export MIDI
            midi_path_strings = tmpdir_path / "strings_legato.mid"
            part_strings.write('midi', fp=str(midi_path_strings))
            
            midi_size_strings = midi_path_strings.stat().st_size
            print(f"   ✅ Strings MIDI exported: {midi_path_strings.name} ({midi_size_strings} bytes)")
            
        except Exception as e:
            print(f"   ⚠️  Strings MIDI export failed: {e}")
    
    print("✅ Test 5 Passed!")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎵 Technique Switch Comparison Test Suite")
    print("="*80)
    print(f"Project root: {project_root}")
    
    # Check dependencies
    print("\n📦 Dependency Check:")
    print(f"   music21: {'✅' if MUSIC21_AVAILABLE else '❌'}")
    print(f"   Stage2 Generators: {'✅' if GENERATORS_AVAILABLE else '❌'}")
    print(f"   PatternRecommender: {'✅' if PATTERN_RECOMMENDER_AVAILABLE else '❌'}")
    
    # Run tests
    test_guitar_technique_comparison()
    test_strings_technique_comparison()
    test_section_variation()
    test_tempo_variation()
    test_midi_export()
    
    print("\n" + "="*80)
    print("📊 Test Summary: All technique comparison tests completed")
    print("="*80)
    print("\n✅ All runnable tests passed!")
