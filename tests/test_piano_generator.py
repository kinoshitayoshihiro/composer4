"""
Piano Generator Test

Piano Generatorの動作確認テスト
"""

from generators.piano import PianoGenerator
from generators.base import (
    Section,
    Chord,
    EmotionProfile,
    Emotion,
    GenerationContext,
)
import pretty_midi as pm


def test_piano_generator():
    """Piano Generator基本テスト"""
    print("="*50)
    print("Piano Generator Test")
    print("="*50)
    
    # Generator初期化
    piano_gen = PianoGenerator()
    print("\n✓ PianoGenerator initialized")
    
    # セクション定義
    section = Section(
        tempo=120.0,
        key="C",
        time_signature=(4, 4),
        chord_progression=[
            Chord(root=60, quality="major"),   # C major
            Chord(root=65, quality="minor"),   # F minor
            Chord(root=67, quality="major"),   # G major
            Chord(root=60, quality="major"),   # C major
        ],
        duration=16.0,  # 16拍 (4コード × 4拍)
        style="pop",
    )
    print("\n✓ Section defined:")
    print(f"  Tempo: {section.tempo} BPM")
    print(f"  Key: {section.key}")
    print(f"  Chords: {len(section.chord_progression)}")
    
    # Emotion定義
    emotion = EmotionProfile(
        primary=Emotion.JOY,
        intensity=0.7,
    )
    print(f"\n✓ Emotion: {emotion.primary.value} (intensity: {emotion.intensity})")
    
    # Context（空）
    context = GenerationContext()
    
    # 生成テスト
    print("\n" + "-"*50)
    print("Generating Piano (pop_comping)...")
    print("-"*50)
    
    notes = piano_gen.generate(
        section=section,
        technique="pop_comping",
        emotion=emotion,
        context=context,
    )
    
    print(f"\n✓ Generated {len(notes)} notes")
    
    # 統計
    pitches = [n.pitch for n in notes]
    velocities = [n.velocity for n in notes]
    durations = [n.duration for n in notes]
    
    print(f"\nStatistics:")
    print(f"  Pitch range: {min(pitches)} - {max(pitches)}")
    print(f"  Velocity range: {min(velocities)} - {max(velocities)}")
    print(f"  Duration range: {min(durations):.2f} - {max(durations):.2f}")
    print(f"  Total duration: {max(n.time + n.duration for n in notes):.2f} beats")
    
    # 検証
    validation = piano_gen.validate(notes)
    print(f"\nValidation:")
    print(f"  Passed: {validation.passed}")
    print(f"  Score: {validation.score:.2f}")
    print(f"  Metrics: {validation.metrics}")
    if validation.issues:
        print(f"  Issues: {validation.issues}")
    
    # MIDI出力
    output_file = "test_output/piano_test.mid"
    save_to_midi(notes, output_file, tempo=section.tempo)
    print(f"\n✓ Saved to: {output_file}")
    
    return notes


def test_techniques():
    """全Techniqueテスト"""
    print("\n" + "="*50)
    print("Testing All Techniques")
    print("="*50)
    
    piano_gen = PianoGenerator()
    
    section = Section(
        tempo=120.0,
        key="C",
        time_signature=(4, 4),
        chord_progression=[
            Chord(root=60, quality="major"),
            Chord(root=67, quality="major"),
        ],
        duration=8.0,
        style="pop",
    )
    
    emotion = EmotionProfile(primary=Emotion.JOY, intensity=0.7)
    context = GenerationContext()
    
    techniques = ["pop_comping", "ballad", "jazz_voicing", "arpeggio"]
    
    for technique in techniques:
        print(f"\n{technique}:")
        notes = piano_gen.generate(section, technique, emotion, context)
        print(f"  Notes: {len(notes)}")
        print(f"  Duration: {max(n.time + n.duration for n in notes):.2f} beats")
        
        # MIDI保存
        output_file = f"test_output/piano_{technique}.mid"
        save_to_midi(notes, output_file, tempo=section.tempo)
        print(f"  Saved: {output_file}")


def test_emotions():
    """全Emotion テスト"""
    print("\n" + "="*50)
    print("Testing All Emotions")
    print("="*50)
    
    piano_gen = PianoGenerator()
    
    section = Section(
        tempo=120.0,
        key="C",
        time_signature=(4, 4),
        chord_progression=[
            Chord(root=60, quality="major"),
            Chord(root=67, quality="major"),
        ],
        duration=8.0,
        style="pop",
    )
    
    context = GenerationContext()
    
    emotions = [
        (Emotion.JOY, "joy"),
        (Emotion.SORROW, "sorrow"),
        (Emotion.TENSION, "tension"),
        (Emotion.PEACE, "peace"),
    ]
    
    for emotion_type, name in emotions:
        print(f"\n{name}:")
        emotion = EmotionProfile(primary=emotion_type, intensity=0.8)
        notes = piano_gen.generate(section, "pop_comping", emotion, context)
        
        velocities = [n.velocity for n in notes]
        print(f"  Notes: {len(notes)}")
        print(f"  Velocity avg: {sum(velocities) / len(velocities):.1f}")
        
        # MIDI保存
        output_file = f"test_output/piano_emotion_{name}.mid"
        save_to_midi(notes, output_file, tempo=section.tempo)
        print(f"  Saved: {output_file}")


def save_to_midi(notes: list, output_file: str, tempo: float = 120.0):
    """MIDIファイルに保存"""
    import pathlib
    
    output_path = pathlib.Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # PrettyMIDI生成
    midi = pm.PrettyMIDI(initial_tempo=tempo)
    piano = pm.Instrument(program=0, name="Acoustic Grand Piano")
    
    for note_event in notes:
        # 拍 → 秒変換
        beat_duration = 60.0 / tempo
        start_time = note_event.time * beat_duration
        end_time = (note_event.time + note_event.duration) * beat_duration
        
        note = pm.Note(
            velocity=note_event.velocity,
            pitch=note_event.pitch,
            start=start_time,
            end=end_time,
        )
        piano.notes.append(note)
    
    midi.instruments.append(piano)
    midi.write(str(output_path))


if __name__ == "__main__":
    import os
    os.makedirs("test_output", exist_ok=True)
    
    # 基本テスト
    test_piano_generator()
    
    # Technique テスト
    test_techniques()
    
    # Emotion テスト
    test_emotions()
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)
