#!/usr/bin/env python3
"""
Suno WAV to MIDI Converter

Suno AIで生成したWAVファイルをMIDIに変換。
複数変換メソッド対応（basic/ensemble/demucs）。

Usage:
    python scripts/suno_wav_to_midi.py \\
        --input-dir data/suno_wav/guitar_strum_mid \\
        --output-dir data/suno_midi/guitar_strum_mid \\
        --method ensemble
"""

import argparse
import pathlib
import json
from typing import Dict, List, Any, Optional
import numpy as np

# Audio processing
try:
    import librosa
    import soundfile as sf
except ImportError:
    print("[ERROR] librosa/soundfile not installed:")
    print("  pip install librosa soundfile")
    exit(1)

# MIDI conversion
try:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
except ImportError:
    print("[ERROR] basic-pitch not installed:")
    print("  pip install basic-pitch")
    exit(1)

# MIDI handling
try:
    import pretty_midi as pm
except ImportError:
    print("[ERROR] pretty_midi not installed:")
    print("  pip install pretty-midi")
    exit(1)


def convert_basic_pitch(
    wav_path: pathlib.Path,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
) -> pm.PrettyMIDI:
    """
    basic-pitch変換（シンプル・高速）
    
    Args:
        wav_path: 入力WAVファイル
        onset_threshold: Note onset検出閾値（高い=厳しい）
        frame_threshold: Note継続検出閾値
    
    Returns:
        PrettyMIDI object
    """
    # Load audio
    audio, sr = librosa.load(str(wav_path), sr=22050, mono=True)
    
    # Predict with basic-pitch
    model_output, midi_data, note_events = predict(
        str(wav_path),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )
    
    # midi_data is already PrettyMIDI object
    return midi_data


def convert_ensemble(
    wav_path: pathlib.Path,
    num_models: int = 3,
    onset_thresholds: List[float] = [0.4, 0.5, 0.6],
    frame_thresholds: List[float] = [0.2, 0.3, 0.4],
) -> pm.PrettyMIDI:
    """
    Ensemble voting変換（高精度・低速）
    
    複数パラメータで変換 → 投票により最も確信度高いnoteを採用
    
    Args:
        wav_path: 入力WAVファイル
        num_models: 使用するパラメータセット数
        onset_thresholds: Onset閾値リスト
        frame_thresholds: Frame閾値リスト
    
    Returns:
        PrettyMIDI object (ensemble結果)
    """
    # Run multiple predictions
    all_notes: List[List[Dict[str, Any]]] = []
    
    for i in range(num_models):
        onset_th = onset_thresholds[i % len(onset_thresholds)]
        frame_th = frame_thresholds[i % len(frame_thresholds)]
        
        midi_data = convert_basic_pitch(wav_path, onset_th, frame_th)
        
        # Extract notes
        notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity,
                })
        
        all_notes.append(notes)
    
    # Voting: 過半数が一致したnoteのみ採用
    voted_notes = vote_notes(all_notes, threshold=0.5)
    
    # Create PrettyMIDI from voted notes
    midi = pm.PrettyMIDI()
    instrument = pm.Instrument(program=0)  # Piano (will be adjusted later)
    
    for note_dict in voted_notes:
        note = pm.Note(
            velocity=note_dict['velocity'],
            pitch=note_dict['pitch'],
            start=note_dict['start'],
            end=note_dict['end'],
        )
        instrument.notes.append(note)
    
    midi.instruments.append(instrument)
    return midi


def vote_notes(
    all_notes: List[List[Dict[str, Any]]],
    threshold: float = 0.5,
    time_tolerance: float = 0.05,  # 50ms
    pitch_tolerance: int = 0,  # Exact pitch match
) -> List[Dict[str, Any]]:
    """
    複数予測結果から投票により確信度高いnoteを抽出
    
    Args:
        all_notes: 各モデルのnote予測結果
        threshold: 採用するための最小投票率（0.5 = 過半数）
        time_tolerance: 時間的な一致判定許容誤差（秒）
        pitch_tolerance: ピッチの一致判定許容誤差（半音）
    
    Returns:
        投票により採用されたnote list
    """
    # Flatten all notes with model index
    flat_notes = []
    for model_idx, notes in enumerate(all_notes):
        for note in notes:
            flat_notes.append({**note, 'model': model_idx})
    
    # Sort by start time
    flat_notes.sort(key=lambda x: x['start'])
    
    # Clustering: 近い時間・ピッチのnoteをグループ化
    clusters: List[List[Dict[str, Any]]] = []
    used = set()
    
    for i, note in enumerate(flat_notes):
        if i in used:
            continue
        
        cluster = [note]
        used.add(i)
        
        # Find similar notes from other models
        for j, other in enumerate(flat_notes):
            if j in used or i == j:
                continue
            
            # Check similarity
            time_diff = abs(note['start'] - other['start'])
            pitch_diff = abs(note['pitch'] - other['pitch'])
            
            if time_diff <= time_tolerance and pitch_diff <= pitch_tolerance:
                cluster.append(other)
                used.add(j)
        
        clusters.append(cluster)
    
    # Voting: threshold以上の投票を得たclusterのみ採用
    num_models = len(all_notes)
    voted_notes = []
    
    for cluster in clusters:
        # Check if cluster has votes from different models
        model_votes = set(note['model'] for note in cluster)
        vote_rate = len(model_votes) / num_models
        
        if vote_rate >= threshold:
            # Average note parameters from cluster
            avg_note = {
                'pitch': int(np.median([n['pitch'] for n in cluster])),
                'start': float(np.mean([n['start'] for n in cluster])),
                'end': float(np.mean([n['end'] for n in cluster])),
                'velocity': int(np.mean([n['velocity'] for n in cluster])),
            }
            voted_notes.append(avg_note)
    
    return voted_notes


def post_process_midi(
    midi: pm.PrettyMIDI,
    quantize: bool = True,
    quantize_resolution: float = 0.03125,  # 32nd note at 120 BPM
    normalize_velocity: bool = True,
    velocity_range: tuple = (40, 100),
) -> pm.PrettyMIDI:
    """
    MIDI後処理（quantize, velocity normalization）
    
    Args:
        midi: 入力PrettyMIDI
        quantize: タイミングをquantize
        quantize_resolution: Quantize分解能（秒）
        normalize_velocity: Velocityを正規化
        velocity_range: 正規化後のvelocity範囲
    
    Returns:
        後処理済みPrettyMIDI
    """
    for instrument in midi.instruments:
        for note in instrument.notes:
            # Quantize timing
            if quantize:
                note.start = round(note.start / quantize_resolution) * quantize_resolution
                note.end = round(note.end / quantize_resolution) * quantize_resolution
            
            # Normalize velocity
            if normalize_velocity:
                # Clamp to range
                note.velocity = max(velocity_range[0], min(velocity_range[1], note.velocity))
    
    return midi


def main():
    parser = argparse.ArgumentParser(description="Convert Suno WAV to MIDI")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Input directory with WAV files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for MIDI files")
    parser.add_argument("--method", type=str, default="basic", 
                        choices=["basic", "ensemble"],
                        help="Conversion method: basic (fast) or ensemble (accurate)")
    parser.add_argument("--num-models", type=int, default=3,
                        help="Number of models for ensemble method (default: 3)")
    parser.add_argument("--onset-threshold", type=float, default=0.5,
                        help="Onset threshold for basic method (0.0-1.0)")
    parser.add_argument("--frame-threshold", type=float, default=0.3,
                        help="Frame threshold for basic method (0.0-1.0)")
    parser.add_argument("--vote-threshold", type=float, default=0.5,
                        help="Vote threshold for ensemble method (0.0-1.0)")
    parser.add_argument("--quantize", action="store_true",
                        help="Enable timing quantization")
    parser.add_argument("--quantize-resolution", type=float, default=0.03125,
                        help="Quantize resolution in seconds (default: 32nd note @ 120 BPM)")
    parser.add_argument("--normalize-velocity", action="store_true",
                        help="Enable velocity normalization")
    parser.add_argument("--velocity-min", type=int, default=40,
                        help="Minimum velocity (default: 40)")
    parser.add_argument("--velocity-max", type=int, default=100,
                        help="Maximum velocity (default: 100)")
    
    args = parser.parse_args()
    
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find WAV files
    wav_files = list(input_dir.glob("*.wav"))
    if not wav_files:
        print(f"[ERROR] No WAV files found in: {input_dir}")
        return 1
    
    print(f"[INFO] Found {len(wav_files)} WAV files")
    print(f"[INFO] Method: {args.method}")
    if args.method == "ensemble":
        print(f"[INFO] Ensemble models: {args.num_models}")
        print(f"[INFO] Vote threshold: {args.vote_threshold}")
    
    # Convert each WAV
    success_count = 0
    failed_count = 0
    total_notes = 0
    
    for i, wav_file in enumerate(wav_files, 1):
        try:
            print(f"\n[{i}/{len(wav_files)}] Converting: {wav_file.name}")
            
            # Convert WAV to MIDI
            if args.method == "basic":
                midi = convert_basic_pitch(
                    wav_file,
                    onset_threshold=args.onset_threshold,
                    frame_threshold=args.frame_threshold,
                )
            elif args.method == "ensemble":
                # Generate threshold arrays
                onset_base = args.onset_threshold
                frame_base = args.frame_threshold
                onset_thresholds = [onset_base - 0.1, onset_base, onset_base + 0.1]
                frame_thresholds = [frame_base - 0.1, frame_base, frame_base + 0.1]
                
                midi = convert_ensemble(
                    wav_file,
                    num_models=args.num_models,
                    onset_thresholds=onset_thresholds[:args.num_models],
                    frame_thresholds=frame_thresholds[:args.num_models],
                    vote_threshold=args.vote_threshold,
                )
            
            # Post-process
            midi = post_process_midi(
                midi,
                quantize=args.quantize,
                quantize_resolution=args.quantize_resolution,
                normalize_velocity=args.normalize_velocity,
                velocity_range=(args.velocity_min, args.velocity_max),
            )
            
            # Save MIDI
            output_file = output_dir / wav_file.with_suffix(".mid").name
            midi.write(str(output_file))
            
            # Count notes
            num_notes = sum(len(inst.notes) for inst in midi.instruments)
            total_notes += num_notes
            
            # Save metadata
            metadata = {
                "source_wav": wav_file.name,
                "conversion_method": args.method,
                "num_notes": num_notes,
                "duration": midi.get_end_time(),
                "onset_threshold": args.onset_threshold,
                "frame_threshold": args.frame_threshold,
            }
            
            if args.method == "ensemble":
                metadata["num_models"] = args.num_models
                metadata["vote_threshold"] = args.vote_threshold
            
            meta_file = output_dir / wav_file.with_suffix(".json").name
            with open(meta_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  ✓ Converted: {num_notes} notes, {midi.get_end_time():.2f}s")
            success_count += 1
        
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed_count += 1
    
    # Summary
    print("\n" + "="*50)
    print(f"[SUMMARY]")
    print(f"  Success:     {success_count}")
    print(f"  Failed:      {failed_count}")
    print(f"  Total:       {len(wav_files)}")
    print(f"  Total notes: {total_notes}")
    if success_count > 0:
        print(f"  Avg notes:   {total_notes / success_count:.1f}")
    print("="*50)
    
    print(f"\n[OUTPUT] MIDI files: {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
