#!/usr/bin/env python3
"""
Merge WAV acoustic features into Stage2 rhythm features

Usage:
    python scripts/merge_wav_features.py \\
        --rhythm-parquet output/rhythm_ai/egmd_stage2/rhythm_features.parquet \\
        --wav-index output/wav_stage1/index/wav_index.pkl \\
        --output output/rhythm_ai/egmd_stage2/rhythm_features_with_wav.parquet
"""

import argparse
import pickle
from pathlib import Path
import pandas as pd


def load_wav_index(wav_index_path: Path) -> pd.DataFrame:
    """Load WAV Stage1 index (pkl or csv)"""
    
    if wav_index_path.suffix == '.pkl':
        with open(wav_index_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            # List of dicts
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Dict of lists
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Unknown WAV index format: {type(data)}")
    
    elif wav_index_path.suffix == '.csv':
        return pd.read_csv(wav_index_path)
    
    else:
        raise ValueError(f"Unsupported WAV index format: {wav_index_path.suffix}")


def match_midi_to_wav(midi_path: str, wav_df: pd.DataFrame) -> dict:
    """Match MIDI file to WAV record by filename similarity
    
    E-GMD convention:
    - MIDI: drummer1/session1/1_funk-groove_120_beat_4-4.midi
    - WAV:  drummer1/session1/audio_mic/1_funk-groove_120_beat_4-4.wav
    """
    
    midi_stem = Path(midi_path).stem
    
    # Find matching WAV (exact stem match)
    matches = wav_df[wav_df['original_path'].str.contains(midi_stem, na=False)]
    
    if len(matches) == 0:
        return {}
    
    # Take first match
    wav_rec = matches.iloc[0]
    
    return {
        'wav_path': wav_rec.get('original_path', ''),
        'wav_duration_s': wav_rec.get('duration_out_s'),
        'wav_onset_rate_hz': wav_rec.get('onset_rate_hz'),
        'wav_clip_ratio': wav_rec.get('clip_ratio'),
        'wav_rms': wav_rec.get('rms'),
        'wav_peak': wav_rec.get('peak'),
    }


def merge_wav_features(
    rhythm_parquet: Path,
    wav_index: Path,
    output: Path
) -> None:
    """Merge WAV acoustic features into rhythm DataFrame"""
    
    print(f"📂 Loading rhythm features: {rhythm_parquet}")
    df_rhythm = pd.read_parquet(rhythm_parquet)
    print(f"   Records: {len(df_rhythm)}")
    print(f"   Columns: {list(df_rhythm.columns)}")
    
    print(f"\n📂 Loading WAV index: {wav_index}")
    df_wav = load_wav_index(wav_index)
    print(f"   WAV records: {len(df_wav)}")
    
    # Match and merge
    print(f"\n🔗 Matching MIDI → WAV...")
    wav_features = []
    
    for i, row in df_rhythm.iterrows():
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(df_rhythm)} ({100*i/len(df_rhythm):.1f}%)")
        
        # Get MIDI path from metadata (if available)
        midi_path = row.get('loop_id', '')  # Fallback to loop_id
        
        # Match WAV
        wav_feats = match_midi_to_wav(midi_path, df_wav)
        wav_features.append(wav_feats)
    
    # Convert to DataFrame
    df_wav_feats = pd.DataFrame(wav_features)
    
    # Merge
    df_merged = pd.concat([df_rhythm, df_wav_feats], axis=1)
    
    # Fill NaN with defaults
    for col in df_wav_feats.columns:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0.0 if col != 'wav_path' else '')
    
    print(f"\n✅ Merged columns: {list(df_wav_feats.columns)}")
    print(f"   Total columns: {len(df_merged.columns)}")
    print(f"   WAV matches: {(df_merged['wav_duration_s'] > 0).sum()}/{len(df_merged)}")
    
    # Save
    df_merged.to_parquet(output, compression='snappy', index=False)
    print(f"\n💾 Saved: {output}")
    print(f"   Records: {len(df_merged)}")
    print(f"   Columns: {len(df_merged.columns)}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge WAV acoustic features into Stage2 rhythm features"
    )
    parser.add_argument(
        '--rhythm-parquet',
        type=Path,
        required=True,
        help="Input rhythm features parquet (from Stage2)"
    )
    parser.add_argument(
        '--wav-index',
        type=Path,
        required=True,
        help="WAV Stage1 index (pkl or csv)"
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help="Output parquet with merged WAV features"
    )
    
    args = parser.parse_args()
    
    merge_wav_features(
        rhythm_parquet=args.rhythm_parquet,
        wav_index=args.wav_index,
        output=args.output
    )


if __name__ == '__main__':
    main()
