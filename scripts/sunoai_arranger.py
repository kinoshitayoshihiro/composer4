#!/usr/bin/env python3
"""
SunoAI Arranger - Phase 1: Audio Analysis Pipeline
既存のchordmap/sections/tempo_map + Demucs stems → Rhythm/Harmony AI統合 → 5トラックMIDI生成

データフロー:
1. 既存解析データ読み込み (chordmap.json, sections.json, tempo_map.json)
2. Demucs stem WAVファイル読み込み (10トラック: Vocals/Bass/Drums/FX/Guitar/Keyboard/Percussion/Strings/Synth/Backing Vocals)
3. Rhythm AI: Drumsトラック解析 → rhythm_patterns.pickle検索 → 最適パターン選択
4. Harmony AI: コード進行 + roleマッピング → harmony_patterns.pickle検索 → 最適パターン選択
5. MIDI生成: Piano/Bass/Drums/Guitar/Strings (5トラック)
6. VST3レンダリング: Analog Lab V → WAV出力
7. ミックス: Demucs vocals + 新MIDI音源 → 最終出力

Usage:
    python scripts/sunoai_arranger.py \\
        --input data/suno_ai/suno_themesong/song_001 \\
        --output output/suno_ai/song_001_arranged \\
        --rhythm-pickle output/rhythm_patterns.pickle \\
        --harmony-pickle output/harmony_patterns.pickle
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import numpy as np
import pandas as pd
import yaml
import librosa

# 内部モジュール
# from generators.emotional_humanizer import EmotionalHumanizer  # 必要に応じて統合


# ============================================================================
# ロギング設定
# ============================================================================

def setup_logging(log_level: str = "INFO"):
    """ロギング設定"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# ============================================================================
# データ読み込み
# ============================================================================

def load_chordmap(path: Path) -> Dict:
    """chordmap.json読み込み"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_sections(path: Path) -> Dict:
    """sections.json読み込み"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_tempo_map(path: Path) -> Dict:
    """tempo_map.json読み込み"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_pickle_index(pickle_path: Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Pickle v1.1.0ファイル読み込み
    
    Returns:
        (preview_df, metadata_dict)
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # metadata取得
    metadata = data.get('metadata', {})
    mode = metadata.get('mode', 'lite')
    
    if mode == 'fat':
        # Fat mode: 全データ埋め込み済み
        features_payload = data.get('features_payload', [])
        features_df = pd.DataFrame(features_payload)
    else:
        # Lite mode: 外部Parquetファイル参照
        features_path = metadata.get('features_path')
        if not features_path or not Path(features_path).exists():
            logging.warning(f"Lite mode: features_path not found ({features_path}), using preview only")
            features_df = pd.DataFrame(data.get('preview_rows', []))
        else:
            features_df = pd.read_parquet(features_path)
    
    # ID indexing取得
    id_index = data.get('id_index', {})
    
    return features_df, metadata, id_index


# ============================================================================
# Stem WAV解析
# ============================================================================

STEM_ROLE_MAPPING = {
    # Demucs stem → Harmony AI role
    'Bass': 'bass',
    'Guitar': 'guitar',
    'Keyboard': 'piano',  # or 'other_keys'
    'Strings': 'strings',
    'Synth': 'other',
    'Percussion': None,  # Rhythm AIで処理
    'Drums': None,  # Rhythm AIで処理
    'Vocals': None,  # ミックス時に原曲使用
    'Backing Vocals': None,
    'FX': None
}


def detect_active_stems(stems_dir: Path) -> Dict[str, Path]:
    """
    Demucs stem WAVファイル検出
    
    Returns:
        {stem_name: wav_path} (例: {'Bass': Path('stem_wav_001_(Bass).wav')})
    """
    active_stems = {}
    
    for stem_file in stems_dir.glob("stem_wav_*.wav"):
        # ファイル名例: stem_wav_001_(Bass).wav
        stem_name = stem_file.stem.split('_(')[-1].rstrip(')')
        active_stems[stem_name] = stem_file
    
    logging.info(f"Detected {len(active_stems)} active stems: {list(active_stems.keys())}")
    return active_stems


def analyze_drums_stem(drums_path: Path, tempo_map: List[Tuple[int, float]]) -> Dict:
    """
    Drums stem解析 → Rhythm AI用特徴量抽出
    
    Args:
        drums_path: Drums stem WAVファイルパス
        tempo_map: [(bar, tempo_bpm), ...] 小節ごとのテンポ
    
    Returns:
        {
            'avg_tempo': 平均テンポ,
            'onset_density': オンセット密度 (events/sec),
            'energy_profile': 小節ごとのエネルギー配列
        }
    """
    y, sr = librosa.load(drums_path, sr=22050, mono=True)
    
    # 平均テンポ
    if tempo_map:
        avg_tempo = np.mean([t for _, t in tempo_map])
    else:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        avg_tempo = float(tempo)
    
    # オンセット検出
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_density = len(onsets) / (len(y) / sr)  # events/sec
    
    # エネルギープロファイル (小節ごと)
    hop_length = 512
    frame_duration = hop_length / sr
    frames_per_bar = int((60.0 / avg_tempo * 4) / frame_duration)  # 4/4拍子前提
    
    energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    num_bars = len(energy) // frames_per_bar
    energy_profile = [
        float(np.mean(energy[i*frames_per_bar:(i+1)*frames_per_bar]))
        for i in range(num_bars)
    ]
    
    logging.info(f"Drums analysis: tempo={avg_tempo:.1f} BPM, onset_density={onset_density:.2f}/s, bars={num_bars}")
    
    return {
        'avg_tempo': avg_tempo,
        'onset_density': onset_density,
        'energy_profile': energy_profile,
        'num_bars': num_bars
    }


# ============================================================================
# Rhythm AI パターンマッチング
# ============================================================================

def match_rhythm_patterns(
    drums_features: Dict,
    sections: List[Dict],
    rhythm_df: pd.DataFrame,
    rhythm_metadata: Dict
) -> List[Dict]:
    """
    Rhythm AI pickle検索 → 最適パターン選択
    
    Args:
        drums_features: analyze_drums_stem()の出力
        sections: sections.jsonのsections配列
        rhythm_df: rhythm_patterns.pickle DataFrame
        rhythm_metadata: pickle metadata
    
    Returns:
        [{'bar': int, 'loop_id': str, 'pattern_data': dict}, ...]
    """
    tempo = drums_features['avg_tempo']
    energy_profile = drums_features['energy_profile']
    num_bars = drums_features['num_bars']
    
    # Family列検出
    family_column = rhythm_metadata.get('family_column', 'family_label')
    if family_column not in rhythm_df.columns:
        family_column = 'family_label' if 'family_label' in rhythm_df.columns else None
    
    # テンポフィルタ (±10 BPM)
    if 'tempo_bpm' in rhythm_df.columns:
        tempo_filtered = rhythm_df[
            (rhythm_df['tempo_bpm'] >= tempo - 10) &
            (rhythm_df['tempo_bpm'] <= tempo + 10)
        ].copy()
    else:
        tempo_filtered = rhythm_df.copy()
    
    logging.info(f"Tempo filter ({tempo-10:.1f}-{tempo+10:.1f} BPM): {len(tempo_filtered)}/{len(rhythm_df)} patterns")
    
    # セクションごとにパターン選択
    selected_patterns = []
    
    for section in sections:
        bar_start = section.get('bar', 0)
        section_label = section.get('label', 'unknown')
        
        # エネルギー取得
        section_energy = energy_profile[bar_start] if bar_start < len(energy_profile) else 0.5
        
        # Family選択 (intro/outro → SWING, verse/chorus → STRAIGHT)
        if section_label in ('intro', 'outro', 'bridge'):
            preferred_family = 'SWING_8'
        else:
            preferred_family = 'STRAIGHT_8'
        
        # Family フィルタ
        if family_column and preferred_family in rhythm_df[family_column].unique():
            family_filtered = tempo_filtered[tempo_filtered[family_column] == preferred_family]
        else:
            family_filtered = tempo_filtered
        
        # ランダム選択 (将来的に類似度計算で改善)
        if len(family_filtered) > 0:
            pattern_row = family_filtered.sample(n=1).iloc[0]
            loop_id = pattern_row.get('loop_id', f"pattern_{bar_start}")
            
            selected_patterns.append({
                'bar': bar_start,
                'section_label': section_label,
                'loop_id': loop_id,
                'family': preferred_family,
                'tempo_bpm': float(pattern_row.get('tempo_bpm', tempo)),
                'energy': section_energy
            })
            
            logging.debug(f"Bar {bar_start} ({section_label}): selected loop_id={loop_id}, family={preferred_family}")
    
    logging.info(f"Selected {len(selected_patterns)} rhythm patterns for {num_bars} bars")
    return selected_patterns


# ============================================================================
# Harmony AI パターンマッチング
# ============================================================================

def match_harmony_patterns(
    chordmap: Dict,
    sections: List[Dict],
    active_stems: Dict[str, Path],
    harmony_df: pd.DataFrame,
    harmony_metadata: Dict
) -> List[Dict]:
    """
    Harmony AI pickle検索 → 最適パターン選択
    
    Args:
        chordmap: chordmap.json
        sections: sections.jsonのsections配列
        active_stems: {stem_name: wav_path}
        harmony_df: harmony_patterns.pickle DataFrame
        harmony_metadata: pickle metadata
    
    Returns:
        [{'bar': int, 'chord': str, 'role': str, 'song_id': str, 'weight': float}, ...]
    """
    events = chordmap.get('events', [])
    
    # Stem → Role マッピング
    active_roles = {}
    for stem_name, stem_path in active_stems.items():
        role = STEM_ROLE_MAPPING.get(stem_name)
        if role:
            active_roles[role] = stem_path
    
    logging.info(f"Active harmony roles: {list(active_roles.keys())}")
    
    # Role フィルタ
    if 'role' in harmony_df.columns:
        role_filtered = harmony_df[harmony_df['role'].isin(active_roles.keys())].copy()
    else:
        role_filtered = harmony_df.copy()
    
    logging.info(f"Role filter: {len(role_filtered)}/{len(harmony_df)} patterns")
    
    # コード進行マッチング (簡易版: ルート音のみマッチング)
    selected_patterns = []
    
    for event in events:
        time_ql = event.get('time', 0.0)
        bar = int(time_ql / 4)  # 4QL = 1小節 (4/4拍子)
        chord_root = event.get('root', 'C')
        chord_quality = event.get('quality', 'maj')
        chord_full = f"{chord_root}:{chord_quality}"
        
        # セクション検出
        section_label = 'unknown'
        for section in sections:
            if bar >= section.get('bar', 0):
                section_label = section.get('label', 'unknown')
        
        # Role別にパターン選択
        for role in active_roles.keys():
            role_patterns = role_filtered[role_filtered['role'] == role]
            
            if len(role_patterns) > 0:
                # ランダム選択 (将来的にコード類似度で改善)
                pattern_row = role_patterns.sample(n=1).iloc[0]
                song_id = pattern_row.get('song_id', f"harmony_{bar}")
                
                selected_patterns.append({
                    'bar': bar,
                    'time_ql': time_ql,
                    'chord': chord_full,
                    'chord_root': chord_root,
                    'section_label': section_label,
                    'role': role,
                    'song_id': song_id,
                    'weight': float(pattern_row.get('weight', 1.0))
                })
    
    logging.info(f"Selected {len(selected_patterns)} harmony patterns ({len(events)} chords × {len(active_roles)} roles)")
    return selected_patterns


# ============================================================================
# MIDI生成 (プレースホルダー)
# ============================================================================

def generate_midi_tracks(
    rhythm_patterns: List[Dict],
    harmony_patterns: List[Dict],
    output_dir: Path
) -> Path:
    """
    選択されたパターン群 → 5トラックMIDI生成
    
    Args:
        rhythm_patterns: match_rhythm_patterns()の出力
        harmony_patterns: match_harmony_patterns()の出力
        output_dir: 出力ディレクトリ
    
    Returns:
        生成されたMIDIファイルパス
    """
    # TODO: Phase 2で実装
    # 1. rhythm_patterns → Drumsトラック生成
    # 2. harmony_patterns → Piano/Bass/Guitar/Stringsトラック生成 (roleごと)
    # 3. mido.MidiFile()で5トラック統合
    
    midi_path = output_dir / "arranged_5tracks.mid"
    logging.warning(f"MIDI generation not yet implemented. Placeholder: {midi_path}")
    
    # プレースホルダー: 空MIDIファイル
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    midi_path.touch()
    
    return midi_path


# ============================================================================
# メイン処理
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SunoAI Arranger - Phase 1: Audio Analysis Pipeline")
    parser.add_argument('--input', type=str, required=True,
                        help='Input song directory (e.g., data/suno_ai/suno_themesong/song_001)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory (e.g., output/suno_ai/song_001_arranged)')
    parser.add_argument('--rhythm-pickle', type=str, required=True,
                        help='Rhythm AI pickle file (e.g., output/rhythm_patterns.pickle)')
    parser.add_argument('--harmony-pickle', type=str, required=True,
                        help='Harmony AI pickle file (e.g., output/harmony_patterns.pickle)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    # パス設定
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_dir = input_dir / "analysis"
    stems_dir = input_dir / "stemswav_001"
    
    # ============================================================================
    # Step 1: 既存解析データ読み込み
    # ============================================================================
    logging.info("=" * 80)
    logging.info("Step 1: Loading existing analysis data")
    logging.info("=" * 80)
    
    chordmap = load_chordmap(analysis_dir / "chordmap.json")
    sections_data = load_sections(analysis_dir / "sections.json")
    tempo_map_data = load_tempo_map(analysis_dir / "tempo_map.json")
    
    sections = sections_data.get('sections', [])
    tempo_map = tempo_map_data.get('tempo_map', [])
    
    logging.info(f"Loaded: {len(chordmap.get('events', []))} chord events, "
                 f"{len(sections)} sections, {len(tempo_map)} tempo points")
    
    # ============================================================================
    # Step 2: Demucs stem WAV検出
    # ============================================================================
    logging.info("=" * 80)
    logging.info("Step 2: Detecting Demucs stem WAV files")
    logging.info("=" * 80)
    
    active_stems = detect_active_stems(stems_dir)
    
    # ============================================================================
    # Step 3: Drums stem解析
    # ============================================================================
    logging.info("=" * 80)
    logging.info("Step 3: Analyzing Drums stem")
    logging.info("=" * 80)
    
    drums_path = active_stems.get('Drums')
    if not drums_path or not drums_path.exists():
        logging.error(f"Drums stem not found: {drums_path}")
        return
    
    drums_features = analyze_drums_stem(drums_path, tempo_map)
    
    # ============================================================================
    # Step 4: Rhythm AI pickle読み込み
    # ============================================================================
    logging.info("=" * 80)
    logging.info("Step 4: Loading Rhythm AI pickle")
    logging.info("=" * 80)
    
    rhythm_pickle_path = Path(args.rhythm_pickle)
    if not rhythm_pickle_path.exists():
        logging.error(f"Rhythm pickle not found: {rhythm_pickle_path}")
        return
    
    rhythm_df, rhythm_metadata, rhythm_id_index = load_pickle_index(rhythm_pickle_path)
    logging.info(f"Loaded rhythm_patterns.pickle: {len(rhythm_df)} patterns, mode={rhythm_metadata.get('mode')}")
    
    # ============================================================================
    # Step 5: Rhythm AI パターンマッチング
    # ============================================================================
    logging.info("=" * 80)
    logging.info("Step 5: Rhythm AI pattern matching")
    logging.info("=" * 80)
    
    rhythm_patterns = match_rhythm_patterns(drums_features, sections, rhythm_df, rhythm_metadata)
    
    # ============================================================================
    # Step 6: Harmony AI pickle読み込み
    # ============================================================================
    logging.info("=" * 80)
    logging.info("Step 6: Loading Harmony AI pickle")
    logging.info("=" * 80)
    
    harmony_pickle_path = Path(args.harmony_pickle)
    if not harmony_pickle_path.exists():
        logging.error(f"Harmony pickle not found: {harmony_pickle_path}")
        return
    
    harmony_df, harmony_metadata, harmony_id_index = load_pickle_index(harmony_pickle_path)
    logging.info(f"Loaded harmony_patterns.pickle: {len(harmony_df)} patterns, mode={harmony_metadata.get('mode')}")
    
    # ============================================================================
    # Step 7: Harmony AI パターンマッチング
    # ============================================================================
    logging.info("=" * 80)
    logging.info("Step 7: Harmony AI pattern matching")
    logging.info("=" * 80)
    
    harmony_patterns = match_harmony_patterns(chordmap, sections, active_stems, harmony_df, harmony_metadata)
    
    # ============================================================================
    # Step 8: MIDI生成
    # ============================================================================
    logging.info("=" * 80)
    logging.info("Step 8: Generating 5-track MIDI")
    logging.info("=" * 80)
    
    midi_path = generate_midi_tracks(rhythm_patterns, harmony_patterns, output_dir)
    
    # ============================================================================
    # Step 9: 結果保存
    # ============================================================================
    logging.info("=" * 80)
    logging.info("Step 9: Saving results")
    logging.info("=" * 80)
    
    result_json = output_dir / "arrangement_result.json"
    result_data = {
        'input_song': str(input_dir),
        'drums_features': drums_features,
        'rhythm_patterns': rhythm_patterns,
        'harmony_patterns': harmony_patterns,
        'midi_output': str(midi_path),
        'metadata': {
            'rhythm_pickle': str(rhythm_pickle_path),
            'harmony_pickle': str(harmony_pickle_path),
            'num_rhythm_patterns': len(rhythm_patterns),
            'num_harmony_patterns': len(harmony_patterns)
        }
    }
    
    with open(result_json, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    logging.info(f"✅ SunoAI Arranger Phase 1 Complete!")
    logging.info(f"   - Rhythm patterns: {len(rhythm_patterns)}")
    logging.info(f"   - Harmony patterns: {len(harmony_patterns)}")
    logging.info(f"   - MIDI output: {midi_path}")
    logging.info(f"   - Result JSON: {result_json}")


if __name__ == '__main__':
    main()
