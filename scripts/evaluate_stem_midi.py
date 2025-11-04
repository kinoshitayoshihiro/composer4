#!/usr/bin/env python3
"""
evaluate_stem_midi.py
---------------------
Suno AI stem MIDI品質評価（Phase C）

弱ラベル統合の前処理として、stem MIDIの信頼度を自動スコアリング:
- grid_f1: ビートグリッド一致度
- chord_tone_match: 和声音一致率
- confidence: 総合信頼度スコア

Usage:
    python3 scripts/evaluate_stem_midi.py \
      --stem-midi data/suno_ai/.../stemmidi_001/melody.mid \
      --audio data/suno_ai/.../full.wav \
      --bars song_packages/.../bars.parquet \
      --chordmap song_packages/.../chordmap.json \
      --out song_packages/.../stem_midi_quality.json
"""
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

try:
    import pretty_midi
    import librosa
    import pandas as pd
except ImportError as e:
    print(f"❌ Required library missing: {e}")
    print("Install: pip install pretty-midi librosa pandas")
    exit(1)


def ongrid_f1(
    onsets_sec: np.ndarray,
    grid_sec: np.ndarray,
    tolerance: float = 0.030
) -> Dict[str, float]:
    """
    ビートグリッド一致度（F1スコア）
    
    Args:
        onsets_sec: MIDI onset時刻配列
        grid_sec: ビートグリッド時刻配列（downbeats+8分音符想定）
        tolerance: 許容誤差（秒）
    
    Returns:
        {'f1', 'precision', 'recall'}
    """
    if len(onsets_sec) == 0 or len(grid_sec) == 0:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    # グリッド各点に対してonsetがtolerance以内にあるか
    matched_grid = np.zeros(len(grid_sec), dtype=bool)
    matched_onset = np.zeros(len(onsets_sec), dtype=bool)
    
    for i, onset_t in enumerate(onsets_sec):
        dists = np.abs(grid_sec - onset_t)
        j = np.argmin(dists)
        if dists[j] <= tolerance:
            matched_grid[j] = True
            matched_onset[i] = True
    
    tp = matched_onset.sum()
    fp = (~matched_onset).sum()
    fn = (~matched_grid).sum()
    
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    
    return {
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }


def chord_tone_match_rate(
    notes: List[pretty_midi.Note],
    chordmap: List[Dict[str, Any]],
    ppq: int = 480,
    tempo_bpm: float = 120.0
) -> float:
    """
    和声音一致率
    
    Args:
        notes: MIDI Note配列
        chordmap: [{bar, beat, chord_symbol}, ...]
        ppq: PPQ
        tempo_bpm: テンポ
    
    Returns:
        chord_tone_match: 0.0～1.0
    """
    if len(notes) == 0 or len(chordmap) == 0:
        return 0.0
    
    # 簡易実装: C/Dm/G7等から1/3/5/(7)を抽出（voicing_engineへ委譲可）
    def get_chord_tones(symbol: str) -> set:
        # 極簡易: root + maj/min判定のみ
        # 実運用ではvoicing_engine.parse_chord()を使用
        root_map = {'C':0, 'D':2, 'E':4, 'F':5, 'G':7, 'A':9, 'B':11,
                    'Db':1, 'Eb':3, 'Gb':6, 'Ab':8, 'Bb':10}
        
        root = symbol[0:2] if len(symbol) > 1 and symbol[1] in ['b', '#'] else symbol[0]
        root_pc = root_map.get(root, 0)
        
        if 'm' in symbol.lower():
            return {root_pc, (root_pc+3)%12, (root_pc+7)%12}  # minor
        else:
            return {root_pc, (root_pc+4)%12, (root_pc+7)%12}  # major
    
    # 小節/拍→秒変換（簡易）
    def bar_beat_to_sec(bar: int, beat: float) -> float:
        beats_per_sec = tempo_bpm / 60.0
        total_beats = bar * 4 + beat  # 4/4想定
        return total_beats / beats_per_sec
    
    # chordmap時刻でソート
    chordmap_sorted = sorted(chordmap, key=lambda c: bar_beat_to_sec(c['bar'], c['beat']))
    
    hits = 0
    total = 0
    
    for note in notes:
        # note時刻に対応するコードを検索
        note_sec = note.start
        chord = None
        for i, c in enumerate(chordmap_sorted):
            c_sec = bar_beat_to_sec(c['bar'], c['beat'])
            if c_sec > note_sec:
                chord = chordmap_sorted[max(0, i-1)]
                break
        else:
            chord = chordmap_sorted[-1] if chordmap_sorted else None
        
        if chord is None:
            continue
        
        chord_tones = get_chord_tones(chord['chord_symbol'])
        note_pc = note.pitch % 12
        
        total += 1
        if note_pc in chord_tones:
            hits += 1
    
    return hits / max(1, total)


def evaluate_stem_midi(
    midi_path: Path,
    audio_path: Path,
    bars_parquet: Optional[Path] = None,
    chordmap_json: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Stem MIDI品質評価
    
    Returns:
        {
            'grid_f1': float,
            'chord_tone_match': float,
            'confidence': float,  # 総合スコア
            'note_count': int,
            'duration_sec': float
        }
    """
    # MIDI読込
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"⚠️  MIDI読込失敗: {e}")
        return {'confidence': 0.0}
    
    # 音声読込・ビート推定
    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beats_sec = librosa.frames_to_time(beats, sr=sr)
    except Exception as e:
        print(f"⚠️  Audio読込/Beat推定失敗: {e}")
        beats_sec = np.array([])
    
    # bars.parquetからdownbeatsを取得（優先）
    if bars_parquet and bars_parquet.exists():
        try:
            bars_df = pd.read_parquet(bars_parquet)
            if 'start_sec' in bars_df.columns:
                downbeats_sec = bars_df['start_sec'].values
                # 8分音符グリッド生成（簡易）
                grid_sec = []
                for db in downbeats_sec:
                    grid_sec.extend([db + i*0.25*(60/tempo) for i in range(8)])  # 4/4想定
                beats_sec = np.array(sorted(grid_sec))
        except Exception as e:
            print(f"⚠️  bars.parquet読込失敗: {e}")
    
    # chordmap読込
    chordmap = []
    if chordmap_json and chordmap_json.exists():
        try:
            chordmap = json.loads(chordmap_json.read_text())
        except Exception as e:
            print(f"⚠️  chordmap読込失敗: {e}")
    
    # メロディトラック抽出（最多ノート）
    if len(pm.instruments) == 0:
        return {'confidence': 0.0, 'note_count': 0}
    
    melody_track = max(pm.instruments, key=lambda ins: len(ins.notes))
    notes = melody_track.notes
    
    if len(notes) == 0:
        return {'confidence': 0.0, 'note_count': 0}
    
    # Onset時刻配列
    onsets_sec = np.array([n.start for n in notes])
    
    # Grid F1計算
    grid_metrics = ongrid_f1(onsets_sec, beats_sec, tolerance=0.030)
    
    # Chord-tone一致率
    tempo_bpm = pm.get_tempo_changes()[1][0] if len(pm.get_tempo_changes()[1]) > 0 else 120.0
    ctm = chord_tone_match_rate(notes, chordmap, ppq=480, tempo_bpm=tempo_bpm)
    
    # 総合信頼度（重み付け平均）
    confidence = 0.5 * grid_metrics['f1'] + 0.5 * ctm
    
    return {
        'grid_f1': grid_metrics['f1'],
        'grid_precision': grid_metrics['precision'],
        'grid_recall': grid_metrics['recall'],
        'chord_tone_match': ctm,
        'confidence': confidence,
        'note_count': len(notes),
        'duration_sec': float(pm.get_end_time())
    }


def main():
    ap = argparse.ArgumentParser(description="Stem MIDI品質評価")
    ap.add_argument('--stem-midi', type=Path, required=True, help='Stem MIDIファイル')
    ap.add_argument('--audio', type=Path, required=True, help='音声ファイル（full.wav）')
    ap.add_argument('--bars', type=Path, help='bars.parquet（オプション）')
    ap.add_argument('--chordmap', type=Path, help='chordmap.json（オプション）')
    ap.add_argument('--out', type=Path, help='出力JSON（デフォルト: stem_midi_quality.json）')
    args = ap.parse_args()
    
    print(f"📊 Evaluating Stem MIDI: {args.stem_midi.name}")
    
    result = evaluate_stem_midi(
        args.stem_midi,
        args.audio,
        args.bars,
        args.chordmap
    )
    
    # 結果表示
    print(f"\n✅ Evaluation Results:")
    print(f"   Grid F1: {result.get('grid_f1', 0.0):.3f}")
    print(f"   Chord-tone Match: {result.get('chord_tone_match', 0.0):.3f}")
    print(f"   Confidence: {result.get('confidence', 0.0):.3f}")
    print(f"   Notes: {result.get('note_count', 0)}")
    
    # 出力
    out_path = args.out or Path('stem_midi_quality.json')
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n💾 Saved: {out_path}")


if __name__ == "__main__":
    main()
