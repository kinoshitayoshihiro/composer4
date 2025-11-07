#!/usr/bin/env python3
"""
Magenta GrooVAE/Drumify Wrapper（軽量CLI）
note-seqを使用してMIDI→Groove/Drumify→MIDI変換

Usage:
  python3 ops/magenta_groove.py groove -i seed.mid -o grooved.mid [--temp 0.8]
  python3 ops/magenta_groove.py drumify -i seed.mid -o drumified.mid [--temp 0.9]
"""
import argparse
import sys
import os

try:
    import note_seq
    from note_seq.protobuf import music_pb2
except ImportError:
    print("❌ note-seq未インストール: pip install note-seq", file=sys.stderr)
    sys.exit(1)

def load_midi(midi_path):
    """MIDI→NoteSequence読み込み"""
    try:
        return note_seq.midi_file_to_note_sequence(midi_path)
    except Exception as e:
        print(f"❌ MIDI読み込み失敗: {e}", file=sys.stderr)
        sys.exit(1)

def save_midi(sequence, midi_path):
    """NoteSequence→MIDI書き出し"""
    try:
        note_seq.sequence_proto_to_midi_file(sequence, midi_path)
        print(f"✅ 保存完了: {midi_path}")
    except Exception as e:
        print(f"❌ MIDI保存失敗: {e}", file=sys.stderr)
        sys.exit(1)

def groove_humanize(sequence, temperature=0.8):
    """
    GrooVAE humanize（簡易版）
    
    注: 本格実装はmagenta.models.music_vae必要（checkpoint読み込み）
    現状はタイミング/Vel微小変動のみ適用（プロトタイプ）
    """
    import random
    random.seed(42)
    
    ns = music_pb2.NoteSequence()
    ns.CopyFrom(sequence)
    
    for note in ns.notes:
        # マイクロタイミング（±10ms相当、ランダムウォーク）
        time_jitter = random.gauss(0, 0.01 * temperature)
        note.start_time = max(0, note.start_time + time_jitter)
        note.end_time = max(note.start_time + 0.05, note.end_time + time_jitter)
        
        # Velランダム化（±5 * temperature）
        vel_jitter = int(random.gauss(0, 5 * temperature))
        note.velocity = max(1, min(127, note.velocity + vel_jitter))
    
    print(f"⚠️  簡易groove適用（checkpoint未使用）: notes={len(ns.notes)}, temp={temperature}")
    return ns

def drumify_simple(sequence, temperature=0.9):
    """
    Drumify簡易版（プロトタイプ）
    
    注: 本格実装はcheckpoint必要
    現状はキック/スネア/ハットの簡易パターン生成のみ
    """
    import random
    random.seed(42)
    
    ns = music_pb2.NoteSequence()
    ns.total_time = sequence.total_time
    ns.ticks_per_quarter = sequence.ticks_per_quarter if sequence.ticks_per_quarter else 480
    
    # 簡易ドラムパターン生成（4/4拍子想定）
    qpm = sequence.tempos[0].qpm if sequence.tempos else 120.0
    beat_duration = 60.0 / qpm
    
    for bar in range(int(ns.total_time / (beat_duration * 4)) + 1):
        bar_start = bar * beat_duration * 4
        
        # キック（0, 2拍）
        for beat in [0, 2]:
            note = ns.notes.add()
            note.instrument = 9
            note.program = 0
            note.pitch = 36
            note.velocity = int(100 + random.gauss(0, 10 * temperature))
            note.start_time = bar_start + beat * beat_duration
            note.end_time = note.start_time + 0.1
        
        # スネア（1, 3拍）
        for beat in [1, 3]:
            note = ns.notes.add()
            note.instrument = 9
            note.program = 0
            note.pitch = 38
            note.velocity = int(90 + random.gauss(0, 8 * temperature))
            note.start_time = bar_start + beat * beat_duration
            note.end_time = note.start_time + 0.1
        
        # ハット（8分音符）
        for eighth in range(8):
            note = ns.notes.add()
            note.instrument = 9
            note.program = 0
            note.pitch = 42 if eighth % 2 == 0 else 46
            note.velocity = int(70 + random.gauss(0, 5 * temperature))
            note.start_time = bar_start + eighth * beat_duration / 2.0
            note.end_time = note.start_time + 0.05
    
    print(f"⚠️  簡易drumify適用（checkpoint未使用）: notes={len(ns.notes)}, temp={temperature}")
    return ns

def main():
    ap = argparse.ArgumentParser(description="Magenta GrooVAE/Drumify Wrapper")
    ap.add_argument("mode", choices=["groove", "drumify"], help="groove=humanize, drumify=パターン生成")
    ap.add_argument("-i", "--input", required=True, help="入力MIDI")
    ap.add_argument("-o", "--output", required=True, help="出力MIDI")
    ap.add_argument("--temp", "--temperature", type=float, default=0.8, help="温度（0.0-1.0、高いほどランダム）")
    args = ap.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 入力ファイル未存在: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    seq = load_midi(args.input)
    
    if args.mode == "groove":
        out_seq = groove_humanize(seq, args.temp)
    elif args.mode == "drumify":
        out_seq = drumify_simple(seq, args.temp)
    else:
        print(f"❌ 未知のモード: {args.mode}", file=sys.stderr)
        sys.exit(1)
    
    save_midi(out_seq, args.output)

if __name__ == "__main__":
    main()
