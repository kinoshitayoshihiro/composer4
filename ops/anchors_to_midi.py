#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/anchors_to_midi.py — Chorus用 歌詞付きMIDI生成（Synthesizer V取り込み想定）

目的:
- lyric_anchors_enriched.json から chorus 区間のアンカーを抽出
- stress アンカーを発音点、次の強勢までを音価に
- sibilant は音符末尾を 20-40ms 削る（高域減衰）
- 各ノートに Lyric メタイベント追加
- 可変テンポを Conductor Track に反映

出力:
- Format 1 MIDI (Track 0: Conductor, Track 1: Vocal Melody + Lyrics)
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple


def create_midi_header(format_type=1, num_tracks=2, ticks_per_quarter=480):
    """MIDI Header Chunk"""
    header = b'MThd'
    header += (6).to_bytes(4, 'big')  # Header length
    header += format_type.to_bytes(2, 'big')
    header += num_tracks.to_bytes(2, 'big')
    header += ticks_per_quarter.to_bytes(2, 'big')
    return header


def write_variable_length(value):
    """可変長数値エンコード"""
    result = bytearray()
    result.append(value & 0x7F)
    value >>= 7
    while value > 0:
        result.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(result)


def create_tempo_track(tempo_map, ticks_per_quarter=480, meter=(4, 4)):
    """Track 0: Conductor Track (Tempo + Time Signature)"""
    events = []
    
    # Time Signature (最初のみ)
    events.append((0, b'\xFF\x58\x04' + bytes([meter[0], (meter[1]//4).bit_length()-1, 24, 8])))
    
    # Tempo changes (可変テンポ)
    for bar, bpm in tempo_map:
        tick = bar * meter[0] * ticks_per_quarter
        microsec_per_quarter = int(60_000_000 / bpm)
        tempo_bytes = microsec_per_quarter.to_bytes(3, 'big')
        events.append((tick, b'\xFF\x51\x03' + tempo_bytes))
    
    # End of Track
    last_tick = max(e[0] for e in events) if events else 0
    events.append((last_tick + ticks_per_quarter, b'\xFF\x2F\x00'))
    
    # イベントをデルタタイムでエンコード
    events.sort(key=lambda e: e[0])
    track_data = bytearray()
    prev_tick = 0
    for tick, data in events:
        delta = tick - prev_tick
        track_data += write_variable_length(delta)
        track_data += data
        prev_tick = tick
    
    # Track Chunk
    chunk = b'MTrk'
    chunk += len(track_data).to_bytes(4, 'big')
    chunk += track_data
    return chunk


def create_vocal_track(notes, ticks_per_quarter=480, meter=(4, 4)):
    """Track 1: Vocal Melody + Lyrics"""
    events = []
    
    # Track Name
    track_name = "Vocal (Chorus)"
    events.append((0, b'\xFF\x03' + bytes([len(track_name)]) + track_name.encode('utf-8')))
    
    # Notes + Lyrics
    for note in notes:
        tick_on = note["tick_on"]
        tick_off = note["tick_off"]
        pitch = note["pitch"]
        velocity = note["velocity"]
        lyric = note.get("lyric", "la")
        
        # Lyric メタイベント (Note On直前)
        lyric_bytes = lyric.encode('utf-8')
        events.append((tick_on, b'\xFF\x05' + bytes([len(lyric_bytes)]) + lyric_bytes))
        
        # Note On
        events.append((tick_on, bytes([0x90, pitch, velocity])))
        
        # Note Off
        events.append((tick_off, bytes([0x80, pitch, 0])))
    
    # End of Track
    last_tick = max(e[0] for e in events) if events else 0
    events.append((last_tick + ticks_per_quarter, b'\xFF\x2F\x00'))
    
    # エンコード
    events.sort(key=lambda e: (e[0], len(e[1])))  # Lyric優先
    track_data = bytearray()
    prev_tick = 0
    for tick, data in events:
        delta = tick - prev_tick
        track_data += write_variable_length(delta)
        track_data += data
        prev_tick = tick
    
    # Track Chunk
    chunk = b'MTrk'
    chunk += len(track_data).to_bytes(4, 'big')
    chunk += track_data
    return chunk


def extract_chorus_anchors(anchors, sections):
    """chorus区間のアンカーのみ抽出"""
    chorus_bars = set()
    for i, s in enumerate(sections):
        if s["label"] == "chorus":
            start = s["bar"]
            end = sections[i + 1]["bar"] if i + 1 < len(sections) else 999999
            chorus_bars.update(range(start, end))
    
    return [a for a in anchors if a.get("bar", -1) in chorus_bars]


def anchors_to_notes(anchors, ticks_per_quarter=480, meter=4, base_pitch=60):
    """アンカー → MIDIノート列"""
    # stress アンカーのみ抽出（発音点）
    stress_anchors = [a for a in anchors if a.get("class") == "stress"]
    stress_anchors.sort(key=lambda a: a["time"])
    
    notes = []
    for i, anchor in enumerate(stress_anchors):
        time_ql = anchor["time_ql"]
        tick_on = int(time_ql * ticks_per_quarter)
        
        # 次のstressまでを音価に（最大1小節）
        if i + 1 < len(stress_anchors):
            next_time_ql = stress_anchors[i + 1]["time_ql"]
            duration_ql = min(next_time_ql - time_ql, meter)  # 最大1小節
        else:
            duration_ql = meter / 2  # 最後は2拍
        
        # sibilant減衰: 次のアンカーがsibilantなら末尾削除
        sibilant_trim = 0
        if i + 1 < len(anchors):
            next_anchor = anchors[i + 1]
            if next_anchor.get("class") == "sibilant" and next_anchor["time_ql"] < time_ql + duration_ql:
                # 20-40ms削る
                window_ms = next_anchor.get("window_ms", [20, 40])
                if isinstance(window_ms, dict):
                    trim_ms = (window_ms.get("pre", 20) + window_ms.get("post", 40)) / 2
                elif isinstance(window_ms, list):
                    trim_ms = sum(window_ms) / 2
                else:
                    trim_ms = 30
                trim_ql = (trim_ms / 1000) * (120 / 60)  # 120BPM基準
                duration_ql -= trim_ql
        
        duration_ql = max(0.25, duration_ql)  # 最短16分音符
        tick_off = int((time_ql + duration_ql) * ticks_per_quarter)
        
        # Pitch: 簡易メロディ（ランダムウォーク風）
        pitch = base_pitch + (i % 7) * 2  # Cメジャースケール風
        pitch = max(48, min(72, pitch))  # C3-C5範囲
        
        # Velocity: stressは強め
        velocity = 80
        
        # Lyric
        lyric = anchor.get("token", "la")
        
        notes.append({
            "tick_on": tick_on,
            "tick_off": tick_off,
            "pitch": pitch,
            "velocity": velocity,
            "lyric": lyric
        })
    
    return notes


def main():
    ap = argparse.ArgumentParser(description="Generate Chorus vocal MIDI with lyrics from enriched anchors")
    ap.add_argument("--anchors", required=True, help="lyric_anchors_enriched.json")
    ap.add_argument("--sections", required=True, help="sections_final.json")
    ap.add_argument("--tempo-json", required=True, help="tempo_map_multistem.json")
    ap.add_argument("--out", required=True, help="Output MIDI file (e.g., chorus_vocal.mid)")
    ap.add_argument("--ticks", type=int, default=480, help="Ticks per quarter note")
    ap.add_argument("--base-pitch", type=int, default=60, help="Base pitch (MIDI note number)")
    args = ap.parse_args()
    
    # 入力読み込み
    with open(args.anchors, encoding="utf-8") as f:
        anchors_data = json.load(f)
    
    with open(args.sections, encoding="utf-8") as f:
        sections_data = json.load(f)
    
    with open(args.tempo_json, encoding="utf-8") as f:
        tempo_data = json.load(f)
    
    anchors = anchors_data.get("anchors", [])
    sections = sections_data.get("sections", [])
    meter_num = anchors_data.get("meter", 4)
    
    # Tempo map準備
    if "tempo_map" in sections_data:
        tempo_map = sections_data["tempo_map"]
    else:
        # Fallback: 固定テンポ
        tempo_map = [[0, 75.0]]
    
    print(f"[INFO] Total anchors: {len(anchors)}")
    print(f"[INFO] Sections: {len(sections)}")
    
    # Chorus区間抽出
    chorus_anchors = extract_chorus_anchors(anchors, sections)
    print(f"[INFO] Chorus anchors: {len(chorus_anchors)}")
    
    if not chorus_anchors:
        print("[WARNING] No chorus anchors found. Creating empty MIDI.", file=sys.stderr)
        chorus_anchors = []
    
    # MIDIノート生成
    notes = anchors_to_notes(chorus_anchors, ticks_per_quarter=args.ticks, meter=meter_num, base_pitch=args.base_pitch)
    print(f"[INFO] MIDI notes: {len(notes)}")
    
    # MIDI生成
    header = create_midi_header(format_type=1, num_tracks=2, ticks_per_quarter=args.ticks)
    tempo_track = create_tempo_track(tempo_map, ticks_per_quarter=args.ticks, meter=(meter_num, 4))
    vocal_track = create_vocal_track(notes, ticks_per_quarter=args.ticks, meter=(meter_num, 4))
    
    # ファイル書き出し
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(header)
        f.write(tempo_track)
        f.write(vocal_track)
    
    print(f"[SUCCESS] Chorus vocal MIDI → {args.out}")
    print(f"  • Tracks: 2 (Conductor + Vocal)")
    print(f"  • Tempo map: {len(tempo_map)} change points")
    print(f"  • Notes: {len(notes)}")
    print(f"  • Lyrics: {'Placeholder (la)' if all(n['lyric'] == 'la' for n in notes) else 'Custom'}")


if __name__ == "__main__":
    main()
