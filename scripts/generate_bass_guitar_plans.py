#!/usr/bin/env python3
"""
generate_bass_guitar_plans.py
------------------------------
SongPackage + chordmap → bass_plan.json & guitar_plan.json生成

Bass: コードルート追従、オクターブ/5度ダブリング、Walking Bass
Guitar: コードストローク、アルペジオ、セクション別ダイナミクス

Usage:
    python3 scripts/generate_bass_guitar_plans.py \
      --song-dir song_packages/suno_project/song_001 \
      --config configs/arranger_weights.yaml \
      --emit-bass \
      --emit-guitar
"""
import argparse
import json
import yaml
import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Callable


# NOTE_MAP: C=0, C#/Db=1, ..., B=11
NOTE_MAP = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


def make_bar_locator(
    bars_df: pd.DataFrame, tempo_bpm: float
) -> Callable[[Dict], Tuple[int, float]]:
    """chordmap イベントから (bar_idx, beat_in_bar) を推定"""
    beats_per_bar = 4.0
    sec_per_beat = 60.0 / float(tempo_bpm)
    sec_per_bar = sec_per_beat * beats_per_bar

    starts_sec = None
    bar_indices = None
    if "start_sec" in bars_df.columns:
        sorted_df = bars_df.sort_values("bar_index")
        starts_sec = np.asarray(sorted_df["start_sec"].values, dtype=float)
        bar_indices = np.asarray(sorted_df["bar_index"].values, dtype=int)

    def _locate(ev: Dict) -> Tuple[int, float]:
        for key in ("bar", "bar_index"):
            if key in ev:
                try:
                    b = int(ev[key])
                    beat = float(ev.get("beat", 1.0))
                    return max(0, b), max(1.0, beat)
                except Exception:
                    pass

        time_val = None
        if "time_sec" in ev:
            time_val = float(ev["time_sec"])
        elif "time" in ev:
            time_ql = float(ev["time"])
            time_val = time_ql / (tempo_bpm / 60.0)

        if time_val is not None:
            if starts_sec is not None and bar_indices is not None:
                i = int(np.searchsorted(starts_sec, time_val, side="right") - 1)
                i = max(0, min(i, len(bar_indices) - 1))
                bar_idx = int(bar_indices[i])
                bar_start = starts_sec[i]
                beat = ((time_val - bar_start) / sec_per_beat) + 1.0
                return bar_idx, max(1.0, beat)
            else:
                bar_idx = int(time_val / sec_per_bar)
                beat = ((time_val % sec_per_bar) / sec_per_beat) + 1.0
                return bar_idx, beat

        return 0, 1.0

    return _locate


def generate_bass_plan(
    bars_df: pd.DataFrame,
    chordmap: List[Dict],
    config: Dict,
    tempo_bpm: float,
    sections_data: Optional[Dict] = None,
) -> Dict:
    """
    Bass Plan生成
    - コードルート追従
    - オクターブ/5度ダブリング
    - Walking Bass（オプション）
    - セクション別パターン変更
    """
    locate = make_bar_locator(bars_df, tempo_bpm)

    # 設定
    bass_cfg = config.get("roles", {}).get("bass", {})
    octave = bass_cfg.get("octave", 2)  # デフォルトC2 (MIDI 36)
    pattern = bass_cfg.get("pattern", "root_fifth")  # root_only, root_fifth, walking
    velocity_base = bass_cfg.get("velocity", {}).get("base", 90)

    # セクション情報（sections.jsonから）
    section_map = {}
    if sections_data and "sections" in sections_data:
        for sec in sections_data["sections"]:
            start_bar = sec.get("start_bar", 0)
            label = sec.get("label", "verse").lower()
            section_map[start_bar] = label

    events = []
    bar_chords = {}

    # chordmapからbar毎のコード取得
    for ev in chordmap:
        bar_idx, beat = locate(ev)
        root = ev.get("root", "C")
        quality = ev.get("quality", "")

        if bar_idx not in bar_chords:
            bar_chords[bar_idx] = []
        bar_chords[bar_idx].append({"beat": beat, "root": root, "quality": quality})

    # 各小節でBassパターン生成
    for bar_idx in sorted(bar_chords.keys()):
        chords_in_bar = bar_chords[bar_idx]

        # 小節内最初のコード使用（簡易実装）
        chord = chords_in_bar[0]
        root = chord["root"]
        root_pitch = 12 * octave + NOTE_MAP.get(root, 0)  # C2=36, D2=38, etc.

        # セクション判定（energy反映）
        energy = 0.5
        section_label = "verse"
        if bar_idx < len(bars_df):
            energy = bars_df.iloc[bar_idx].get("energy", 0.5)
            # section_map優先、なければbars.parquetのsection_label
            section_label = section_map.get(
                bar_idx, bars_df.iloc[bar_idx].get("section_label", "verse")
            )

        vel = int(velocity_base * (0.8 + 0.4 * energy))
        vel = max(60, min(127, vel))

        # セクション別パターン変更
        current_pattern = pattern
        if section_label in ("intro", "outro"):
            current_pattern = "root_only"  # イントロ/アウトロは控えめ
        elif section_label == "chorus":
            current_pattern = "walking"  # コーラスは動的

        if current_pattern == "root_only":
            # ルート音のみ（1拍目）
            events.append(
                {"bar": bar_idx, "beat": 1.0, "pitch": root_pitch, "dur_beats": 3.5, "vel": vel}
            )

        elif current_pattern == "root_fifth":
            # ルート + 5度（1拍目 + 3拍目）
            events.append(
                {"bar": bar_idx, "beat": 1.0, "pitch": root_pitch, "dur_beats": 1.5, "vel": vel}
            )
            events.append(
                {
                    "bar": bar_idx,
                    "beat": 3.0,
                    "pitch": root_pitch + 7,  # 5度上
                    "dur_beats": 1.5,
                    "vel": vel - 5,
                }
            )

        elif current_pattern == "walking":
            # Walking Bass（4分音符刻み）
            scale = [0, 2, 4, 5, 7, 9, 11]  # メジャースケール簡易
            for beat_offset in range(4):
                pitch_offset = random.choice(scale) if beat_offset > 0 else 0
                events.append(
                    {
                        "bar": bar_idx,
                        "beat": 1.0 + beat_offset,
                        "pitch": root_pitch + pitch_offset,
                        "dur_beats": 0.9,
                        "vel": vel - random.randint(0, 10),
                    }
                )

    return {
        "ppq": 480,
        "tempo_bpm": tempo_bpm,
        "tracks": [
            {
                "name": "Bass",
                "role": "bass",
                "channel": 1,
                "program": 33,  # Fingered Bass
                "events": events,
            }
        ],
    }


def generate_guitar_plan(
    bars_df: pd.DataFrame,
    chordmap: List[Dict],
    config: Dict,
    tempo_bpm: float,
    sections_data: Optional[Dict] = None,
) -> Dict:
    """
    Guitar Plan生成
    - コードストローク
    - アルペジオ
    - セクション別ダイナミクス（intro: カッティング, verse: ストラム, chorus: パワーコード）
    """
    locate = make_bar_locator(bars_df, tempo_bpm)

    # 設定
    guitar_cfg = config.get("roles", {}).get("guitar", {})
    octave = guitar_cfg.get("octave", 3)  # デフォルトC3 (MIDI 48)
    pattern = guitar_cfg.get("pattern", "strum")  # strum, arpeggio
    velocity_base = guitar_cfg.get("velocity", {}).get("base", 88)

    # セクション情報
    section_map = {}
    if sections_data and "sections" in sections_data:
        for sec in sections_data["sections"]:
            start_bar = sec.get("start_bar", 0)
            label = sec.get("label", "verse").lower()
            section_map[start_bar] = label

    # コード基本フォーム（トライアド）
    CHORD_SHAPES = {
        "": [0, 4, 7],  # Major
        "m": [0, 3, 7],  # Minor
        "7": [0, 4, 7, 10],  # Dominant 7th
        "maj7": [0, 4, 7, 11],  # Major 7th
        "m7": [0, 3, 7, 10],  # Minor 7th
        "dim": [0, 3, 6],  # Diminished
        "aug": [0, 4, 8],  # Augmented
        "sus4": [0, 5, 7],  # Suspended 4th
        "sus2": [0, 2, 7],  # Suspended 2nd
        "6": [0, 4, 7, 9],  # Major 6th
        "add9": [0, 4, 7, 14],  # Add 9
    }

    events = []
    bar_chords = {}

    # chordmapからbar毎のコード取得
    for ev in chordmap:
        bar_idx, beat = locate(ev)
        root = ev.get("root", "C")
        quality = ev.get("quality", "")

        if bar_idx not in bar_chords:
            bar_chords[bar_idx] = []
        bar_chords[bar_idx].append({"beat": beat, "root": root, "quality": quality})

    # 各小節でGuitarパターン生成
    for bar_idx in sorted(bar_chords.keys()):
        chords_in_bar = bar_chords[bar_idx]

        # セクション判定
        section_label = "verse"
        if bar_idx < len(bars_df):
            section_label = section_map.get(
                bar_idx, bars_df.iloc[bar_idx].get("section_label", "verse")
            )

        # セクション別奏法
        current_pattern = pattern
        stroke_pattern = [1.0, 2.5]  # デフォルト

        if section_label in ("intro", "outro"):
            current_pattern = "arpeggio"  # アルペジオ
        elif section_label == "verse":
            current_pattern = "strum"  # ストラム
            stroke_pattern = [1.0, 1.5, 2.5, 3.0, 3.5]  # カッティング
        elif section_label == "chorus":
            current_pattern = "strum"  # ストラム（強め）
            stroke_pattern = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # 16分刻み

        for chord in chords_in_bar:
            root = chord["root"]
            quality = chord["quality"]
            beat = chord["beat"]

            root_pitch = 12 * octave + NOTE_MAP.get(root, 0)

            # コード形状取得
            intervals = CHORD_SHAPES.get(quality, [0, 4, 7])
            pitches = [root_pitch + interval for interval in intervals]

            # セクション判定（energy反映）
            energy = 0.5
            if bar_idx < len(bars_df):
                energy = bars_df.iloc[bar_idx].get("energy", 0.5)

            vel = int(velocity_base * (0.8 + 0.4 * energy))
            vel = max(60, min(127, vel))

            if current_pattern == "strum":
                # ストローク（セクション別パターン）
                for beat_offset in stroke_pattern:
                    for pitch in pitches:
                        events.append(
                            {
                                "bar": bar_idx,
                                "beat": beat + beat_offset - 1.0,  # beat補正
                                "pitch": pitch,
                                "dur_beats": 0.25,
                                "vel": vel,
                            }
                        )

            elif current_pattern == "arpeggio":
                for pitch in pitches:
                    events.append(
                        {"bar": bar_idx, "beat": beat, "pitch": pitch, "dur_beats": 1.5, "vel": vel}
                    )

            elif pattern == "arpeggio":
                # アルペジオ（16分音符刻み）
                for i, pitch in enumerate(pitches):
                    events.append(
                        {
                            "bar": bar_idx,
                            "beat": beat + i * 0.25,
                            "pitch": pitch,
                            "dur_beats": 0.5,
                            "vel": vel - i * 5,
                        }
                    )

    return {
        "ppq": 480,
        "tempo_bpm": tempo_bpm,
        "tracks": [
            {
                "name": "Guitar",
                "role": "guitar",
                "channel": 2,
                "program": 25,  # Steel Guitar
                "events": events,
            }
        ],
    }


def main():
    ap = argparse.ArgumentParser(description="Bass/Guitar Plans生成")
    ap.add_argument("--song-dir", type=Path, required=True, help="song_packages/.../song_XXX")
    ap.add_argument("--config", type=Path, default=Path("configs/arranger_weights.yaml"))
    ap.add_argument("--emit-bass", action="store_true", help="bass_plan.json生成")
    ap.add_argument("--emit-guitar", action="store_true", help="guitar_plan.json生成")
    args = ap.parse_args()

    # 絶対パス化
    song_dir = args.song_dir.resolve()
    config_path = args.config.resolve()

    # 必須ファイル確認
    bars_path = song_dir / "bars.parquet"
    chordmap_path = song_dir / "chordmap.json"

    if not bars_path.exists():
        raise FileNotFoundError(f"bars.parquet not found: {bars_path}")
    if not chordmap_path.exists():
        raise FileNotFoundError(f"chordmap.json not found: {chordmap_path}")

    # sections.json読み込み（オプション）
    sections_path = song_dir / "sections.json"
    sections_data = None
    if sections_path.exists():
        sections_data = json.loads(sections_path.read_text(encoding="utf-8"))
        print(f"✅ Loaded sections: {sections_path}")
    else:
        print(f"⚠️  sections.json not found, using default patterns")

    # データロード
    bars_df = pd.read_parquet(bars_path)
    chordmap_data = json.loads(chordmap_path.read_text(encoding="utf-8"))

    # chordmapの形式確認（{"events": [...]} or直接配列）
    if isinstance(chordmap_data, dict) and "events" in chordmap_data:
        chordmap = chordmap_data["events"]
    else:
        chordmap = chordmap_data if isinstance(chordmap_data, list) else []

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    # テンポ取得（bars.parquetから、chordmapのtempo_bpmフィールドも確認）
    tempo_bpm = 120.0
    if len(bars_df) > 0 and "tempo_bpm" in bars_df.columns:
        tempo_bpm = bars_df.iloc[0].get("tempo_bpm", 120.0)
    elif isinstance(chordmap_data, dict) and "tempo_bpm" in chordmap_data:
        tempo_bpm = chordmap_data["tempo_bpm"]

    print(f"📦 Song: {song_dir.name}")
    print(f"   Bars: {len(bars_df)}, Chords: {len(chordmap)}, Tempo: {tempo_bpm:.2f} BPM")
    print()

    # Bass Plan生成
    if args.emit_bass:
        bass_plan = generate_bass_plan(bars_df, chordmap, config, tempo_bpm, sections_data)
        bass_path = song_dir / "bass_plan.json"
        bass_path.write_text(json.dumps(bass_plan, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✅ bass_plan.json: {len(bass_plan['tracks'][0]['events'])} events")

    # Guitar Plan生成
    if args.emit_guitar:
        guitar_plan = generate_guitar_plan(bars_df, chordmap, config, tempo_bpm, sections_data)
        guitar_path = song_dir / "guitar_plan.json"
        guitar_path.write_text(
            json.dumps(guitar_plan, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"✅ guitar_plan.json: {len(guitar_plan['tracks'][0]['events'])} events")

    print(
        f"\n🎉 Generated: {', '.join([n for n, f in [('bass_plan.json', args.emit_bass), ('guitar_plan.json', args.emit_guitar)] if f])}"
    )


if __name__ == "__main__":
    main()
