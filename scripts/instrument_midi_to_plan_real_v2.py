#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
instrument_midi_to_plan_real.py
--------------------------------
Bass / Guitar / Piano / Strings を「実グルーヴ → Plan」に落とす多機能スクリプト。

機能:
- 実MIDI or パターン推薦結果を節単位に展開（bars.parquet基準）
- chordmap.json に追従しつつ、移調・テンション付与（7/9/11(#11)/13）
- lyric_anchors.json によるボーカル衝突回避（オンセット抑制 or ベロシティ減衰）
- stems_features.parquet（drums_active/energy/hat_density 等）に連動した密度・アーティキュレーション調整
- セクション終止/遷移でのフィル/リック自動挿入
- 生成結果は Plan JSON 形式（midi_writer.py 互換）

想定入力:
    --role {bass|guitar|piano|strings}
    --song-package <song_dir>/song_package.yaml
    --bars <song_dir>/bars.parquet
    --chordmap <song_dir>/chordmap.json
    [--sections <song_dir>/sections.json]
    [--stems-features <song_dir>/stem_features.parquet]
    [--lyric-anchors <song_dir>/lyric_anchors.json]
    [--source-midi <pattern.mid>]  # リズム骨格のみ使用（音高は再配分）
    [--tension-policy auto|none]   # デフォルト auto
    [--out <path>]

出力:
    JSON Plan（例: bass_plan.json）
"""

from __future__ import annotations
import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pretty_midi
except Exception as e:
    pretty_midi = None
    print("⚠️ pretty_midi が見つかりません。--source-midi を使わない場合は動作します。", flush=True)


# ==============================
# ユーティリティ
# ==============================

NOTE_NAME_TO_PC = {
    "C":0,"B#":0,
    "C#":1,"Db":1,
    "D":2,
    "D#":3,"Eb":3,
    "E":4,"Fb":4,
    "F":5,"E#":5,
    "F#":6,"Gb":6,
    "G":7,
    "G#":8,"Ab":8,
    "A":9,
    "A#":10,"Bb":10,
    "B":11,"Cb":11
}

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def sec_to_beats(sec: float, bpm: float) -> float:
    return sec * bpm / 60.0

def beats_to_sec(beats: float, bpm: float) -> float:
    return beats * 60.0 / bpm

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def try_load_json(path: Optional[Path]) -> Optional[dict]:
    if path and path.exists():
        return load_json(path)
    return None

def resolve_song_paths(song_pkg_path: Path) -> Dict[str, Path]:
    import yaml
    pkg = yaml.safe_load(song_pkg_path.read_text(encoding="utf-8"))
    base = song_pkg_path.parent
    paths = pkg.get("paths", {})
    out = {
        "chordmap": base / paths.get("chordmap", "chordmap.json"),
        "sections": base / paths.get("sections", "sections.json"),
        "lyric_anchors": base / paths.get("lyric_anchors", "lyric_anchors.json"),
        "bars": base / paths.get("bars", "bars.parquet"),
        "stems_dir": base / paths.get("stems_dir", "stemswav_001"),
    }
    meta = pkg.get("meta", {})
    bpm = meta.get("bpm", meta.get("tempo_bpm", 120.0))
    out["bpm"] = bpm
    return out

def safe_read_parquet(p: Path) -> pd.DataFrame:
    # bars.parquet / stems_features.parquet 両対応（列名ゆらぎを吸収）
    df = pd.read_parquet(p)
    # bars系の列名を正規化
    rename_map = {}
    if "bar" in df.columns and "bar_index" not in df.columns:
        rename_map["bar"] = "bar_index"
    if "start_beat" in df.columns and "start_beats" not in df.columns:
        rename_map["start_beat"] = "start_beats"
    if "end_beat" in df.columns and "end_beats" not in df.columns:
        rename_map["end_beat"] = "end_beats"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

# ==============================
# Chord パーサ & Voicing
# ==============================

@dataclass
class ChordInfo:
    root_pc: int            # 0-11
    quality: str            # 'maj','min','dom','maj7','min7','dim','halfdim','sus4','sus2','aug','power'
    tension_flags: Dict[str, bool]  # {'9':True,'11':False,'#11':False,'13':True}

CHORD_RE = re.compile(r"^\s*([A-G][b#]?)(.*)\s*$", re.IGNORECASE)

def parse_chord_symbol(sym: str, mode_hint: str = "ionian") -> ChordInfo:
    """超簡易 Chord Symbol パーサ（music21を使わず依存レス）"""
    m = CHORD_RE.match(sym)
    if not m:
        # fallback: C
        return ChordInfo(0, "maj", {"9":False,"11":False,"#11":False,"13":False})
    root_name = m.group(1).capitalize()
    suffix = (m.group(2) or "").lower().strip()

    root_pc = NOTE_NAME_TO_PC.get(root_name, 0)
    q = "maj"
    if "sus4" in suffix:
        q = "sus4"
    elif "sus2" in suffix:
        q = "sus2"
    elif "maj7" in suffix or "ma7" in suffix or "△7" in suffix:
        q = "maj7"
    elif "m7b5" in suffix or "ø" in suffix or "half" in suffix:
        q = "halfdim"
    elif "dim" in suffix or "o" in suffix:
        q = "dim"
    elif "aug" in suffix or "+" in suffix:
        q = "aug"
    elif "7" in suffix:
        q = "dom"
    elif "m" in suffix and "maj" not in suffix:
        q = "min"
    elif "5" in suffix and "add" not in suffix:
        q = "power"  # 5th only

    # テンション推定（超簡易）
    tens = {"9":False,"11":False,"#11":False,"13":False}
    if q in ("maj7","maj"):
        tens["9"] = True
        if mode_hint in ("lydian", "lydian_dominant"):
            tens["#11"] = True
        else:
            # Major系の素の11は避けがち
            tens["11"] = False
        tens["13"] = True
    elif q in ("dom","aug"):
        tens["9"] = True
        tens["11"] = True
        tens["13"] = True
    elif q in ("min","min7"):
        tens["9"] = True
        tens["11"] = True
        tens["13"] = False
    elif q in ("sus4","sus2","power"):
        tens["9"] = True
        tens["11"] = False
        tens["13"] = True

    # 明示指定（add9, add13, 9, 11, #11, 13）
    for tkn in ("9","#11","11","13"):
        if tkn in suffix:
            key = tkn
            tens[key] = True

    return ChordInfo(root_pc, q, tens)

def chord_degrees(quality: str) -> List[int]:
    """0=root, 2=9, 4=3rd or sus2, 5=4th, 7=5th, 9=6th(13), 10=b7, 11=maj7"""
    if quality == "maj":
        return [0,4,7]
    if quality == "min":
        return [0,3,7]
    if quality == "dom":
        return [0,4,7,10]
    if quality == "maj7":
        return [0,4,7,11]
    if quality == "min7":
        return [0,3,7,10]
    if quality == "halfdim":
        return [0,3,6,10]
    if quality == "dim":
        return [0,3,6]
    if quality == "aug":
        return [0,4,8]
    if quality == "sus4":
        return [0,5,7]
    if quality == "sus2":
        return [0,2,7]
    if quality == "power":
        return [0,7]
    # fallback
    return [0,4,7]

def add_tensions(base: List[int], info: ChordInfo) -> List[int]:
    out = base[:]
    if info.tension_flags.get("9"):   out += [2]
    if info.tension_flags.get("#11"): out += [6]  # #11 as 6 semitone above 4? (PC 6 from root)
    elif info.tension_flags.get("11"):out += [5]
    if info.tension_flags.get("13"):  out += [9]
    # 重複を除去して上に伸ばす
    # （音高は後でオクターブ配分）
    uniq = []
    for d in out:
        if d not in uniq:
            uniq.append(d)
    return uniq

def degree_to_midi(root_pc: int, degree: int, register: int) -> int:
    """degree(0..11相当) を register基準（ミドルC=60付近を中心に）でMidiに"""
    pc = (root_pc + degree) % 12
    base = register
    # 最も近い同名音高まで持ち上げ
    while base % 12 != pc:
        base += 1
    return base

def build_voicing(info: ChordInfo, role: str, energy: float, mode_hint: str, tension_policy: str) -> List[int]:
    """
    役割別の基本ボイシング生成：
    - bass: ルート中心（+ オクターブ or 5th）
    - guitar: 3-7中心 + 9/13を状況付与
    - piano: 3-7-9-(13) など
    - strings: 広がり重視のパッド（root, 5th, 9, 13…）
    """
    base = chord_degrees(info.quality)
    if tension_policy == "auto":
        degs = add_tensions(base, info)
    else:
        degs = base

    # register 設定
    if role == "bass":
        reg = 40  # E2あたり
        # 低域は root + 5th or octave
        seq = [0, 7, 12]
        return [degree_to_midi(info.root_pc, d, reg) for d in seq]
    elif role == "guitar":
        reg = 52  # G3〜
        # 3-7 を中心に 9/13 を追加
        # energy 高いときは 13 を積極採用
        core = [4, 10 if "dom" in info.quality or info.quality in ("min7","halfdim") else 11]
        ext  = []
        if 2 in degs: ext.append(2)
        if energy >= 0.6 and 9 in degs: ext.append(9)
        if ("#11" in info.tension_flags and info.tension_flags["#11"]): ext.append(6)
        vo = core + ext
        if not vo:
            vo = degs[:3]
        return [degree_to_midi(info.root_pc, d, reg+i*3) for i,d in enumerate(vo)]
    elif role == "piano":
        reg = 60  # C4〜
        # 3-7-9-(13) のクローズド / 少しスプレッド
        seq = []
        if 4 in degs: seq.append(4)
        elif 3 in degs: seq.append(3)
        if 10 in degs: seq.append(10)
        elif 11 in degs: seq.append(11)
        if 2 in degs: seq.append(2)
        if energy >= 0.7 and 9 in degs: seq.append(9)
        if not seq:
            seq = degs[:4]
        return [degree_to_midi(info.root_pc, d, reg + (i*2)) for i,d in enumerate(seq)]
    else:  # strings
        reg = 55  # mid-low pad
        seq = []
        for d in (0, 7, 2, 9, 11):  # root,5,9,13,maj7 (優先)
            if d in degs:
                seq.append(d)
        if not seq:
            seq = degs[:3]
        return [degree_to_midi(info.root_pc, d, reg + (i*5)) for i,d in enumerate(seq)]


# ==============================
# Lyric Anchors 衝突回避
# ==============================

@dataclass
class AnchorWindow:
    start_beats: float
    end_beats: float

def load_anchor_windows(anchors_json: Optional[Path], bpm: float, bars_df: pd.DataFrame) -> List[AnchorWindow]:
    windows: List[AnchorWindow] = []
    if not anchors_json or not anchors_json.exists():
        return windows

    data = load_json(anchors_json)
    # 形式に応じて解釈：{anchors:[{start_sec,end_sec},...]} or notes with "time"
    anchors = data.get("anchors") or data.get("items") or []
    # bars.parquet に start_sec/end_sec があればそれを主とする
    has_sec = all(c in bars_df.columns for c in ("start_sec","end_sec"))
    # ボーカル有効時間を直接使う
    for a in anchors:
        if "start_sec" in a and "end_sec" in a:
            s = a["start_sec"]; e = a["end_sec"]
        elif "time" in a:  # QLやbeatsの可能性
            t = a["time"]  # QL想定
            s = beats_to_sec(t, bpm)  # 粗変換
            e = s + 0.2
        else:
            continue
        windows.append(AnchorWindow(sec_to_beats(s, bpm), sec_to_beats(e, bpm)))
    return windows

def onset_allowed(t_on: float, vocal_windows: List[AnchorWindow], margin_beats: float = 0.24) -> bool:
    """オンセットがボーカルに近すぎる場合は避ける（±margin）"""
    for w in vocal_windows:
        if (t_on >= w.start_beats - margin_beats) and (t_on <= w.end_beats + margin_beats):
            return False
    return True


# ==============================
# パターン骨格の取得（MIDI -> onsets/durations）
# ==============================

@dataclass
class PatternNote:
    t_on: float
    t_off: float
    vel: int

@dataclass
class PatternSkeleton:
    notes: List[PatternNote]

def midi_to_skeleton(mid_path: Path, bpm: float, beats_per_bar: float = 4.0) -> PatternSkeleton:
    if pretty_midi is None:
        # pretty_midiが無い環境向けのfallback（4分刻み）
        base = [PatternNote(t_on=0.0, t_off=1.0, vel=90)]
        return PatternSkeleton(notes=base)

    pm = pretty_midi.PrettyMIDI(str(mid_path))
    # 最初の非ドラムトラックを使用（無ければ最初のトラック）
    insts = [i for i in pm.instruments if not i.is_drum]
    if not insts:
        insts = pm.instruments[:1] if pm.instruments else []
    notes = []
    for inst in insts:
        for n in inst.notes:
            t_on_b = sec_to_beats(n.start, bpm)
            t_off_b = sec_to_beats(n.end, bpm)
            # 1小節内の相対値へ正規化（繰り返しに耐えやすく）
            t_on_mod = t_on_b % beats_per_bar
            dur = max(0.05, t_off_b - t_on_b)
            t_off_mod = min(beats_per_bar, t_on_mod + dur)
            notes.append(PatternNote(t_on=t_on_mod, t_off=t_off_mod, vel=int(n.velocity)))
    if not notes:
        # デフォルト: 8分刻み
        notes = [PatternNote(t_on=i*0.5, t_off=i*0.5+0.45, vel=90) for i in range(8)]
    # 密度過多は上限をかける
    notes = sorted(notes, key=lambda x: (x.t_on, -x.vel))
    return PatternSkeleton(notes=notes)


# ==============================
# 役割別の音高割り当て・フィル挿入
# ==============================

def assign_pitches_from_voicing(sk: PatternSkeleton, voicing: List[int], role: str) -> List[int]:
    """リズム骨格にボイシングを添わせる（簡易: 交互/アルペ）"""
    if not voicing:
        voicing = [60]
    pitches = []
    if role in ("guitar","piano","strings"):
        # アルペ or 交互
        for i, note in enumerate(sk.notes):
            pitches.append(voicing[i % len(voicing)])
    else:  # bass
        # root中心 + ときどき5th/oct
        for i, note in enumerate(sk.notes):
            if i % 4 == 3 and len(voicing) >= 3:
                pitches.append(voicing[2])  # octave
            elif i % 2 == 1 and len(voicing) >= 2:
                pitches.append(voicing[1])  # 5th
            else:
                pitches.append(voicing[0])  # root
    return pitches

def inject_fill_lick(events: List[dict], bar_idx: int, next_chord: ChordInfo, role: str, bar_start: float, bar_end: float, energy: float):
    """セクション終止/遷移や4小節ごとに軽いフィル/リックを足す（最後の 0.5beat に）"""
    space = max(0.12, min(0.6, 0.5 + 0.2*(energy-0.5)))
    start = max(bar_start, bar_end - space)
    end   = min(bar_end, start + (space * 0.9))
    if role == "bass":
        # next root へのクロマチックアプローチ
        tgt = degree_to_midi(next_chord.root_pc, 0, 43)  # G2近傍
        events.append({"bar":bar_idx,"start_beats":start,"end_beats":end,"pitch":tgt-1,"velocity":78})
    elif role == "guitar":
        # 低→高への小さなレイク
        root = degree_to_midi(next_chord.root_pc, 0, 52)
        seq = [root, root+4, root+7, root+9]
        dt = (end-start)/len(seq)
        t = start
        for p in seq:
            events.append({"bar":bar_idx,"start_beats":t,"end_beats":min(end,t+dt*0.9),"pitch":p,"velocity":74})
            t += dt
    elif role == "piano":
        # スケール内 4音ラン
        root = degree_to_midi(next_chord.root_pc, 0, 60)
        seq = [root+2, root+4, root+7, root+9]
        dt = (end-start)/len(seq)
        t = start
        for p in seq:
            events.append({"bar":bar_idx,"start_beats":t,"end_beats":min(end,t+dt*0.85),"pitch":p,"velocity":70})
            t += dt
    else:  # strings 控えめなスウェル
        p = degree_to_midi(next_chord.root_pc, 9, 57)  # 13th付近
        events.append({"bar":bar_idx,"start_beats":start,"end_beats":end,"pitch":p,"velocity":60})


# ==============================
# メイン：Plan 生成
# ==============================

def build_chord_per_bar(chordmap: dict, total_bars: int, beats_per_bar: float = 4.0, mode_hint: str = "ionian") -> List[ChordInfo]:
    evs = chordmap.get("events", [])
    unit = chordmap.get("unit", "QL")  # 'QL'想定
    # time を beats で扱う（QL=1なら beats=QL、4 QL=1小節）
    def ev_time_in_beats(ev):
        t = ev.get("time", 0.0)
        return float(t)  # QLをbeats扱い
    chord_infos = []
    for b in range(total_bars):
        bar_start = b * beats_per_bar
        # bar開始の直前までで一番新しいイベントを探す
        candidates = [ev for ev in evs if ev_time_in_beats(ev) <= bar_start + 1e-6]
        if candidates:
            sym = candidates[-1].get("symbol", "C")
        else:
            sym = "C"
        chord_infos.append(parse_chord_symbol(sym, mode_hint=mode_hint))
    return chord_infos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["bass","guitar","piano","strings"])
    ap.add_argument("--song-package", required=True, help="song_package.yaml")
    ap.add_argument("--bars", required=False, help="bars.parquet（未指定なら song_package から解決）")
    ap.add_argument("--chordmap", required=False, help="chordmap.json（未指定なら song_package から解決）")
    ap.add_argument("--sections", required=False, help="sections.json（任意）")
    ap.add_argument("--stems-features", required=False, help="stem_features.parquet（任意）")
    ap.add_argument("--lyric-anchors", required=False, help="lyric_anchors.json（任意）")
    ap.add_argument("--source-midi", required=False, help="パターンMIDI（リズム骨格のみ利用）")
    ap.add_argument("--tension-policy", default="auto", choices=["auto","none"])
    ap.add_argument("--beats-per-bar", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    song_pkg = Path(args.song_package)
    spaths = resolve_song_paths(song_pkg)
    bpm = float(spaths["bpm"])

    bars_path = Path(args.bars) if args.bars else spaths["bars"]
    chordmap_path = Path(args.chordmap) if args.chordmap else spaths["chordmap"]
    sections_path = Path(args.sections) if args.sections else spaths["sections"]
    lyric_path = Path(args.lyric_anchors) if args.lyric_anchors else spaths["lyric_anchors"]
    stems_path = Path(args.stems_features) if args.stems_features else (song_pkg.parent / "stem_features.parquet")

    # データ読込
    bars_df = safe_read_parquet(bars_path)
    if "bar_index" not in bars_df.columns or "start_beats" not in bars_df.columns or "end_beats" not in bars_df.columns:
        raise ValueError("bars.parquet に必要列（bar_index/start_beats/end_beats）がありません。")

    total_bars = int(bars_df["bar_index"].max() + 1)
    chordmap = load_json(chordmap_path)
    chord_per_bar = build_chord_per_bar(chordmap, total_bars, beats_per_bar=args.beats_per_bar,
                                        mode_hint=chordmap.get("mode","ionian"))

    # stems features
    drums_active = None
    energy = None
    if stems_path.exists():
        sfeat = safe_read_parquet(stems_path)
        if "bar_index" in sfeat.columns and "drums_active" in sfeat.columns:
            drums_active = sfeat.set_index("bar_index")["drums_active"]
        if "bar_index" in sfeat.columns and "energy" in sfeat.columns:
            energy = sfeat.set_index("bar_index")["energy"]

    # lyric anchors (beats)
    vocal_windows = load_anchor_windows(lyric_path if lyric_path.exists() else None, bpm=bpm, bars_df=bars_df)

    # パターン骨格
    if args.source_midi:
        skel = midi_to_skeleton(Path(args.source_midi), bpm=bpm, beats_per_bar=args.beats_per_bar)
    else:
        # デフォルト骨格（役割により変化）
        base = []
        if args.role == "bass":
            base = [PatternNote(t_on=i*0.5, t_off=i*0.5+0.45, vel=84) for i in range(8)]  # 8分
        elif args.role == "guitar":
            base = [PatternNote(t_on=i*0.5, t_off=i*0.5+0.35, vel=72) for i in range(8)]  # 8分ストラム基礎
        elif args.role == "piano":
            base = [PatternNote(t_on=i*0.5, t_off=i*0.5+0.48, vel=78) for i in range(8)]  # アルペ基礎
        else:  # strings
            base = [PatternNote(t_on=0.0, t_off=4.0, vel=60)]  # パッド
        skel = PatternSkeleton(notes=base)

    events: List[dict] = []
    beats_per_bar = args.beats_per_bar

    # 役割ごとに生成
    for _, row in bars_df.sort_values("bar_index").iterrows():
        b = int(row["bar_index"])
        bar_start = float(row["start_beats"])
        bar_end = float(row["end_beats"])
        chord_info = chord_per_bar[b] if b < len(chord_per_bar) else parse_chord_symbol("C")

        # エネルギー/ドラム有無
        bar_energy = float(energy[b]) if energy is not None and b in energy.index else 0.5
        is_break = bool(drums_active[b] == 0) if drums_active is not None and b in drums_active.index else False

        # ボイシング
        voi = build_voicing(chord_info, args.role, bar_energy, chordmap.get("mode","ionian"), args.tension_policy)

        # bar内のリズム骨格を複製し音高割り当て
        pitches = assign_pitches_from_voicing(skel, voi, args.role)

        # vocal回避用に、オンセット制御しつつ配置
        for i, note in enumerate(skel.notes):
            t_on = bar_start + note.t_on
            t_off = bar_start + min(beats_per_bar, max(note.t_off, note.t_on + 0.08))
            vel = int(clamp(note.vel * (0.9 + 0.2*bar_energy), 30, 118))
            pit = int(pitches[i % len(pitches)])

            # break中は密度を落とす
            if is_break and (i % 2 == 1) and args.role in ("guitar","piano","strings"):
                continue

            # ボーカル衝突回避：オンセット禁止 or ベロシティ減衰
            if not onset_allowed(t_on, vocal_windows, margin_beats=0.24):
                if args.role in ("guitar","piano"):
                    # 衝突時は発音しない
                    continue
                else:
                    # bass/strings は控えめベロシティ
                    vel = int(vel * 0.6)

            # 追加の安全
            if t_off <= t_on + 0.04:
                t_off = t_on + 0.08
                t_off = min(t_off, bar_end)

            events.append({
                "bar": b,
                "start_beats": float(t_on),
                "end_beats": float(min(bar_end, t_off)),
                "pitch": pit,
                "velocity": vel
            })

        # 4小節ごと or セクション境界で軽いフィル/リック
        is_transition = False
        if "section_label" in bars_df.columns:
            lab = row["section_label"]
            nxt_lab = None
            if b+1 < len(bars_df):
                nxt = bars_df.iloc[b+1]
                nxt_lab = nxt.get("section_label", lab)
            is_transition = (nxt_lab is not None) and (nxt_lab != lab)

        if (b % 4 == 3) or is_transition:
            next_ch = chord_per_bar[b+1] if (b+1) < len(chord_per_bar) else chord_info
            inject_fill_lick(events, b, next_ch, args.role, bar_start, bar_end, bar_energy)

    # Plan JSON へ
    role_name = args.role.title()
    plan = {
        "meta": {
            "role": args.role,
            "total_events": len(events),
            "tension_policy": args.tension_policy,
            "beats_per_bar": beats_per_bar
        },
        "tracks": [{
            "name": role_name,
            "role": args.role,
            "events": events
        }]
    }

    outp = Path(args.out)
    outp.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ {args.role}_plan.json written: {len(events)} events → {outp}")

if __name__ == "__main__":
    main()
