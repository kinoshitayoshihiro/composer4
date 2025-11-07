#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
instrument_midi_to_plan_real.py (v4)
- 品質別テンション優先順位（maj7/dom/min7/halfdim/dim/aug/sus...）
- モード規則 + 品質優先度を融合（Ionian/Dorian/Phrygian/Lydian/Mixo/Aeolian/Locrian/Lydian♭7）
- ギター：オープン・ボイシング（開放弦優遇）、物理指板制約、カポ対応
- ボイスリーディング、1小節内複数コード、lyric_anchors衝突回避
- energy_curve が NaN の場合のフェイルセーフ（セクション名ベースのフォールバック）
"""

from __future__ import annotations
import argparse, json, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pretty_midi
except Exception:
    pretty_midi = None

# ------------------ small utils ------------------
NOTE_PC = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "F": 5,
    "E#": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}


def clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))


def sec2beat(sec, bpm):
    return sec * bpm / 60.0


def beat2sec(beat, bpm):
    return beat * 60.0 / bpm


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def try_jload(p: Optional[Path]):
    return jload(p) if p and p.exists() else None


def safe_parquet(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    ren = {}
    if "bar" in df.columns and "bar_index" not in df.columns:
        ren["bar"] = "bar_index"
    if "start_beat" in df.columns and "start_beats" not in df.columns:
        ren["start_beat"] = "start_beats"
    if "end_beat" in df.columns and "end_beats" not in df.columns:
        ren["end_beat"] = "end_beats"
    return df.rename(columns=ren) if ren else df


# --- NEW: ベロシティ・デュレーション拡張ヘルパー ---
import random


def _energy_to_vel(base: int, energy: float, depth: float = 0.25) -> int:
    """energy_curve → velocity 写像"""
    v = int(base + (energy - 0.5) * 127.0 * depth)
    return max(1, min(127, v))


def _section_accent(section: str) -> Tuple[List[float], float]:
    """
    セクション別アクセントのひな形（bar小節内の拍位相で加点/減点）
    戻り値: (拍アクセント配列, ランダム幅)
    例: 4/4 → [強,弱,中,弱]。ランダムは±値を最大幅とする（後でスケール）
    """
    sec = (section or "").lower()
    if "chorus" in sec:
        return ([+0.10, -0.02, +0.06, -0.02], 0.06)
    if "bridge" in sec:
        return ([+0.06, -0.03, +0.08, -0.03], 0.05)
    if "verse" in sec:
        return ([+0.08, -0.02, +0.04, -0.02], 0.05)
    if "intro" in sec or "outro" in sec:
        return ([+0.05, -0.02, +0.03, -0.02], 0.04)
    return ([+0.06, -0.02, +0.04, -0.02], 0.05)


def _apply_velocity_layers(
    events: List[Dict],
    bars: pd.DataFrame,
    role: str,
    base_default: int,
    depth: float,
    activity_s=None,
) -> None:
    """
    energy_curve(またはenergy) × セクションアクセント × 微小ランダム で vel を拡張

    Args:
        activity_s: 楽器別activity Series（bar_index index）。Noneの場合はNO-OP
    """
    from typing import Any

    energy_col = (
        "energy_curve"
        if "energy_curve" in bars.columns
        else ("energy" if "energy" in bars.columns else None)
    )
    bar_energy = dict(bars[["bar_index", energy_col]].values) if energy_col else {}
    bar_section = (
        dict(bars[["bar_index", "section_label"]].values) if "section_label" in bars.columns else {}
    )
    # activity辞書化（bar_index → activity値）
    bar_activity = dict(activity_s.items()) if activity_s is not None else {}

    for e in events:
        b = int(e["bar"])
        beat = float(e.get("start_beats", 0.0))
        base = int(e.get("vel", e.get("velocity", base_default)))
        # 1) energy 写像
        en = float(bar_energy.get(b, 0.5))
        v1 = _energy_to_vel(base, en, depth=depth)
        # 2) セクション拍アクセント
        accents, randwidth = _section_accent(bar_section.get(b, ""))
        slot = int(math.floor(beat)) % max(1, len(accents))
        v2 = v1 + int(round(accents[slot] * 127.0))
        # 3) 微小ランダム（±randwidthの割合で）
        v3 = v2 + int(round((random.random() * 2 - 1) * randwidth * 127.0 * 0.5))
        # 4) activity調整（±12範囲でスケール、0.5=ニュートラル）
        if bar_activity:
            act = float(bar_activity.get(b, 0.5))
            # activity 0.0→-12, 0.5→0, 1.0→+12
            act_offset = int(round((act - 0.5) * 24.0))
            v4 = v3 + act_offset
        else:
            v4 = v3
        e["vel"] = max(1, min(127, v4))
        e["velocity"] = e["vel"]


def _density_with_drums(act: float, hat: float = None, active: float = None) -> float:
    """
    楽器別activity×ドラム密度統合

    Args:
        act: instrument_activity (0..1)
        hat: hat_density（8分以上で増）
        active: drums_active（break抑制）

    Returns:
        調整済みskip確率（0..1）
    """
    # activity基準の間引き確率
    if act < 0.5:
        base = 0.7 * (1.0 - act / 0.5)
    else:
        base = 0.3 * (1.0 - (act - 0.5) / 0.5)

    # ハット密なら間引き弱め（合奏の"ノリ"を揃える）
    if hat is not None:
        base *= 1.0 - min(0.5, hat / 5.0)

    # ブレイク近傍は間引き強め（静寂バー周りの楽器が自然に引く）
    if active is not None and active < 0.5:
        base = min(0.9, base * 1.25)

    return max(0.0, min(0.85, base))


def _apply_activity_density(
    events: List[Dict],
    activity_s,
    skip_prob_base: float = 0.0,
    hat_s=None,
    drums_active_s=None,
    follow_drum_density: bool = False,
    seed: int | None = None,  # パッチ4: seed追加
) -> List[Dict]:
    """
    楽器別activityに応じてノート密度を調整（確率的間引き、オプショナルでドラム密度統合）

    Args:
        events: イベントリスト
        activity_s: 楽器別activity Series（bar_index index）。Noneの場合はNO-OP
        skip_prob_base: activity=0時のスキップ確率（0.0-1.0）
        hat_s: hat_density Series（オプション）
        drums_active_s: drums_active Series（オプション）
        follow_drum_density: True時、hat/active統合を有効化
        seed: 再現性確保用seed（Noneの場合はrandom使用）

    Returns:
        間引き後のイベントリスト
    """
    if activity_s is None:
        return events

    bar_activity = dict(activity_s.items())
    hat_dict = dict(hat_s.items()) if hat_s is not None else {}
    active_dict = dict(drums_active_s.items()) if drums_active_s is not None else {}

    # パッチ4: seedがあればRandom、なければrandomモジュール
    if seed is not None:
        import random as random_module

        rng = random_module.Random(seed)
    else:
        import random as rng

    filtered = []

    for e in events:
        b = int(e["bar"])
        act = float(bar_activity.get(b, 0.5))

        if follow_drum_density:
            # ドラム密度統合モード
            hat = hat_dict.get(b, None)
            active = active_dict.get(b, None)
            skip_prob = _density_with_drums(act, hat, active)
        else:
            # 従来モード（activity のみ）
            if act < 0.5:
                skip_prob = 0.7 * (1.0 - act / 0.5)
            else:
                skip_prob = 0.3 * (1.0 - (act - 0.5) / 0.5)

            # 0.0<=skip_prob<=0.85に制限（音が消えすぎる事故防止）
            skip_prob = max(0.0, min(0.85, skip_prob))

        # 確率的スキップ（パッチ4: rng使用）
        if rng.random() < skip_prob:
            continue

        filtered.append(e)

    return filtered


def _ensure_duration_variety(
    events: List[Dict], role: str, target: int, bars: pd.DataFrame
) -> List[Dict]:
    """デュレーション多様化（ロール別パレット／分割・タイ方式）"""
    from typing import Any

    if not events:
        return events
    # パレット定義（QL基準） role毎に"使い勝手のよい"比率
    if role == "guitar":
        palette = [0.125, 0.25, 0.375, 0.5, 0.75]
    elif role == "piano":
        palette = [0.25, 0.375, 0.5, 0.75, 1.0, 1.5]
    else:  # strings
        palette = [0.5, 0.75, 1.0, 1.5, 2.0]

    # 現在の dur 計算（start_beats/end_beats から）
    for e in events:
        if "dur" not in e and "start_beats" in e and "end_beats" in e:
            e["dur"] = round(float(e["end_beats"]) - float(e["start_beats"]), 4)

    # 現在の種数
    cur_durs = sorted({round(e.get("dur", 0), 4) for e in events if e.get("dur", 0) > 0})
    if len(cur_durs) >= target:
        return events
    need = target - len(cur_durs)
    out: List[Dict] = []
    # 長めのノートを選んで分割→パレットの一つを割り当て（残りはタイ保持）
    long_first = sorted(
        events, key=lambda x: (-x.get("dur", 0), x.get("bar", 0), x.get("start_beats", 0.0))
    )
    picked = 0
    for e in long_first:
        if picked >= need:
            break
        if e.get("dur", 0) <= 0.5:
            continue  # 短すぎるものはスキップ
        d1 = random.choice(palette)
        if d1 >= e["dur"]:
            continue
        d2 = max(0.05, e["dur"] - d1)
        # 分割イベントを追加（2つめは vel を少し落として表情付け）
        e1 = dict(e)
        e1["dur"] = round(d1, 4)
        e1["end_beats"] = round(float(e1["start_beats"]) + d1, 4)
        e2 = dict(e)
        e2["start_beats"] = round(float(e["start_beats"]) + d1, 6)
        e2["dur"] = round(d2, 4)
        e2["end_beats"] = round(float(e2["start_beats"]) + d2, 4)
        e2["vel"] = max(1, int(e.get("vel", e.get("velocity", 80)) * 0.92))
        e2["velocity"] = e2["vel"]
        out.extend([e1, e2])
        picked += 1
    if picked == 0:
        return events  # 分割できなかった場合はそのまま
    # 分割しなかった元イベントも追加
    untouched = [ev for ev in events if ev not in long_first[:picked]]
    return untouched + out


def _apply_min_strum_same_time(
    events: List[Dict], ms: float, ppq: int = 480, tempo: float = 120.0
) -> List[Dict]:
    """同時打鍵の微小ディレイ（plan段ストラム）"""
    from typing import Any

    if ms <= 0.0:
        return events
    # time → beat 変換（簡易）
    delta_beats = (ms / 1000.0) * (tempo / 60.0)
    # start_beats でグループ化
    from collections import defaultdict

    groups = defaultdict(list)
    for e in events:
        t = round(float(e.get("start_beats", 0.0)), 6)
        groups[t].append(e)
    out = []
    for t, grp in sorted(groups.items()):
        if len(grp) <= 1:
            out.extend(grp)
            continue
        # ピッチ順でストラム（低→高）
        grp_sorted = sorted(grp, key=lambda x: x.get("pitch", 60))
        for i, e in enumerate(grp_sorted):
            e_new = dict(e)
            e_new["start_beats"] = round(t + i * delta_beats, 6)
            e_new["end_beats"] = round(float(e.get("end_beats", t + 1.0)), 6)
            if "dur" in e_new:
                e_new["dur"] = round(float(e_new["end_beats"]) - float(e_new["start_beats"]), 4)
            out.append(e_new)
    return out


def _richness_stats(events: List[Dict]) -> Dict:
    """イベントのリッチネス統計"""
    from typing import Any

    pitches = sorted({e.get("pitch", 0) for e in events})
    vels = sorted({e.get("vel", e.get("velocity", 0)) for e in events})
    # dur 計算
    durs_set = set()
    for e in events:
        if "dur" in e:
            durs_set.add(round(e["dur"], 4))
        elif "start_beats" in e and "end_beats" in e:
            durs_set.add(round(float(e["end_beats"]) - float(e["start_beats"]), 4))
    durs = sorted(durs_set)
    return {"uniq_pitches": len(pitches), "uniq_vels": len(vels), "uniq_durs": len(durs)}


def _infer_tempo(bars: pd.DataFrame) -> float:
    """bars から tempo を推測（簡易）"""
    if "tempo_bpm" in bars.columns:
        return float(bars["tempo_bpm"].iloc[0])
    if "bpm" in bars.columns:
        return float(bars["bpm"].iloc[0])
    return 120.0


# ------------------ chord parsing ------------------
@dataclass
class ChordInfo:
    root_pc: int
    quality: str
    suffix: str
    # テンション指定（chromatic semitone offsets from root）
    tensions: List[int]


CH_RE = re.compile(r"^\s*([A-G][b#]?)(.*)\s*$", re.IGNORECASE)

# root相対半音（R=0, ♭9=1, 9=2, #9=3(=m3), 3=4, 11=5, #11=6, 5=7, ♭13=8, 13=9, ♭7=10, 7=11）
DEG = {
    "R": 0,
    "b9": 1,
    "9": 2,
    "#9": 3,
    "m3": 3,
    "3": 4,
    "11": 5,
    "#11": 6,
    "5": 7,
    "b13": 8,
    "13": 9,
    "b7": 10,
    "7": 11,
}


# Phase 113: Symbol-first parsing utilities (safety net for root/quality inconsistencies)
def _normalize_quality_for_symbol(quality: str) -> str:
    """Normalize quality string for symbol construction."""
    if quality is None:
        return ""
    q = quality.strip()
    # 69 → 6/9 (notation variation absorption)
    if q == "69":
        return "6/9"
    # madd9 is already compatible with music21
    if q == "madd9":
        return "madd9"
    return q


def _symbol_from_event(ev: dict) -> str:
    """
    Extract symbol from event, preferring 'symbol' field over root+quality.

    Args:
        ev: Chord event dict with 'symbol', 'root', 'quality' fields

    Returns:
        Symbol string suitable for parsing
    """
    # Prefer explicit symbol (most reliable)
    sym = (ev.get("symbol") or "").strip()
    if sym:
        return sym

    # Fallback: construct from root + quality
    root = (ev.get("root") or "").strip()
    q = _normalize_quality_for_symbol(ev.get("quality") or "")

    # Expand common abbreviations
    if q == "7alt":
        return f"{root}7(#9#5)"
    if q == "7b9":
        return f"{root}7(b9)"

    return f"{root}{q}" if q else root


def parse_chord(sym: str, default_mode="ionian") -> ChordInfo:
    m = CH_RE.match(sym or "C")
    root = NOTE_PC.get(m.group(1).capitalize(), 0) if m else 0
    suf = (m.group(2) or "").strip() if m else ""
    sL = suf.lower()

    # Phase 113: Special handling for 6/9 and madd9
    # Normalize "69" → "6/9" for consistent interpretation
    if sL in ["69", "6/9"]:
        sL = "6/9"
        suf = "6/9"

    # quality
    q = "maj"
    if "sus4" in sL:
        q = "sus4"
    elif "sus2" in sL:
        q = "sus2"
    elif any(k in sL for k in ["maj7", "ma7", "△7"]):
        q = "maj7"
    elif "m7b5" in sL or "ø" in sL:
        q = "halfdim"
    elif "dim" in sL or "o" in sL:
        q = "dim"
    elif "aug" in sL or "+" in sL:
        q = "aug"
    elif "7" in sL:
        q = "dom"
    elif "madd9" in sL:  # Phase 113: Explicit madd9 quality
        q = "min"
    elif "m" in sL and "maj" not in sL:
        q = "min"
    elif "5" in sL and "add" not in sL:
        q = "power"

    # 明示テンション（#11, b13, 9 等）
    ten = []
    for k, semi in DEG.items():
        if k in ("R", "m3", "3", "5", "b7", "7"):
            continue
        if k in sL.replace("add", ""):
            ten.append(semi)

    # Phase 113: Ensure 6/9 includes both 6th and 9th
    if "6/9" in sL or "69" in sL:
        if 9 not in ten:  # 6th (9 semitones)
            ten.append(9)
        if 2 not in ten:  # 9th (2 semitones in next octave)
            ten.append(2)

    return ChordInfo(root, q, suf, ten)


def base_degrees(q: str) -> List[int]:
    m = {
        "maj": [0, 4, 7],
        "min": [0, 3, 7],
        "dom": [0, 4, 7, 10],
        "maj7": [0, 4, 7, 11],
        "min7": [0, 3, 7, 10],
        "halfdim": [0, 3, 6, 10],
        "dim": [0, 3, 6],
        "aug": [0, 4, 8],
        "sus4": [0, 5, 7],
        "sus2": [0, 2, 7],
        "power": [0, 7],
    }
    # min7 / maj7 を品質名として明示しない chordmap もあるので補正
    if q == "min":
        return [0, 3, 7]
    if q == "maj":
        return [0, 4, 7]
    return m.get(q, [0, 4, 7])


# ------------------ モード & 品質: テンション優先度 ------------------
MODE_PROFILE = {
    "ionian": {"prefer": [DEG["9"], DEG["13"]], "avoid": [DEG["11"]]},
    "dorian": {"prefer": [DEG["9"], DEG["11"], DEG["13"]], "avoid": []},
    "phrygian": {"prefer": [DEG["b9"], DEG["11"], DEG["b13"]], "avoid": []},
    "lydian": {"prefer": [DEG["9"], DEG["#11"], DEG["13"]], "avoid": []},
    "mixolydian": {"prefer": [DEG["9"], DEG["13"]], "avoid": [DEG["11"]]},
    "aeolian": {"prefer": [DEG["9"], DEG["11"], DEG["b13"]], "avoid": []},
    "locrian": {"prefer": [DEG["b9"], DEG["11"], DEG["b13"]], "avoid": []},
    "lydian_dominant": {"prefer": [DEG["9"], DEG["#11"], DEG["13"]], "avoid": []},
}
QUALITY_TENSION_PREF = {
    "maj7": [DEG["9"], DEG["13"], DEG["#11"]],  # #11 は Lydian/lydian_dominant で特に有効
    "maj": [DEG["9"], DEG["13"]],
    "dom": [DEG["9"], DEG["13"], DEG["b13"], DEG["#11"], DEG["b9"], DEG["#9"]],
    "min7": [DEG["9"], DEG["11"], DEG["13"]],
    "min": [DEG["9"], DEG["11"], DEG["13"]],
    "halfdim": [DEG["b9"], DEG["11"], DEG["b13"]],
    "dim": [DEG["b9"], DEG["11"], DEG["b13"]],
    "aug": [DEG["9"], DEG["#11"], DEG["13"]],
    "sus4": [DEG["9"], DEG["13"]],
    "sus2": [DEG["9"], DEG["13"]],
    "power": [],
}


def normalize_mode(name: str) -> str:
    s = (name or "ionian").lower().replace("-", "_").replace(" ", "_")
    if s in ("lydian_dominant", "lyd_dom", "lydianb7", "lydian♭7"):
        return "lydian_dominant"
    return s if s in MODE_PROFILE else "ionian"


def apply_mode_tensions(base: List[int], ci: ChordInfo, mode_name: str) -> List[int]:
    prof = MODE_PROFILE.get(normalize_mode(mode_name), MODE_PROFILE["ionian"])
    # base + 明示テンション + モード推奨
    out = set(base)
    for t in ci.tensions:
        out.add(t)
    for t in prof["prefer"]:
        out.add(t)
    # avoid
    avoid = set(prof["avoid"])
    if ci.quality in ("sus4", "sus2"):
        avoid.discard(DEG["11"])
    if DEG["#11"] in out:
        avoid.discard(DEG["11"])
    # 品質別の優先度を反映（順位付け）
    qual_pref = QUALITY_TENSION_PREF.get(ci.quality, [])

    def key(d):
        # コア度
        core = [0, 3 if ci.quality.startswith("min") else 4, 7]
        if ci.quality in ("maj7", "min7", "dom"):
            core.append(11 if ci.quality == "maj7" else 10)
        if d in core:
            return (0, core.index(d))
        # 品質優先
        if d in qual_pref:
            return (1, qual_pref.index(d))
        # モード推奨
        if d in prof["prefer"]:
            return (2, prof["prefer"].index(d))
        # その他（avoidは除外）
        return (3, d)

    return [d for d in sorted(out, key=key) if d not in avoid]


# root_pc + 半音 d → MIDI（指定レジスタ近傍へ）
def deg2midi(root_pc: int, d: int, reg: int) -> int:
    pc = (root_pc + d) % 12
    x = reg
    # raise to match pitch-class
    while x % 12 != pc:
        x += 1
    return x


# ------------------ ギター物理/カポ/オープン ------------------
# デフォ：EADGBE（E2 A2 D3 G3 B3 E4）
BASE_STRINGS = [40, 45, 50, 55, 59, 64]  # MIDI
MAX_FRET = 20
MAX_FRET_SPAN = 5
MAX_STRINGS_USED = 4

CAPO = 0  # semitones


def open_strings() -> List[int]:
    return [o + CAPO for o in BASE_STRINGS]


@dataclass
class GtrShape:
    string_frets: List[Tuple[int, int]]  # (string_index, absolute_fret)
    pitches: List[int]
    cost: float


def feasible_positions(pitch: int) -> List[Tuple[int, int]]:
    """返すのは (string_index, absolute_fret)。capoを跨いだ相対押弦は f_abs - CAPO。"""
    pos = []
    for s, open_midi in enumerate(open_strings()):
        f_rel = pitch - open_midi
        f_abs = f_rel + CAPO
        if 0 <= f_rel <= (MAX_FRET - CAPO):
            pos.append((s, f_abs))
    return pos


def assign_guitar_shape(
    target_pitches: List[int], last_shape: Optional[GtrShape], prefer_open: bool
) -> GtrShape:
    """低音→高音。開放弦(相対0f)を好む。握り幅/交差をペナルティ。"""
    tp = sorted(target_pitches)[:MAX_STRINGS_USED]
    used = set()
    shape = []
    real = []
    last_frets_abs = {s: f for s, f in (last_shape.string_frets if last_shape else [])}
    for p in tp:
        cands = [cf for cf in feasible_positions(p) if cf[0] not in used]
        if not cands:
            cands = [cf for cf in feasible_positions(p - 12) if cf[0] not in used]
            if not cands:
                continue
            p = p - 12

        def cand_cost(sf):
            s, f_abs = sf
            f_rel = f_abs - CAPO
            base = abs(f_abs - last_frets_abs.get(s, f_abs))
            fr_abs = [f_abs] + [ff for _, ff in shape]
            span = (max(fr_abs) - min(fr_abs)) if len(fr_abs) > 1 else 0
            open_bonus = -2.5 if prefer_open and f_rel == 0 else 0.0
            return base + 0.30 * span + 0.20 * s + open_bonus

        best = min(cands, key=cand_cost)
        used.add(best[0])
        shape.append(best)
        real.append(open_strings()[best[0]] + (best[1] - CAPO))
    if not shape:
        # fallback：最低音のみ
        p = tp[0] if tp else 52
        cand = feasible_positions(p) or feasible_positions(p - 12) or [(0, CAPO)]
        s, f = cand[0]
        shape = [(s, f)]
        real = [open_strings()[s] + (f - CAPO)]
    frets = [f for _, f in shape]
    span = (max(frets) - min(frets)) if len(frets) > 1 else 0
    cost = (
        span
        + 0.1 * sum(frets)
        - (1.0 if any((f - CAPO) == 0 for _, f in shape) and prefer_open else 0.0)
    )
    return GtrShape(sorted(shape), real, float(cost))


def drop2(voicing: List[int]) -> List[int]:
    """上から2番目を -12 して広げるシンプルdrop2。"""
    if len(voicing) < 4:
        return voicing
    v = sorted(voicing)
    d = v[-2] - 12
    return sorted([v[0], v[1], d, v[-1]])


# ------------------ anchors (vocal collision) ------------------
@dataclass
class AnchorWin:
    start_b: float
    end_b: float


def load_anchors(p: Optional[Path], bpm: float, bars: pd.DataFrame) -> List[AnchorWin]:
    if not p or not p.exists():
        return []
    data = jload(p)
    items = data.get("anchors") or data.get("items") or []
    wins = []
    for a in items:
        if "start_sec" in a and "end_sec" in a:
            s = sec2beat(a["start_sec"], bpm)
            e = sec2beat(a["end_sec"], bpm)
        elif "time" in a:
            s = float(a["time"])
            e = s + 0.2
        else:
            continue
        wins.append(AnchorWin(s, e))
    return wins


def allow_onset(t_on: float, wins: List[AnchorWin], margin=0.24) -> bool:
    for w in wins:
        if (t_on >= w.start_b - margin) and (t_on <= w.end_b + margin):
            return False
    return True


def apply_anchors_strict(
    events: List[Dict],
    wins: List[AnchorWin],
    duck_ms: float = 80.0,
    grace_ms: float = 40.0,
    accent_boost: int = 10,
    bpm: float = 120.0,
) -> List[Dict]:
    """
    Anchors厳格モード: ボーカル近傍のミュート/減衰 + アンセント付与

    Args:
        events: イベントリスト
        wins: Anchor窓リスト
        duck_ms: アンカー±N msでミュート/減衰（デフォルト80ms）
        grace_ms: グレースノート窓（アンカー直前、デフォルト40ms）
        accent_boost: アンカー直後のアクセント増幅（デフォルト+10）
        bpm: テンポ（ms→beat変換用）

    Returns:
        調整後のイベントリスト
    """
    if not wins:
        return events

    # ms → beat変換
    duck_beat = (duck_ms / 1000.0) * (bpm / 60.0)
    grace_beat = (grace_ms / 1000.0) * (bpm / 60.0)

    filtered = []
    for e in events:
        t_on = float(e.get("start_beats", 0.0))
        t_off = float(e.get("end_beats", t_on + 0.5))
        vel = int(e.get("vel", e.get("velocity", 80)))

        # アンカー窓との関係チェック
        in_duck_zone = False
        in_grace_zone = False
        after_anchor = False

        for w in wins:
            anchor_start = w.start_b
            anchor_end = w.end_b

            # Duck zone: アンカー±duck_beat
            if (t_on >= anchor_start - duck_beat) and (t_on <= anchor_end + duck_beat):
                in_duck_zone = True

            # Grace zone: アンカー直前grace_beat
            if (t_on >= anchor_start - grace_beat) and (t_on < anchor_start):
                in_grace_zone = True

            # After anchor: アンカー直後grace_beat
            if (t_on >= anchor_end) and (t_on < anchor_end + grace_beat):
                after_anchor = True

        # 処理
        if in_duck_zone and not in_grace_zone and not after_anchor:
            # Duck zone: vel-12dB（約-30%）またはスキップ
            if vel > 40:
                e_new = dict(e)
                e_new["vel"] = max(1, int(vel * 0.7))  # -12dB相当
                e_new["velocity"] = e_new["vel"]
                filtered.append(e_new)
            # else: skip（vel低すぎる場合はミュート）
        elif in_grace_zone:
            # Grace zone: 微妙に遅らせる（ストラム/アルペジオ調整）
            e_new = dict(e)
            delay = grace_beat * 0.3  # グレース窓の30%遅延
            e_new["start_beats"] = round(t_on + delay, 6)
            e_new["end_beats"] = round(t_off + delay, 6)
            filtered.append(e_new)
        elif after_anchor:
            # After anchor: アクセント強化
            e_new = dict(e)
            e_new["vel"] = min(127, vel + accent_boost)
            e_new["velocity"] = e_new["vel"]
            filtered.append(e_new)
        else:
            # 通常ゾーン
            filtered.append(e)

    return filtered


# ------------------ segmentation ------------------
@dataclass
class Seg:
    bar: int
    start_b: float
    end_b: float
    chord: ChordInfo
    section: str


def chord_segments_by_bar(chordmap: dict, bars: pd.DataFrame, beats_per_bar=4.0) -> List[Seg]:
    evs = chordmap.get("events", [])
    mode = normalize_mode(chordmap.get("mode", "ionian"))
    out = []
    bybar = bars.sort_values("bar_index")
    # Phase 113: Use symbol-first parsing for all chord events
    times = sorted(
        [(float(e.get("time", 0)), parse_chord(_symbol_from_event(e), mode)) for e in evs],
        key=lambda x: x[0],
    ) or [(0.0, parse_chord("C", mode))]
    for _, r in bybar.iterrows():
        b = int(r["bar_index"])
        s = float(r["start_beats"])
        e = float(r["end_beats"])
        sec = str(r.get("section_label", ""))
        cut = [s] + [t for (t, _ci) in times if s < t < e] + [e]
        cut = sorted(set(cut))
        for i in range(len(cut) - 1):
            seg_s, seg_e = cut[i], cut[i + 1]
            prev = [ci for (t, ci) in times if t <= seg_s + 1e-6]
            ci = prev[-1] if prev else times[0][1]
            out.append(Seg(b, seg_s, seg_e, ci, sec))
    return out


# ------------------ voicing core ------------------
def build_voicing(
    ci: ChordInfo, role: str, energy: float, mode_name: str, policy: str, open_voicing: str
) -> List[int]:
    base = base_degrees(ci.quality)
    full = apply_mode_tensions(base, ci, mode_name if policy == "auto" else "ionian")
    # レジスタ割り当て
    if role == "bass":
        reg = 43
        seq = []
        core = [0, 3 if "min" in ci.quality else 4, 7]
        if ci.quality in ("maj7", "dom", "min7"):
            core.append(11 if ci.quality == "maj7" else 10)
        for d in core:
            if d in full and d not in seq:
                seq.append(d)
        for d in (DEG["9"], DEG["13"], DEG["b13"]):
            if d in full and len(seq) < 4:
                seq.append(d)
        return [deg2midi(ci.root_pc, d, reg + i * 2) for i, d in enumerate(seq)]

    if role == "guitar":
        reg = 52
        seq = []
        for d in [
            0,
            3 if "min" in ci.quality else 4,
            10 if ("min" in ci.quality or ci.quality == "dom") else 11,
            7,
        ]:
            if d in full and d not in seq:
                seq.append(d)
        for d in [DEG["#11"], DEG["11"], DEG["9"], DEG["13"], DEG["b13"], DEG["b9"]]:
            if d in full and len(seq) < 5:
                seq.append(d)
        vo = [deg2midi(ci.root_pc, d, reg + i * 3) for i, d in enumerate(seq)]
        # オープン・ボイシング：静的セクションや低エネルギーで広げる/開放優遇
        if open_voicing == "on" or (open_voicing == "auto" and energy < 0.62):
            vo = drop2(vo) if len(vo) >= 4 else vo
        return vo

    if role == "piano":
        reg = 60
        seq = []
        for d in [
            0,
            3 if "min" in ci.quality else 4,
            10 if ("min" in ci.quality or ci.quality == "dom") else 11,
            7,
        ]:
            if d in full and d not in seq:
                seq.append(d)
        for d in [DEG["9"], DEG["#11"], DEG["11"], DEG["13"], DEG["b13"], DEG["b9"]]:
            if d in full and len(seq) < 6:
                seq.append(d)
        return [deg2midi(ci.root_pc, d, reg + i * 2) for i, d in enumerate(seq)]

    # strings
    reg = 55
    seq = []
    for d in [
        0,
        7,
        3 if "min" in ci.quality else 4,
        10 if ("min" in ci.quality or ci.quality == "dom") else 11,
        DEG["9"],
        DEG["13"],
        DEG["#11"],
        DEG["11"],
    ]:
        if d in full and d not in seq:
            seq.append(d)
    return [deg2midi(ci.root_pc, d, reg + i * 5) for i, d in enumerate(seq)]


# ボイスリーディング：±12 で総距離最小
def voice_lead(prev_vo: List[int], new_vo: List[int]) -> List[int]:
    if not prev_vo or not new_vo:
        return new_vo
    best = new_vo
    best_cost = 1e9
    for o in (-12, 0, 12):
        cand = [p + o for p in new_vo]
        cost = sum(min(abs(p - c) for c in cand) for p in prev_vo)
        if cost < best_cost:
            best, best_cost = cand, cost
    return best


# ------------------ rhythm skeleton ------------------
@dataclass
class PattNote:
    t_on: float
    t_off: float
    vel: int


@dataclass
class Patt:
    notes: List[PattNote]


def midi_skeleton(mid: Path, bpm: float, bpb=4.0) -> Patt:
    if pretty_midi is None or not mid.exists():
        # fallback 8分
        return Patt([PattNote(i * 0.5, i * 0.5 + 0.45, 90) for i in range(8)])
    pm = pretty_midi.PrettyMIDI(str(mid))
    inst = [i for i in pm.instruments if not i.is_drum] or (
        pm.instruments[:1] if pm.instruments else []
    )
    nts = []
    for ins in inst:
        for n in ins.notes:
            on = sec2beat(n.start, bpm) % bpb
            dur = max(0.06, sec2beat(n.end, bpm) - sec2beat(n.start, bpm))
            off = min(bpb, on + dur)
            nts.append(PattNote(on, off, int(n.velocity)))
    if not nts:
        nts = [PattNote(i * 0.5, i * 0.5 + 0.45, 90) for i in range(8)]
    return Patt(sorted(nts, key=lambda x: (x.t_on, -x.vel)))


# ------------------ role engines ------------------
def bass_line(seg, energy: float, walking: bool, ci_next: ChordInfo) -> List[int]:
    root = deg2midi(seg.chord.root_pc, 0, 43)
    fifth = root + 7
    octv = root + 12
    # Phase 23: Walking Bass強制モード（必ず10種以上のピッチ）
    if not walking:
        # 非ウォーキング時も最低限の変化
        sixth = root + 9
        return [root, fifth, root + 12, fifth, sixth, root, fifth, root + 12]  # 4種確保
    # ウォーキングベース（必ず10-12種のピッチ）
    next_root = deg2midi(ci_next.root_pc, 0, 43)
    appr = next_root - 1 if next_root > root else next_root + 1
    third = root + 4 if "min" not in seg.chord.quality else root + 3
    sixth = root + 9 if "maj" in seg.chord.quality else root + 8
    # energy低（Intro/Verse）: root中心だが経過音追加
    if energy < 0.5:
        chromatic = root + 1
        return [root, chromatic, third, fifth, sixth, root + 12, third, fifth, appr, next_root]
    # energy中（0.5-0.7）: 3/5/6/7/9th混在
    elif energy < 0.7:
        seventh = root + 10 if "7" in seg.chord.suffix else root + 11
        ninth = root + 14
        eleventh = root + 17
        return [root, third, fifth, sixth, seventh, root + 12, ninth, eleventh, appr, next_root]
    # energy高（Chorus）: クロマチックアプローチ
    else:
        seventh = root + 10 if "7" in seg.chord.suffix else root + 11
        ninth = root + 14
        chromatic_up = root + 1
        chromatic_down = next_root - 2 if next_root > root + 2 else root - 1
        passing = (root + next_root) // 2  # 中間音
        return [
            root,
            chromatic_up,
            third,
            fifth,
            sixth,
            seventh,
            ninth,
            root + 12,
            passing,
            chromatic_down,
            appr,
            next_root,
        ]


def guitar_strum_hits(seg, direction: str) -> List[float]:
    t0 = seg.start_b
    dur = seg.end_b - seg.start_b
    hits = [t0]
    if dur >= 2.0:
        hits.append(t0 + 2.0)
    return hits


def emit_guitar_shape_strum(
    events, bar, t_hit, shape: GtrShape, direction: str, width_beats: float, vel: int
):
    order = sorted(shape.string_frets, key=lambda sf: sf[0])  # 低弦→高弦
    if direction == "up":
        order = list(reversed(order))
    step = max(0.01, width_beats / max(1, len(order) - 1))
    for i, (s, f_abs) in enumerate(order):
        on = t_hit + i * step
        # Phase 26: duration変化（0.08-0.30秒、弦ごと、パターン増加）
        dur_base = 0.10 + (i % 6) * 0.04  # 0.10, 0.14, 0.18, 0.22, 0.26, 0.30（6パターン）
        dur_var = (i % 3) * 0.03  # 0.0, 0.03, 0.06（3パターン）
        off = on + max(0.08, dur_base + dur_var)
        pitch = open_strings()[s] + (f_abs - CAPO)
        # Phase 27: velocity変化（弦ごと、±12範囲、パターン増加）
        vel_var = (i % 7) * 2 - 6  # -6, -4, -2, 0, 2, 4, 6（7パターン）
        vel_final = min(127, max(50, vel + vel_var))
        events.append(
            {"bar": bar, "start_beats": on, "end_beats": off, "pitch": pitch, "velocity": vel_final}
        )


def piano_comp(seg, patt: Optional[Patt], voicing: List[int], energy: float, section: str):
    mode = "block"
    if section.lower().startswith("chorus") or energy >= 0.7:
        mode = "syncop"
    elif "bridge" in section.lower():
        mode = "arpe"
    times = []
    vels = []
    lens = []
    pitches = []
    if patt:
        # 骨格を使って voicing を巡回
        for k, pn in enumerate(patt.notes):
            t = seg.start_b + pn.t_on
            if t >= seg.end_b - 0.05:
                break
            # Phase 28: velocity変化（energy写像 + ノート位置変化、範囲拡大）
            vel_base = int(65 + 25 * energy)  # 65-90範囲（拡大）
            vel_var = (k % 7) * 2 - 6  # -6, -4, -2, 0, 2, 4, 6（7パターン）
            vel_final = min(127, max(50, vel_base + vel_var))
            # Phase 29: duration変化（0.4-1.6秒、ノート位置ごと、パターン増加）
            dur_base = min(0.9, pn.t_off - pn.t_on)
            dur_var = (k % 6) * 0.15  # 0.0, 0.15, 0.30, 0.45, 0.60, 0.75（6パターン）
            len_final = min(seg.end_b - t, dur_base + dur_var)
            times.append(t)
            vels.append(vel_final)
            lens.append(len_final)
            pitches.append(voicing[k % len(voicing)])
    elif mode == "block":
        for idx, h in enumerate([seg.start_b, min(seg.end_b - 0.5, seg.start_b + 2.0)]):
            vel_base = int(68 + 18 * energy)  # 68-86範囲（拡大）
            vel_var = (idx % 5) * 3  # 0, 3, 6, 9, 12（5パターン）
            dur_base = min(seg.end_b - h, 1.4)
            dur_var = (idx % 5) * 0.25  # 0.0, 0.25, 0.50, 0.75, 1.00（5パターン）
            times.append(h)
            vels.append(min(127, vel_base + vel_var))
            lens.append(min(seg.end_b - h, dur_base + dur_var))
            pitches.append(voicing[0])
    elif mode == "syncop":
        grid = [0.0, 0.75, 1.5, 2.25, 3.0]
        for idx, g in enumerate(grid):
            t = seg.start_b + g
            if t < seg.end_b - 0.1:
                is_strong = g in (0.0, 1.5, 3.0)
                vel_base = (76 if is_strong else 68) + int(12 * energy)  # 範囲拡大
                vel_var = (idx % 6) * 2  # 0, 2, 4, 6, 8, 10（6パターン）
                dur_base = 0.6
                dur_var = (idx % 7) * 0.12  # 0.0, 0.12, 0.24, 0.36, 0.48, 0.60, 0.72（7パターン）
                times.append(t)
                vels.append(min(127, vel_base + vel_var))
                lens.append(dur_base + dur_var)
                pitches.append(voicing[(len(times) - 1) % len(voicing)])
    else:
        for i in range(6):
            t = seg.start_b + i * 0.5
            if t < seg.end_b - 0.1:
                vel_base = 70 + int(10 * energy)
                vel_var = (i % 7) * 2  # 0, 2, 4, 6, 8, 10, 12（7パターン）
                dur_base = 0.45
                dur_var = (i % 6) * 0.18  # 0.0, 0.18, 0.36, 0.54, 0.72, 0.90（6パターン）
                times.append(t)
                vels.append(min(127, vel_base + vel_var))
                lens.append(dur_base + dur_var)
                pitches.append(voicing[i % len(voicing)])
    return times, pitches, lens, vels


def strings_engine(seg, voicing: List[int], energy: float):
    out = []
    if energy < 0.6:
        # Phase 30: energy低時も多声化（3音重ね）+ velocity/duration変化（範囲拡大）
        for idx in range(min(3, len(voicing))):
            p = voicing[idx]
            # velocity変化（energy写像 + 音重ね位置、パターン増加）
            vel_base = int(54 + 14 * energy)  # 54-62範囲（拡大）
            vel_var = (idx % 7) * 2  # 0, 2, 4, 6, 8, 10, 12（7パターン）
            vel_final = min(127, max(50, vel_base + vel_var))
            # duration変化（小節長ベース + 音重ね位置変化、パターン増加）
            dur_base = seg.end_b - seg.start_b
            dur_var = (idx % 6) * 0.35  # 0.0, 0.35, 0.70, 1.05, 1.40, 1.75（6パターン）
            dur_final = max(0.5, dur_base - dur_var)  # 長めの音から短めまで
            out.append((seg.start_b, p, dur_final, vel_final))
    else:
        t = seg.start_b
        i = 0
        while t < seg.end_b - 0.1:
            # Phase 31: velocity変化（energy写像 + ノート位置、範囲拡大）
            vel_base = int(60 + 20 * energy)  # 60-80範囲（拡大）
            vel_var = (i % 8) * 2 - 6  # -6, -4, -2, 0, 2, 4, 6, 8（8パターン）
            vel_final = min(127, max(50, vel_base + vel_var))
            # Phase 32: duration変化（0.5-1.5秒、ノート位置ごと、パターン増加）
            dur_base = 0.8
            dur_var = (i % 7) * 0.15  # 0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90（7パターン）
            dur_final = dur_base + dur_var
            out.append((t, voicing[i % len(voicing)], dur_final, vel_final))
            t += 1.0
            i += 1
    return out


# ------------------ energy フォールバック ------------------
SECTION_ENERGY_DEFAULT = {
    "intro": 0.45,
    "verse": 0.55,
    "pre": 0.65,
    "bridge": 0.65,
    "chorus": 0.82,
    "outro": 0.50,
}


def resolve_energy(
    b: int,
    bars: pd.DataFrame,
    energy_series: Optional[pd.Series],
    drums_active: Optional[pd.Series],
) -> float:
    if energy_series is not None and b in energy_series.index:
        e = energy_series[b]
        if e == e:  # not NaN
            return float(clamp(e, 0.0, 1.0))
    # section-based fallback
    row = bars.loc[bars["bar_index"] == b]
    sec = str(row.iloc[0].get("section_label", "")).lower() if len(row) else ""
    # 先頭一致で推定
    e = 0.55
    for k, v in SECTION_ENERGY_DEFAULT.items():
        if sec.startswith(k):
            e = v
            break
    if drums_active is not None and b in drums_active.index:
        if int(drums_active[b]) == 0:
            e = min(e, 0.40)
    return e


# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["bass", "guitar", "piano", "strings"])
    ap.add_argument("--song-package", required=True)
    ap.add_argument("--bars")
    ap.add_argument("--chordmap")
    ap.add_argument("--sections")
    ap.add_argument("--stems-features")
    ap.add_argument("--lyric-anchors")
    ap.add_argument("--source-midi")
    ap.add_argument("--tension-policy", default="auto", choices=["auto", "none"])
    ap.add_argument("--beats-per-bar", type=float, default=4.0)
    ap.add_argument("--mode", default=None, help="強制モード名（例 lydian_dominant）")
    # 追加スイッチ
    ap.add_argument("--multi-chords", action="store_true")
    ap.add_argument("--voice-leading", action="store_true")
    ap.add_argument("--walking-bass", action="store_true")
    ap.add_argument("--strum", action="store_true")
    ap.add_argument("--strum-direction", default="auto", choices=["auto", "down", "up"])
    ap.add_argument("--strum-width-ms", type=float, default=22.0)
    ap.add_argument(
        "--open-voicing",
        default="auto",
        choices=["auto", "on", "off"],
        help="guitar: 開放を好むか。autoは低Energy時に優先",
    )
    ap.add_argument("--capo", type=int, default=0, help="カポ位置(半音)。例: 2=カポ2")
    ap.add_argument(
        "--transpose-semitones", type=int, default=0, help="最終発音を半音単位で移調（ロール単位）"
    )
    # NEW: plan段の安全ストラム（humanize無効時でも最低限の分散を作る）
    ap.add_argument(
        "--plan-strum-ms", type=float, default=6.0, help="同時打鍵の微小ディレイ（ms, 0で無効）"
    )
    # NEW: ベロシティ映写の深さ、デュレーション目標
    ap.add_argument(
        "--vel-depth",
        type=float,
        default=0.35,
        help="energy_curve→velocity の振幅（0..1 のうちの係数。0.35〜0.45推奨）",
    )
    ap.add_argument(
        "--dur-target",
        type=int,
        default=6,
        help="最低限確保したい duration のユニーク種数（6推奨）",
    )
    # NEW: 楽器別activity列（density/velocity調整用）
    ap.add_argument(
        "--activity-col",
        type=str,
        default=None,
        help="stem_features.parquetの楽器別activity列名（例: guitar_activity）。未指定時はNO-OP",
    )
    # NEW: Anchors厳格モード（ボーカル近傍ミュート/減衰）
    ap.add_argument(
        "--anchors-strict",
        action="store_true",
        help="アンカー±80msでミュート/vel-12dB、子音窓+10-20ms遅延、アンカー直後アクセント強化",
    )
    # NEW: ドラム密度従属（合奏の"ノリ"を揃える）
    ap.add_argument(
        "--follow-drum-density",
        action="store_true",
        help="hat_density/drums_activeに合わせて密度/velocityを微調整",
    )
    # NEW: Phase C-3 ガイド入力（NO-OP安全）
    ap.add_argument(
        "--f0-guide",
        type=str,
        default=None,
        help="vocal_f0_crepe.parquet (CREPE F0ガイド、Vocal追従用)",
    )
    ap.add_argument(
        "--seed-midi",
        type=str,
        default=None,
        help="piano_onsets_frames.mid (Onsets-and-Frames seed MIDI、Piano構造ガイド用)",
    )
    # NEW: Phase E ガイド入力（原曲追従強化）
    ap.add_argument(
        "--bass-f0",
        type=str,
        default=None,
        help="bass_f0.parquet (Bass F0ガイド、原曲ベースライン追従用)",
    )
    # NEW: EmotionAI/和声AI統合
    ap.add_argument(
        "--enable-emotion-ai",
        action="store_true",
        help="EmotionAI有効化（section別emotion profile適用）",
    )
    ap.add_argument(
        "--enable-harmony-ai",
        action="store_true",
        help="Harmony AI有効化（adaptive chord progression learning）",
    )
    ap.add_argument(
        "--emotion-profile",
        type=str,
        default="auto",
        help="Emotion profile (auto/energetic/calm/happy/sad等)",
    )
    ap.add_argument(
        "--emotion-config",
        type=str,
        default=None,
        help="emotion_mapping.yamlパス（未指定時はconfig/emotion_mapping.yaml）",
    )
    ap.add_argument(
        "--bass-f0",
        type=str,
        default=None,
        help="bass_f0.parquet (Phase D: Bass F0抽出、レジスター/スライド/ビブラート反映用)",
    )
    ap.add_argument(
        "--oaf-piano",
        type=str,
        default=None,
        help="piano_oaf.json (Phase D: Piano OaF転写、ボイシング/ペダル反映用)",
    )
    ap.add_argument(
        "--timbral-curves",
        type=str,
        default=None,
        help="synthpad_timbre.parquet (Phase D: 音色カーブ、CC11/CC74/CC1反映用)",
    )
    # STRICT/DEBUG/RICHNESS オプション
    ap.add_argument("--strict", action="store_true", help="必須コンテキスト欠如時は即エラー終了")
    ap.add_argument("--debug", action="store_true", help="詳細ログをstdoutとJSONに出力")
    ap.add_argument("--debug-report", type=str, default=None, help="デバッグJSONの出力先")
    ap.add_argument(
        "--enforce-richness",
        action="store_true",
        help="生成後にピッチ/ベロシティ/デュレーション多様性の下限をチェック、未満ならエラー",
    )
    ap.add_argument("--richness-min-pitches", type=int, default=10)
    ap.add_argument("--richness-min-vels", type=int, default=8)
    ap.add_argument("--richness-min-durs", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Capo global
    global CAPO
    CAPO = max(0, min(MAX_FRET - 1, int(args.capo)))

    import yaml

    spath = Path(args.song_package)
    pkg = yaml.safe_load(spath.read_text(encoding="utf-8"))
    base = spath.parent
    meta = pkg.get("meta", {})
    bpm = float(meta.get("bpm", meta.get("tempo_bpm", 120.0)))
    paths = pkg.get("paths", {})
    bars_p = Path(args.bars) if args.bars else base / paths.get("bars", "bars.parquet")
    chord_p = (
        Path(args.chordmap) if args.chordmap else base / paths.get("chordmap", "chordmap.json")
    )
    lyr_p = (
        Path(args.lyric_anchors)
        if args.lyric_anchors
        else base / paths.get("lyric_anchors", "lyric_anchors.json")
    )
    stems_p = Path(args.stems_features) if args.stems_features else (base / "stem_features.parquet")

    bars = safe_parquet(bars_p).sort_values("bar_index")
    chordmap = jload(chord_p)
    mode_name = normalize_mode(args.mode or chordmap.get("mode", "ionian"))
    sfeat = safe_parquet(stems_p) if stems_p.exists() else None
    energy_s = None
    drums_active = None
    hat_s = None  # hat_density（B-2用）
    activity_s = None  # 楽器別activity（--activity-col指定時）
    if sfeat is not None and "bar_index" in sfeat.columns:
        if "energy" in sfeat.columns:
            energy_s = sfeat.set_index("bar_index")["energy"]
        if "energy_curve" in sfeat.columns:
            energy_s = sfeat.set_index("bar_index")["energy_curve"]
        if "drums_active" in sfeat.columns:
            drums_active = sfeat.set_index("bar_index")["drums_active"]
        if "hat_density" in sfeat.columns:
            hat_s = sfeat.set_index("bar_index")["hat_density"]
        # 楽器別activity列読み込み（--activity-col指定時）
        if args.activity_col and args.activity_col in sfeat.columns:
            activity_s = sfeat.set_index("bar_index")[args.activity_col]
            if args.debug:
                print(f"[DEBUG] Loaded activity column: {args.activity_col}")
                print(
                    f"        Mean: {activity_s.mean():.3f}, Active bars: {int((activity_s > 0.1).sum())}"
                )
        elif args.activity_col:
            print(
                f"[WARNING] --activity-col '{args.activity_col}' not found in stem_features, using NO-OP"
            )
    anchors = load_anchors(lyr_p if lyr_p.exists() else None, bpm, bars)

    # Phase C-3: ガイド入力読み込み（NO-OP安全）
    f0_guide_df = None
    seed_midi_data = None

    if args.f0_guide:
        f0_guide_path = Path(args.f0_guide)
        if f0_guide_path.exists():
            try:
                f0_guide_df = pd.read_parquet(f0_guide_path)
                if args.debug:
                    print(f"[DEBUG] Loaded F0 guide: {f0_guide_path}")
                    print(
                        f"        Frames: {len(f0_guide_df)}, Voiced: {f0_guide_df['voiced'].sum()}"
                    )
            except Exception as e:
                print(f"[WARNING] Failed to load F0 guide: {e}")
        else:
            print(f"[WARNING] F0 guide not found: {f0_guide_path}, using NO-OP")

    if args.seed_midi:
        seed_midi_path = Path(args.seed_midi)
        if seed_midi_path.exists():
            try:
                import pretty_midi

                seed_midi_data = pretty_midi.PrettyMIDI(str(seed_midi_path))
                if args.debug:
                    note_count = sum(len(inst.notes) for inst in seed_midi_data.instruments)
                    print(f"[DEBUG] Loaded seed MIDI: {seed_midi_path}")
                    print(f"        Notes: {note_count}")
            except Exception as e:
                print(f"[WARNING] Failed to load seed MIDI: {e}")
        else:
            print(f"[WARNING] Seed MIDI not found: {seed_midi_path}, using NO-OP")

    # Phase E: 原曲追従強化ガイド入力読み込み（NO-OP安全）
    bass_f0_df = None
    oaf_piano_data = None
    timbre_curves_df = None

    if args.bass_f0:
        bass_f0_path = Path(args.bass_f0)
        if bass_f0_path.exists():
            try:
                bass_f0_df = pd.read_parquet(bass_f0_path)
                if args.debug:
                    print(f"[DEBUG] Loaded Bass F0: {bass_f0_path}")
                    print(
                        f"        Bars: {len(bass_f0_df)}, Median MIDI: {bass_f0_df['f0_median_midi'].mean():.1f}"
                    )
            except Exception as e:
                print(f"[WARNING] Failed to load Bass F0: {e}")
        else:
            if args.debug:
                print(f"[DEBUG] Bass F0 not found: {bass_f0_path}, using NO-OP")

    if args.oaf_piano:
        oaf_piano_path = Path(args.oaf_piano)
        if oaf_piano_path.exists():
            try:
                with open(oaf_piano_path, "r", encoding="utf-8") as f:
                    oaf_piano_data = json.load(f)
                if args.debug:
                    note_count = len(oaf_piano_data.get("notes", []))
                    pedal_count = len(oaf_piano_data.get("pedal", []))
                    print(f"[DEBUG] Loaded Piano OaF: {oaf_piano_path}")
                    print(f"        Notes: {note_count}, Pedal segments: {pedal_count}")
            except Exception as e:
                print(f"[WARNING] Failed to load Piano OaF: {e}")
        else:
            if args.debug:
                print(f"[DEBUG] Piano OaF not found: {oaf_piano_path}, using NO-OP")

    if args.timbral_curves:
        timbre_curves_path = Path(args.timbral_curves)
        if timbre_curves_path.exists():
            try:
                timbre_curves_df = pd.read_parquet(timbre_curves_path)
                if args.debug:
                    print(f"[DEBUG] Loaded Timbral curves: {timbre_curves_path}")
                    print(
                        f"        Bars: {len(timbre_curves_df)}, Brightness mean: {timbre_curves_df['brightness'].mean():.3f}"
                    )
            except Exception as e:
                print(f"[WARNING] Failed to load Timbral curves: {e}")
        else:
            if args.debug:
                print(f"[DEBUG] Timbral curves not found: {timbre_curves_path}, using NO-OP")

    # パッチ3: Phase E NO-OP可視化ログ強化（1行サマリー）
    print(
        f"[GUIDE] bass_f0={'ON' if bass_f0_df is not None else 'OFF'} "
        f"oaf_piano={'ON' if oaf_piano_data is not None else 'OFF'} "
        f"timbre_cc={'ON' if timbre_curves_df is not None else 'OFF'}"
    )

    # Phase E: bars拡張（原曲追従強化属性追加）
    # Bass F0反映: register_hint, allow_position_shift, add_vibrato
    if bass_f0_df is not None and args.role == "bass":
        from collections import deque

        bass_f0_indexed = bass_f0_df.set_index("bar_index")

        # パッチ1: 3-bar running median to stabilize octave choice
        medq = deque(maxlen=3)
        smooth_regs = {}
        for b in sorted(bass_f0_indexed.index):
            row = bass_f0_indexed.loc[b]
            mm = float(row.get("f0_median_midi", np.nan))
            reg = int(mm // 12) if np.isfinite(mm) else None
            if reg is not None:
                medq.append(reg)
                smooth = int(round(np.median(list(medq))))
                smooth_regs[b] = smooth

        for idx in bars.index:
            b = int(bars.loc[idx, "bar_index"])
            if b in bass_f0_indexed.index:
                row = bass_f0_indexed.loc[b]
                # 平滑化されたregister_hintを使用
                if b in smooth_regs:
                    bars.loc[idx, "register_hint"] = smooth_regs[b]
                if row.get("slide_activity", 0.0) > 0.5:
                    bars.loc[idx, "allow_position_shift"] = True
                if row.get("vibrato_rate_hz", 0.0) > 4.0:
                    bars.loc[idx, "add_vibrato"] = True
        if args.debug:
            register_count = (
                int((bars["register_hint"] > 0).sum()) if "register_hint" in bars.columns else 0
            )
            slide_count = (
                int(bars.get("allow_position_shift", False).sum())
                if "allow_position_shift" in bars.columns
                else 0
            )
            vibrato_count = (
                int(bars.get("add_vibrato", False).sum()) if "add_vibrato" in bars.columns else 0
            )
            print(
                f"[DEBUG] Bass F0 reflected (3-bar smoothed): register_hint={register_count} bars, slide={slide_count}, vibrato={vibrato_count}"
            )

    # Piano OaF反映: voicing_complexity, apply_pedal
    if oaf_piano_data is not None and args.role == "piano":
        notes = oaf_piano_data.get("notes", [])
        pedal = oaf_piano_data.get("pedal", [])

        # パッチ2: sec_to_bar_index関数でbars_extendedを使用して正確にマッピング
        def sec_to_bar_index(sec, bars_df):
            """bars.parquetのstart_sec/end_secを使って正確なbar_index決定"""
            if "start_sec" not in bars_df.columns or "end_sec" not in bars_df.columns:
                # フォールバック: BPM計算
                return int((sec * bpm / 60.0) // 4)
            row = bars_df[(bars_df["start_sec"] <= sec) & (sec < bars_df["end_sec"])]
            return int(row["bar_index"].iloc[0]) if len(row) > 0 else None

        # ボイシング複雑度
        for idx in bars.index:
            b = int(bars.loc[idx, "bar_index"])
            bar_notes = [n for n in notes if n.get("bar") == b]
            if bar_notes:
                unique_starts = set(n["start_sec"] for n in bar_notes)
                avg_polyphony = len(bar_notes) / max(1, len(unique_starts))
                bars.loc[idx, "voicing_complexity"] = min(4, int(avg_polyphony))

        # ペダル延長（bars_extendedで正確にマッピング）
        for seg in pedal:
            bi = sec_to_bar_index(seg["start_sec"], bars)
            if bi is not None:
                bars_idx = bars[bars["bar_index"] == bi].index
                if len(bars_idx) > 0:
                    bars.loc[bars_idx[0], "apply_pedal"] = True

        if args.debug:
            voicing_count = (
                int((bars["voicing_complexity"] > 0).sum())
                if "voicing_complexity" in bars.columns
                else 0
            )
            pedal_count = (
                int(bars.get("apply_pedal", False).sum()) if "apply_pedal" in bars.columns else 0
            )
            print(
                f"[DEBUG] Piano OaF reflected (sec→bar mapping): voicing_complexity={voicing_count} bars, pedal={pedal_count}"
            )

    # Synth/Pad Timbre反映: cc11_expression, cc74_brightness, cc1_modulation, filter_open
    if timbre_curves_df is not None and args.role in [
        "strings",
        "piano",
    ]:  # stringsをSynth/Pad代表とする
        timbre_indexed = timbre_curves_df.set_index("bar_index")
        for idx in bars.index:
            b = int(bars.loc[idx, "bar_index"])
            if b in timbre_indexed.index:
                row = timbre_indexed.loc[b]
                bars.loc[idx, "cc11_expression"] = int(row.get("am_env", 0.5) * 127)
                bars.loc[idx, "cc74_brightness"] = int(row.get("brightness", 0.5) * 127)
                vib = row.get("vibrato_rate_hz", 0.0)
                bars.loc[idx, "cc1_modulation"] = int(np.clip((vib - 4) / 5 * 63 + 64, 0, 127))
                if row.get("roughness", 0.0) > 0.7:
                    bars.loc[idx, "filter_open"] = True
        if args.debug:
            cc11_count = (
                int((bars["cc11_expression"] > 0).sum()) if "cc11_expression" in bars.columns else 0
            )
            cc74_count = (
                int((bars["cc74_brightness"] > 0).sum()) if "cc74_brightness" in bars.columns else 0
            )
            cc1_count = (
                int((bars["cc1_modulation"] > 0).sum()) if "cc1_modulation" in bars.columns else 0
            )
            filter_count = (
                int(bars.get("filter_open", False).sum()) if "filter_open" in bars.columns else 0
            )
            print(
                f"[DEBUG] Timbre curves reflected: CC11={cc11_count}, CC74={cc74_count}, CC1={cc1_count}, filter={filter_count}"
            )

    # セグメント（複数コード/小節内分割対応）
    def segs_from_chordmap():
        evs = chordmap.get("events", [])

        # Phase 113: Use symbol-first parsing (safety net for root/quality inconsistencies)
        times = sorted(
            [(float(e.get("time", 0)), parse_chord(_symbol_from_event(e), mode_name)) for e in evs],
            key=lambda x: x[0],
        ) or [(0.0, parse_chord("C", mode_name))]
        segs = []
        for _, r in bars.iterrows():
            b = int(r["bar_index"])
            s = float(r["start_beats"])
            e = float(r["end_beats"])
            sec = str(r.get("section_label", ""))
            if args.multi_chords:
                cut = [s] + [t for (t, _ci) in times if s < t < e] + [e]
                cut = sorted(set(cut))
                for i in range(len(cut) - 1):
                    seg_s, seg_e = cut[i], cut[i + 1]
                    prev = [ci for (t, ci) in times if t <= seg_s + 1e-6]
                    ci = prev[-1] if prev else times[0][1]
                    segs.append((b, seg_s, seg_e, ci, sec))
            else:
                prev = [ci for (t, ci) in times if t <= s + 1e-6]
                ci = prev[-1] if prev else times[0][1]
                segs.append((b, s, e, ci, sec))
        return [Seg(*x) for x in segs]

    segs = segs_from_chordmap()

    patt = (
        midi_skeleton(Path(args.source_midi), bpm, bpb=args.beats_per_bar)
        if args.source_midi
        else None
    )

    events = []
    prev_vo = []
    last_gtr = None
    for i, seg in enumerate(segs):
        b = seg.bar
        e_val = resolve_energy(b, bars, energy_s, drums_active)
        is_break = (
            bool(int(drums_active[b]) == 0)
            if (drums_active is not None and b in drums_active.index)
            else False
        )

        vo = build_voicing(
            seg.chord, args.role, e_val, mode_name, args.tension_policy, args.open_voicing
        )
        if args.voice_leading:
            vo = voice_lead(prev_vo, vo)
        prev_vo = vo[:]

        # 役割別のイベント生成
        if args.role == "bass":
            ci_next = segs[i + 1].chord if i + 1 < len(segs) else seg.chord
            pitches = bass_line(seg, e_val, args.walking_bass, ci_next)

            # Phase E: register_hint反映（オクターブ補正）
            bar_row = bars[bars["bar_index"] == b]
            register_hint = None
            if len(bar_row) > 0 and "register_hint" in bar_row.columns:
                reg_val = bar_row.iloc[0].get("register_hint", np.nan)
                if pd.notna(reg_val):
                    register_hint = int(reg_val)

            if register_hint is not None:
                # register_hint: C0=0, C1=1, C2=2, ...
                # 現在のpitchesの平均オクターブを計算し、register_hintに近づける
                current_octaves = [p // 12 for p in pitches]
                avg_octave = sum(current_octaves) / max(1, len(current_octaves))
                octave_shift = register_hint - avg_octave
                if abs(octave_shift) >= 0.5:  # 半オクターブ以上ずれている場合
                    shift_semitones = int(round(octave_shift)) * 12
                    pitches = [p + shift_semitones for p in pitches]
                    if args.debug and i == 0:  # 最初のセグメントのみログ
                        print(
                            f"[DEBUG] Bass register_hint={register_hint}: shifted {shift_semitones} semitones"
                        )

            grid = [
                seg.start_b + k * ((seg.end_b - seg.start_b) / max(1, len(pitches)))
                for k in range(len(pitches))
            ]
            for gi, p in enumerate(pitches):
                t = grid[gi]
                # Phase 20: energy_curve → velocity (範囲拡大 70-95)
                vel_base = int(70 + 25 * e_val)  # 0.0→70, 1.0→95
                vel_var = gi % 5  # 微小変化（0,1,2,3,4）
                vel = min(127, max(50, vel_base + vel_var - 2))
                # Phase 23: duration変化（energy高→短め、低→長め）
                dur_base = 0.9 if e_val < 0.5 else (0.7 if e_val < 0.7 else 0.5)
                dur_var = (gi % 3) * 0.1  # 0.0, 0.1, 0.2のバリエーション
                dur = min(seg.end_b - t, dur_base + dur_var)

                # Phase E: allow_position_shift反映（スライド表現）
                allow_shift = False
                if len(bar_row) > 0 and "allow_position_shift" in bar_row.columns:
                    allow_shift = bool(bar_row.iloc[0].get("allow_position_shift", False))
                if allow_shift and gi > 0:
                    # 前のピッチから半音〜全音ずらす（スライド風）
                    prev_p = pitches[gi - 1]
                    if abs(p - prev_p) <= 2:  # 近接音
                        p += random.choice([-1, 1])  # 半音ずらす

                # Phase E: add_vibrato反映（ベロシティ揺らぎ）
                add_vib = False
                if len(bar_row) > 0 and "add_vibrato" in bar_row.columns:
                    add_vib = bool(bar_row.iloc[0].get("add_vibrato", False))
                if add_vib:
                    vel += random.choice([-3, 0, 3])  # ビブラート風揺らぎ

                off = t + dur
                if not allow_onset(t, anchors):
                    if gi % 2 == 1:
                        continue
                p += args.transpose_semitones
                events.append(
                    {"bar": b, "start_beats": t, "end_beats": off, "pitch": p, "velocity": vel}
                )

        elif args.role == "guitar":
            prefer_open = (args.open_voicing == "on") or (
                args.open_voicing == "auto"
                and e_val < 0.62
                and (
                    "intro" in (seg.section or "").lower() or "verse" in (seg.section or "").lower()
                )
            )
            shape = assign_guitar_shape(vo, last_gtr, prefer_open)
            last_gtr = shape
            direction = args.strum_direction
            if direction == "auto":
                direction = "down" if "chorus" in (seg.section or "").lower() else "up"
            width_beats = max(0.01, args.strum_width_ms / 1000.0 * (bpm / 60.0))
            hits = (
                guitar_strum_hits(seg, direction)
                if args.strum
                else [seg.start_b, min(seg.end_b - 0.5, seg.start_b + 2.0)]
            )
            for t_hit in hits:
                if allow_onset(t_hit, anchors):
                    # transpose は発音ピッチ側で適用
                    before = len(events)
                    emit_guitar_shape_strum(
                        events, b, t_hit, shape, direction, width_beats, vel=int(70 + 10 * e_val)
                    )
                    if args.transpose_semitones != 0:
                        for idx in range(before, len(events)):
                            events[idx]["pitch"] += args.transpose_semitones
            if is_break:
                events[:] = [
                    ev
                    for ev in events
                    if not (ev["bar"] == b and ev["start_beats"] > seg.start_b + 0.2)
                ]

        elif args.role == "piano":
            # Phase E: voicing_complexity反映（ボイシング密度調整）
            bar_row = bars[bars["bar_index"] == b]
            voicing_complexity = 2  # デフォルト
            apply_pedal = False
            if len(bar_row) > 0:
                if "voicing_complexity" in bar_row.columns:
                    vc_val = bar_row.iloc[0].get("voicing_complexity", np.nan)
                    if pd.notna(vc_val):
                        voicing_complexity = int(vc_val)
                if "apply_pedal" in bar_row.columns:
                    apply_pedal = bool(bar_row.iloc[0].get("apply_pedal", False))

            # voicing_complexityに応じてボイシング拡張
            expanded_vo = vo[:]
            if voicing_complexity >= 3 and len(vo) >= 2:
                # 複雑度3以上: 追加音を挿入（オクターブ上）
                expanded_vo.append(vo[1] + 12)
            if voicing_complexity >= 4 and len(vo) >= 3:
                # 複雑度4: さらに追加（3度上）
                expanded_vo.append(vo[2] + 12)

            times, pitches, lens, vels = piano_comp(
                seg, patt, expanded_vo, e_val, seg.section or ""
            )

            # apply_pedal反映: duration延長
            if apply_pedal:
                lens = [min(l * 1.5, seg.end_b - t) for t, l in zip(times, lens)]

            for t, p, l, v in zip(times, pitches, lens, vels):
                if allow_onset(t, anchors):
                    events.append(
                        {
                            "bar": b,
                            "start_beats": t,
                            "end_beats": min(seg.end_b, t + l),
                            "pitch": p + args.transpose_semitones,
                            "velocity": v,
                        }
                    )

        else:  # strings
            for t, p, l, vel in strings_engine(seg, vo, e_val):
                if allow_onset(t, anchors):
                    events.append(
                        {
                            "bar": b,
                            "start_beats": t,
                            "end_beats": min(seg.end_b, t + l),
                            "pitch": p + args.transpose_semitones,
                            "velocity": vel,
                        }
                    )

        # 小リック（4小節目/セクション終端）
        row = bars.loc[bars["bar_index"] == b]
        sec = str(row.iloc[0].get("section_label", "")) if len(row) else ""
        is_tr = False
        if b + 1 in bars["bar_index"].values:
            nxt = str(bars.loc[bars["bar_index"] == b + 1].iloc[0].get("section_label", ""))
            is_tr = nxt != sec
        if (b % 4 == 3) or is_tr:
            if args.role == "bass":
                nr = (
                    deg2midi((segs[i + 1].chord if i + 1 < len(segs) else seg.chord).root_pc, 0, 43)
                    + args.transpose_semitones
                )
                t = max(seg.end_b - 0.4, seg.start_b)
                if allow_onset(t, anchors):
                    events.append(
                        {
                            "bar": b,
                            "start_beats": t,
                            "end_beats": min(seg.end_b, t + 0.35),
                            "pitch": nr - 1,
                            "velocity": 78,
                        }
                    )
            elif args.role == "guitar" and last_gtr:
                t = max(seg.end_b - 0.4, seg.start_b)
                if allow_onset(t, anchors):
                    ordered = list(reversed(last_gtr.string_frets))
                    for j, (s, f_abs) in enumerate(ordered):
                        on = t + j * 0.06
                        if on < seg.end_b - 0.05:
                            pitch = open_strings()[s] + (f_abs - CAPO) + args.transpose_semitones
                            events.append(
                                {
                                    "bar": b,
                                    "start_beats": on,
                                    "end_beats": min(seg.end_b, on + 0.08),
                                    "pitch": pitch,
                                    "velocity": 74,
                                }
                            )
            elif args.role == "piano":
                base = vo[0] + args.transpose_semitones
                seq = [base + 2, base + 4, base + 7, base + 9]
                dt = 0.12
                t = max(seg.end_b - 0.48, seg.start_b)
                for s in seq:
                    if t < seg.end_b - 0.05 and allow_onset(t, anchors):
                        events.append(
                            {
                                "bar": b,
                                "start_beats": t,
                                "end_beats": min(seg.end_b, t + dt),
                                "pitch": s,
                                "velocity": 72,
                            }
                        )
                    t += dt
            else:
                t = max(seg.end_b - 0.5, seg.start_b)
                p = vo[min(2, len(vo) - 1)] + args.transpose_semitones
                if allow_onset(t, anchors):
                    events.append(
                        {
                            "bar": b,
                            "start_beats": t,
                            "end_beats": min(seg.end_b, t + 0.45),
                            "pitch": p,
                            "velocity": 62,
                        }
                    )

    # --- NEW: Phase 20+: velocity layers（3層：energy × セクションアクセント × ランダム） ---
    debug = {"phases": {}}
    have_energy = (
        (energy_s is not None) or ("energy" in bars.columns) or ("energy_curve" in bars.columns)
    )
    debug["phases"]["energy_curve"] = {"applied": have_energy, "depth": args.vel_depth}

    # --- activity density調整（--activity-col指定時） ---
    if activity_s is not None:
        before_n = len(events)

        # パッチ4: Activity密度間引きの再現性確保
        song_id = meta.get("song_id", "unknown")
        seed = abs(hash(f"{song_id}:{args.role}")) % (2**32)

        events = _apply_activity_density(
            events,
            activity_s,
            hat_s=hat_s,
            drums_active_s=drums_active,
            follow_drum_density=args.follow_drum_density,
            seed=seed,  # パッチ4: seed追加
        )
        debug["phases"]["activity_density"] = {
            "applied": True,
            "column": args.activity_col,
            "follow_drum_density": args.follow_drum_density,
            "notes_before": before_n,
            "notes_after": len(events),
            "seed": seed,  # パッチ4: seed記録
        }
    else:
        debug["phases"]["activity_density"] = {"applied": False}

    # --- velocity layers（activity反映） ---
    base = 78 if args.role in ("guitar", "piano", "strings") else 80
    _apply_velocity_layers(
        events, bars, role=args.role, base_default=base, depth=args.vel_depth, activity_s=activity_s
    )

    # --- plan段の最小ストラム（同時打鍵） ---
    if args.role in ("guitar", "piano") and args.plan_strum_ms > 0.0:
        before_n = len(events)
        events = _apply_min_strum_same_time(events, ms=args.plan_strum_ms, tempo=_infer_tempo(bars))
        debug["phases"]["plan_strum"] = {
            "applied": True,
            "ms": args.plan_strum_ms,
            "notes": before_n,
        }
    else:
        debug["phases"]["plan_strum"] = {"applied": False}

    # --- 弦ごとの sustain 差：guitar のみ（高弦ほど短めでカッティング感） ---
    if args.role == "guitar":
        # 簡易：高い音ほど短く（E4=64 付近を境に傾斜）
        for e in events:
            p = int(e["pitch"])
            scale = 0.85 if p < 52 else (0.78 if p < 60 else 0.70)
            dur = round(float(e.get("end_beats", 0)) - float(e.get("start_beats", 0)), 4)
            e["dur"] = max(0.0625, round(dur * scale, 4))
            e["end_beats"] = round(float(e["start_beats"]) + e["dur"], 6)

    # --- duration 多様化の強制（全ロール対象） ---
    before_stats = _richness_stats(events)
    events = _ensure_duration_variety(events, role=args.role, target=args.dur_target, bars=bars)
    after_stats = _richness_stats(events)
    debug["phases"]["duration_variety"] = {
        "target": args.dur_target,
        "before": {"uniq_durs": before_stats["uniq_durs"]},
        "after": {"uniq_durs": after_stats["uniq_durs"]},
    }

    # --- anchors-strict 適用（ボーカル近傍ミュート/減衰） ---
    if args.anchors_strict and anchors:
        before_n = len(events)
        # arranger_weights.yamlからduck_ms/grace_ms/accent_boost取得（デフォルト値あり）
        duck_ms = 80.0
        grace_ms = 40.0
        accent_boost = 10
        events = apply_anchors_strict(
            events, anchors, duck_ms=duck_ms, grace_ms=grace_ms, accent_boost=accent_boost, bpm=bpm
        )
        debug["phases"]["anchors_strict"] = {
            "applied": True,
            "duck_ms": duck_ms,
            "grace_ms": grace_ms,
            "accent_boost": accent_boost,
            "notes_before": before_n,
            "notes_after": len(events),
        }
    else:
        debug["phases"]["anchors_strict"] = {"applied": False}

    # --- enforce-richness チェック ---
    if args.enforce_richness:
        stats = _richness_stats(events)
        if (
            stats["uniq_pitches"] < args.richness_min_pitches
            or stats["uniq_vels"] < args.richness_min_vels
            or stats["uniq_durs"] < args.richness_min_durs
        ):
            raise SystemExit(
                f"[STRICT] Richness too low: "
                f"pitches={stats['uniq_pitches']}/{args.richness_min_pitches}, "
                f"vels={stats['uniq_vels']}/{args.richness_min_vels}, "
                f"durs={stats['uniq_durs']}/{args.richness_min_durs}"
            )

    # --- debug レポート出力 ---
    if args.debug:
        debug["final_stats"] = _richness_stats(events)
        debug_path = (
            Path(args.debug_report)
            if args.debug_report
            else (out.parent / f"plan_debug_{args.role}.json")
        )
        debug_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.debug:
            print(f"[DEBUG] Report written to {debug_path}")

    plan = {
        "meta": {
            "role": args.role,
            "version": "v4.1",
            "options": {
                "multi_chords": args.multi_chords,
                "voice_leading": args.voice_leading,
                "walking_bass": args.walking_bass,
                "strum": args.strum,
                "strum_direction": args.strum_direction,
                "strum_width_ms": args.strum_width_ms,
                "mode": mode_name,
                "open_voicing": args.open_voicing,
                "capo": CAPO,
                "transpose": args.transpose_semitones,
            },
            "beats_per_bar": args.beats_per_bar,
            "context_sources": {
                "chordmap": chord_p.exists(),
                "bars": bars_p.exists(),
                "lyric_anchors": lyr_p.exists(),
                "stem_features": stems_p.exists(),
                "energy_curve": (energy_s is not None),
                "drums_active": (drums_active is not None),
                "activity": (activity_s is not None),
                "activity_column": args.activity_col if activity_s is not None else None,
                # Phase E: 原曲追従強化ガイド
                "bass_f0": (bass_f0_df is not None),
                "oaf_piano": (oaf_piano_data is not None),
                "timbral_curves": (timbre_curves_df is not None),
            },
        },
        "tracks": [{"name": args.role.title(), "role": args.role, "events": events}],
    }

    # Phase E: Timbre CC情報をbarsメタに追加（plan_to_midiで利用）
    if timbre_curves_df is not None and args.role in ["strings", "piano"]:
        bars_cc = []
        for _, row in bars.iterrows():
            b = int(row["bar_index"])
            bar_cc = {"bar_index": b}
            if "cc11_expression" in row and pd.notna(row["cc11_expression"]):
                bar_cc["cc11"] = int(row["cc11_expression"])
            if "cc74_brightness" in row and pd.notna(row["cc74_brightness"]):
                bar_cc["cc74"] = int(row["cc74_brightness"])
            if "cc1_modulation" in row and pd.notna(row["cc1_modulation"]):
                bar_cc["cc1"] = int(row["cc1_modulation"])
            if "filter_open" in row and pd.notna(row["filter_open"]) and row["filter_open"]:
                bar_cc["filter_open"] = True
            if len(bar_cc) > 1:  # bar_index以外にデータがあれば追加
                bars_cc.append(bar_cc)
        if bars_cc:
            plan["meta"]["timbre_cc"] = bars_cc

    out = Path(args.out)
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    # リッチネス統計をstdoutに出力（E2E診断用）
    if events:
        pitches = sorted({e["pitch"] for e in events})
        vels = sorted({e.get("velocity", e.get("vel", 0)) for e in events})
        durs = sorted({round(e.get("end_beats", 0) - e.get("start_beats", 0), 3) for e in events})
        print(f"✅ {args.role}_plan written → {out} (events={len(events)})")
        print(
            f"   [RICHNESS] uniq_pitches={len(pitches)} uniq_vels={len(vels)} uniq_durs={len(durs)}"
        )
        print(
            f"   [CONTEXT] energy={energy_s is not None} drums_active={drums_active is not None} "
            f"activity={activity_s is not None} mode={mode_name}"
        )
        print(
            f"   [OPTIONS] multi_chords={args.multi_chords} voice_leading={args.voice_leading} "
            f"walking_bass={args.walking_bass} strum={args.strum}"
        )
    else:
        print(f"⚠️  {args.role}_plan written → {out} (events=0 - EMPTY!)")


if __name__ == "__main__":
    main()
