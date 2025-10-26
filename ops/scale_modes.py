#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/scale_modes.py - 12音級(PC)マスク生成（Key/Mode + Profile + Tuning）
完全版：NO-OP保証 / sections取得 / マスク乗算 / 安全正規化 / 例外時フォールバック

order: [C, C#, D, D#, E, F, F#, G, G#, A, A#, B]
"""

import logging
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ---------------- Pitch-class & aliases ----------------
_PITCH_CLASS = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4,
    "F": 5, "E#": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11
}

_MODE_ALIASES = {
    # English
    "ionian": "ionian", "major": "ionian",
    "dorian": "dorian",
    "phrygian": "phrygian",
    "lydian": "lydian",
    "mixolydian": "mixolydian", "mixolyd": "mixolydian",
    "aeolian": "aeolian", "minor": "aeolian", "natural minor": "aeolian",
    "locrian": "locrian",
    # Japanese
    "アイオニアン": "ionian", "長音階": "ionian",
    "ドリアン": "dorian",
    "フリジアン": "phrygian",
    "リディアン": "lydian",
    "ミクソリディアン": "mixolydian",
    "エオリアン": "aeolian", "自然短音階": "aeolian",
    "ロクリアン": "locrian",
}

# 7 modes → semitone offsets (relative to tonic)
_MODAL_INTERVALS = {
    "ionian": [0, 2, 4, 5, 7, 9, 11],       # 1 2 3 4 5 6 7
    "dorian": [0, 2, 3, 5, 7, 9, 10],       # 1 2 b3 4 5 6 b7
    "phrygian": [0, 1, 3, 5, 7, 8, 10],     # 1 b2 b3 4 5 b6 b7
    "lydian": [0, 2, 4, 6, 7, 9, 11],       # 1 2 3 #4 5 6 7
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],   # 1 2 3 4 5 6 b7
    "aeolian": [0, 2, 3, 5, 7, 8, 10],      # 1 2 b3 4 5 b6 b7
    "locrian": [0, 1, 3, 5, 6, 8, 10],      # 1 b2 b3 4 b5 b6 b7
}

# モード別：どの度数を「特徴」/「Avoid」にするか（0-based: 0=1度, 3=4度, 6=7度）
_MODE_DEGREES = {
    "ionian":     {"char_p": [6],    "char_s": [1],     "char_t": [],     "avoid_p": [3]},
    "dorian":     {"char_p": [5],    "char_s": [1],     "char_t": [],     "avoid_p": []},
    "phrygian":   {"char_p": [1],    "char_s": [5],     "char_t": [],     "avoid_p": []},
    "lydian":     {"char_p": [3],    "char_s": [1],     "char_t": [],     "avoid_p": []},
    "mixolydian": {"char_p": [6],    "char_s": [1],     "char_t": [],     "avoid_p": []},
    "aeolian":    {"char_p": [5],    "char_s": [6],     "char_t": [],     "avoid_p": []},
    "locrian":    {"char_p": [4],    "char_s": [1],     "char_t": [],     "avoid_p": []},
}

# プロファイル（基礎配合）
_PROFILE: Dict[str, Dict[str, float]] = {
    "balanced":  {
        "nondiat": 0.12, "diat": 0.74, "root": 1.00, "third": 0.92, "fifth": 0.90,
        "char_p": 0.90, "char_s": 0.86, "char_t": 0.84, "leading": 0.82, "avoid_p": 0.62
    },
    "melodic":   {
        "nondiat": 0.12, "diat": 0.78, "root": 1.00, "third": 0.90, "fifth": 0.88,
        "char_p": 0.92, "char_s": 0.88, "char_t": 0.86, "leading": 0.84, "avoid_p": 0.65
    },
    "chordal":   {
        "nondiat": 0.12, "diat": 0.68, "root": 1.00, "third": 0.95, "fifth": 0.93,
        "char_p": 0.86, "char_s": 0.84, "char_t": 0.82, "leading": 0.78, "avoid_p": 0.60
    },
    "airy":      {
        "nondiat": 0.14, "diat": 0.72, "root": 0.98, "third": 0.88, "fifth": 0.86,
        "char_p": 0.96, "char_s": 0.92, "char_t": 0.88, "leading": 0.86, "avoid_p": 0.64
    },
    "cinematic": {
        "nondiat": 0.10, "diat": 0.70, "root": 1.00, "third": 0.92, "fifth": 0.90,
        "char_p": 0.98, "char_s": 0.94, "char_t": 0.90, "leading": 0.80, "avoid_p": 0.58
    },
    "dark_minor": {
        "nondiat": 0.10, "diat": 0.74, "root": 1.00, "third": 0.94, "fifth": 0.90,
        "char_p": 0.94, "char_s": 0.90, "char_t": 0.86, "leading": 0.70, "avoid_p": 0.60
    },
}


def tune_profiles(updates: Dict[str, Dict[str, float]]):
    """例: tune_profiles({'balanced': {'char_p':0.91, 'avoid_p':0.60}})"""
    for name, kv in updates.items():
        base = _PROFILE.setdefault(name, {})
        base.update(kv)


def tune_mode_degrees(mode: str, *, char_p=None, char_s=None, char_t=None, avoid_p=None):
    """例: tune_mode_degrees('ionian', char_s=[1,4], avoid_p=[3])"""
    mode = _MODE_ALIASES.get(mode.lower(), mode.lower())
    cfg = _MODE_DEGREES.setdefault(mode, {"char_p": [], "char_s": [], "char_t": [], "avoid_p": []})
    if char_p is not None:
        cfg["char_p"] = list(char_p)
    if char_s is not None:
        cfg["char_s"] = list(char_s)
    if char_t is not None:
        cfg["char_t"] = list(char_t)
    if avoid_p is not None:
        cfg["avoid_p"] = list(avoid_p)


def load_tuning_from_dict(d: Dict):
    """外部YAML/JSONを読み込んで一括適用"""
    if "profiles" in d:
        tune_profiles(d["profiles"])
    if "modes" in d:
        for m, vals in d["modes"].items():
            tune_mode_degrees(
                m,
                char_p=vals.get("char_p"),
                char_s=vals.get("char_s"),
                char_t=vals.get("char_t"),
                avoid_p=vals.get("avoid_p")
            )


def _parse_key_mode_text(key: str, mode: Optional[str]) -> Tuple[int, str]:
    """key: 'D', 'Fm', 'A minor' など → (root_pc, canonical_mode)"""
    s = (key or "").strip()
    toks = s.replace("-", " ").replace("_", " ").split()
    root = None
    minor_flag = False
    explicit_mode = None

    if toks:
        name = toks[0]
        if len(name) >= 2 and name[1] in "#b":
            name = name[:2]
        else:
            name = name[:1]
        if name in _PITCH_CLASS:
            root = _PITCH_CLASS[name]

    sl = s.lower()
    if (" minor" in sl) or (" min" in sl) or (" m" in sl and "mixolyd" not in sl):
        minor_flag = True

    if mode:
        explicit_mode = _MODE_ALIASES.get(mode.strip().lower(), mode.strip().lower())
    else:
        for k in _MODE_ALIASES:
            if k in sl:
                explicit_mode = _MODE_ALIASES[k]
                break

    if explicit_mode in _MODAL_INTERVALS:
        canon_mode = explicit_mode
    else:
        canon_mode = "aeolian" if minor_flag else "ionian"

    if root is None:
        root = 0
    return root, canon_mode


def _build_mask(root_pc: int, mode: str, prof: Dict[str, float], normalize: bool) -> List[float]:
    """内部関数：root/mode/profileから12要素マスクを生成"""
    ints = _MODAL_INTERVALS[mode]
    deg_pc = [(root_pc + iv) % 12 for iv in ints]

    mask = [float(prof["nondiat"])] * 12

    for pc in deg_pc:
        mask[pc] = float(prof["diat"])

    tonic, third, fifth = deg_pc[0], deg_pc[2], deg_pc[4]
    mask[tonic] = float(prof["root"])
    mask[third] = max(mask[third], float(prof["third"]))
    mask[fifth] = max(mask[fifth], float(prof["fifth"]))

    md = _MODE_DEGREES.get(mode, {})
    for idx in md.get("char_p", []):
        mask[deg_pc[idx]] = max(mask[deg_pc[idx]], float(prof["char_p"]))
    for idx in md.get("char_s", []):
        mask[deg_pc[idx]] = max(mask[deg_pc[idx]], float(prof["char_s"]))
    for idx in md.get("char_t", []):
        mask[deg_pc[idx]] = max(mask[deg_pc[idx]], float(prof["char_t"]))
    for idx in md.get("avoid_p", []):
        mask[deg_pc[idx]] = min(mask[deg_pc[idx]], float(prof["avoid_p"]))

    if mode in ("ionian", "lydian", "mixolydian"):
        mask[deg_pc[6]] = max(mask[deg_pc[6]], float(prof["leading"]))

    if normalize:
        s = sum(mask)
        if s > 1e-12:
            mask = [x / s for x in mask]

    return mask


def mask_for_key_mode(
    key: str,
    mode: str,
    *,
    scheme: str = "balanced",
    normalize: bool = True,
    char_gain: float = 1.0,
    avoid_gain: float = 1.0
) -> List[float]:
    """キーとモードから12要素マスクを生成"""
    root_pc, canon_mode = _parse_key_mode_text(key, mode)
    base = _PROFILE.get(scheme.lower(), _PROFILE["balanced"]).copy()

    for k in ("char_p", "char_s", "char_t"):
        base[k] *= float(char_gain)
    base["avoid_p"] *= float(avoid_gain)

    return _build_mask(root_pc, canon_mode, base, normalize)


def mask_for_key(key: str, *, scheme: str = "balanced", normalize: bool = True) -> List[float]:
    """後方互換API: key から Ionian/Aeolian を自動推定"""
    root_pc, canon_mode = _parse_key_mode_text(key, mode=None)
    base = _PROFILE.get(scheme.lower(), _PROFILE["balanced"])
    return _build_mask(root_pc, canon_mode, base, normalize)


def scale_mask_for_point(
    *,
    t_ql: float,
    sections: Optional[Dict[str, Any]],
    chord_root: Optional[str] = None,
    chord_quality: Optional[str] = None,
    scheme: str = "balanced"
) -> Optional[List[float]]:
    """InstrumentStage2Base統合用：sections から key/mode を取得してマスク生成"""
    try:
        if not sections:
            return None

        import math

        meter = int(sections.get("meter") or sections.get("timesig", {}).get("num") or 4)
        bar_idx = int(math.floor(t_ql / max(meter, 1)))

        key = None
        kh = sections.get("key_hint") or sections.get("key_changes")
        if isinstance(kh, list) and kh:
            last_key = None
            for entry in kh:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    b, k = int(entry[0]), str(entry[1])
                else:
                    b, k = int(entry.get("bar", 0)), str(entry.get("key", ""))
                if b <= bar_idx:
                    last_key = k
                else:
                    break
            key = last_key

        if not key:
            return None

        mode = None
        mh = sections.get("mode_hint")
        if isinstance(mh, list) and mh:
            last_mode = None
            for entry in mh:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    b, m = int(entry[0]), str(entry[1])
                else:
                    b, m = int(entry.get("bar", 0)), str(entry.get("mode", ""))
                if b <= bar_idx:
                    last_mode = m
                else:
                    break
            mode = last_mode

        if mode:
            return mask_for_key_mode(key, mode, scheme=scheme, normalize=True)
        else:
            return mask_for_key(key, scheme=scheme, normalize=True)

    except Exception as e:
        logger.debug(f"scale_mask_for_point: 例外によりNone返却 err={e}")
        return None


if __name__ == "__main__":
    import sys

    print("[Test 1] D Ionian:")
    mask1 = mask_for_key_mode("D", "ionian", scheme="balanced")
    print(f"  Mask: {[round(x, 2) for x in mask1]}")

    print("\n[Test 2] G Mixolydian:")
    mask2 = mask_for_key_mode("G", "mixolydian", scheme="balanced")
    print(f"  Mask: {[round(x, 2) for x in mask2]}")

    print("\n[Test 3] NO-OP:")
    result = scale_mask_for_point(t_ql=0.0, sections=None)
    print(f"  Result: {result}")

    print("\n[Test 4] sections integration:")
    test_sections = {
        "meter": 4,
        "key_hint": [[0, "D"], [48, "G"]],
        "mode_hint": [[0, "ionian"], [48, "mixolydian"]]
    }
    mask4a = scale_mask_for_point(t_ql=16.0, sections=test_sections)
    mask4b = scale_mask_for_point(t_ql=192.0, sections=test_sections)
    print(f"  Bar 4 (D Ionian): {[round(x, 2) for x in mask4a] if mask4a else None}")
    print(f"  Bar 48 (G Mixolydian): {[round(x, 2) for x in mask4b] if mask4b else None}")

    print("\n✅ Basic tests completed!")


# =========================
# Song-specific Presets 🎛️
# （コード相対ブースト & ブルース度ノブ 対応版）
# =========================

# ▼ プリセット定義辞書
_PRESETS = {
    "lydian_shimmer": {
        "mode": "lydian",
        "profile": "airy",
        "char_gain": 1.12,
        "avoid_gain": 1.00,
        # Lydianは #11(=+6) と 9th(=+2) をより空間的に
        "extra_pc_offsets": [2, 6],
        "extra_boost": 0.60,
        "code_offsets_mode": "key",
        "blues": 0.10
    },
    "dorian_soul": {
        "mode": "dorian",
        "profile": "melodic",
        "char_gain": 1.08,
        "avoid_gain": 1.00,
        # Dorianは Nat6(=+9) と 9th(=+2) を少し前へ
        "extra_pc_offsets": [2, 9],
        "extra_boost": 0.55,
        "code_offsets_mode": "key",
        "blues": 0.15
    },
    "mixolydian_blues": {
        "mode": "mixolydian",
        "profile": "chordal",
        "char_gain": 1.10,
        "avoid_gain": 0.95,   # avoid弱め＝"濁り"の余地を残す
        # Blue notes: b3(=+3), #11/b5(=+6) を控えめに許容
        "extra_pc_offsets": [3, 6],
        "extra_boost": 0.52,
        "code_offsets_mode": "key",
        "blues": 0.30
    },
    "phrygian_spice": {
        "mode": "phrygian",
        "profile": "cinematic",
        "char_gain": 1.00,
        "avoid_gain": 1.15,   # 不安定域の濁り抑制
        # b2(=+1) を軸に、b6(=+8) を副次的に
        "extra_pc_offsets": [1, 8],
        "extra_boost": 0.50,
        "code_offsets_mode": "key",
        "blues": 0.12
    },
    "aeolian_cinematic": {
        "mode": "aeolian",
        "profile": "dark_minor",
        "char_gain": 1.08,
        "avoid_gain": 1.10,
        # b6(=+8), b7(=+10) と 9th(=+2) を映画的に
        "extra_pc_offsets": [8, 10, 2],
        "extra_boost": 0.55,
        "code_offsets_mode": "key",
        "blues": 0.15
    },
    "ionian_vintage": {
        "mode": "ionian",
        "profile": "chordal",
        "char_gain": 1.00,
        "avoid_gain": 0.95,
        # 6th(=+9) と 9th(=+2) をほんのり（ジャズ/シティポップの気配）
        "extra_pc_offsets": [2, 9],
        "extra_boost": 0.50,
        "code_offsets_mode": "key",
        "blues": 0.08
    },
    "locrian_ambient": {
        "mode": "locrian",
        "profile": "airy",
        "char_gain": 0.95,    # ロクリアンは特色を少し柔らげる
        "avoid_gain": 1.10,
        # b5(=+6) を中心に、b2(=+1) を薄く
        "extra_pc_offsets": [6, 1],
        "extra_boost": 0.48,
        "code_offsets_mode": "key",
        "blues": 0.05
    },
    # ▼ 新規プリセット（要望の3種）
    "aeolian_dream": {
        "mode": "aeolian",
        "profile": "dark_minor",
        "char_gain": 1.12,
        "avoid_gain": 1.05,
        # バラード向け：コード相対 9th(+2), 11th(+5) を持ち上げ
        "extra_pc_offsets": [2, 5],
        "extra_boost": 0.56,
        "code_offsets_mode": "chord",
        "blues": 0.15
    },
    "ionian_citypop": {
        "mode": "ionian",
        "profile": "chordal",
        "char_gain": 1.06,
        "avoid_gain": 0.92,             # avoidを弱めて#11許容余地
        # コード相対 9th(+2), 13(+9), #11(+6)をうっすら
        "extra_pc_offsets": [2, 9, 6],
        "extra_boost": 0.52,
        "code_offsets_mode": "chord",
        "blues": 0.10
    },
    "dorian_gospel": {
        "mode": "dorian",
        "profile": "melodic",
        "char_gain": 1.15,
        "avoid_gain": 1.00,
        # コード相対 9th(+2), 11th(+5), 13(+9)
        "extra_pc_offsets": [2, 5, 9],
        "extra_boost": 0.60,
        "code_offsets_mode": "chord",
        "blues": 0.25
    },
}


# ▼ ユーティリティ関数
def list_presets() -> List[str]:
    """登録済みプリセット名を返す。"""
    return sorted(_PRESETS.keys())


def describe_preset(name: str) -> Dict:
    """プリセットの内容を取得（UI/ログ用）。存在しない場合は {}。"""
    p = _PRESETS.get(name.lower())
    return dict(p) if p else {}


def _parse_chord_root_pc(chord_symbol: Optional[str]) -> Optional[int]:
    """
    超軽量パーサ: 'Dmaj7', 'Bm7', 'G#7', 'F#m7b5', 'C/E' 等から ルートPC を抽出。
    スラッシュは前側を優先（コード・ベース分離）。失敗時は None。
    """
    if not chord_symbol:
        return None
    s = chord_symbol.strip()
    if "/" in s:
        s = s.split("/", 1)[0]
    if not s:
        return None
    head = s[0]
    acc = s[1] if len(s) >= 2 and s[1] in "#b" else ""
    name = (head + acc).upper().replace("＃", "#").replace("♯", "#").replace("♭", "b")
    return _PITCH_CLASS.get(name)


def _apply_extra_boosts_key_relative(
    mask: List[float],
    root_pc_key: int,
    offsets: List[int],
    level: float,
    normalize: bool
) -> List[float]:
    """キー相対: ルート(=Key)からの半音オフセットでPCを持ち上げる"""
    if not offsets or level <= 0.0:
        return mask
    floor_val = mask[root_pc_key] * float(level)
    for off in offsets:
        pc = (root_pc_key + int(off)) % 12
        if mask[pc] < floor_val:
            mask[pc] = floor_val
    if normalize:
        s = sum(mask)
        if s > 1e-12:
            mask = [x / s for x in mask]
    return mask


def _apply_extra_boosts_chord_relative(
    mask: List[float],
    root_pc_chord: Optional[int],
    offsets: List[int],
    level: float,
    normalize: bool
) -> List[float]:
    """コード相対: I=0, b3=+3, #11/b5=+6 など chord root 基準でブースト"""
    if root_pc_chord is None or not offsets or level <= 0.0:
        return mask
    floor_val = mask[root_pc_chord] * float(level)
    for off in offsets:
        pc = (root_pc_chord + int(off)) % 12
        if mask[pc] < floor_val:
            mask[pc] = floor_val
    if normalize:
        s = sum(mask)
        if s > 1e-12:
            mask = [x / s for x in mask]
    return mask


def _apply_blues_knob(
    mask: List[float],
    key_root_pc: int,
    mode: str,
    level: float,
    normalize: bool
) -> List[float]:
    """
    "ブルース度"ノブ: 非ダイアトニック(青音)を全体的に底上げ。
      level: 0.0..1.0（0=無効, 1=強い）
    手順:
      1) ダイアトPC集合を算出
      2) diat_mean を求める
      3) 非ダイアトPCを  new = (1-level)*old + level*(diat_mean*target)
         target=0.5（推奨）で「半分くらい許容」ニュアンス
    """
    if level <= 0.0:
        return mask
    ints = _MODAL_INTERVALS.get(mode, _MODAL_INTERVALS["ionian"])
    diat_pcs = {(key_root_pc + iv) % 12 for iv in ints}
    diat_vals = [mask[pc] for pc in diat_pcs]
    diat_mean = sum(diat_vals) / max(1, len(diat_vals))
    target = diat_mean * 0.5  # ここは曲想で調整可
    out = list(mask)
    for pc in range(12):
        if pc not in diat_pcs:
            out[pc] = (1.0 - level) * out[pc] + level * target
    if normalize:
        s = sum(out)
        if s > 1e-12:
            out = [x / s for x in out]
    return out


def mask_for_preset(
    key: str,
    preset_name: str,
    *,
    chord_symbol: Optional[str] = None,
    chord_root_pc: Optional[int] = None,
    code_offsets_mode: Optional[str] = None,
    blues: Optional[float] = None,
    normalize: bool = True
) -> List[float]:
    """
    プリセット名からマスクを生成。コード相対ブースト & ブルース度ノブ対応。
    
    Args:
        key: キー名 ("D", "Fm" など)
        preset_name: プリセット名 ("lydian_shimmer", "ionian_citypop" など)
        chord_symbol: コードシンボル ("Dmaj7", "Bm7" など) - コード相対ブースト用
        chord_root_pc: コードルートPC (0-11) - 直接指定する場合
        code_offsets_mode: 'chord' | 'key' | None (プリセット定義優先)
        blues: ブルース度 (0.0-1.0) - 非ダイアトニック底上げ
        normalize: 正規化するか
    
    Returns:
        12要素のマスク
    
    Examples:
        >>> mask_for_preset('D', 'lydian_shimmer')
        >>> mask_for_preset('D', 'ionian_citypop', chord_symbol='Dmaj7', blues=0.2)
        >>> mask_for_preset('Fm', 'dorian_gospel', chord_symbol='Bb7', blues=0.3)
    """
    p = _PRESETS.get(preset_name.lower())
    if not p:
        # 不明プリセット→通常解決（NO-OP的フォールバック）
        logger.warning(f"Unknown preset '{preset_name}', falling back to balanced")
        return mask_for_key(key, scheme="balanced", normalize=normalize)

    mode = p["mode"]
    scheme = p["profile"]
    cg = p.get("char_gain", 1.0)
    ag = p.get("avoid_gain", 1.0)
    extras = p.get("extra_pc_offsets", [])
    boost = float(p.get("extra_boost", 0.0))
    
    # 優先順位: 引数→プリセット定義
    code_mode = (code_offsets_mode or p.get("code_offsets_mode") or "key").lower()
    blues_lv = blues if blues is not None else float(p.get("blues", 0.0))

    # ベースマスク
    base = mask_for_key_mode(
        key, mode,
        scheme=scheme,
        normalize=True,
        char_gain=cg,
        avoid_gain=ag
    )

    # キー/コード相対ブースト
    key_root_pc, canon_mode = _parse_key_mode_text(key, mode)
    chord_pc = chord_root_pc if chord_root_pc is not None else _parse_chord_root_pc(chord_symbol)

    out = list(base)
    if extras and boost > 0.0:
        if code_mode == "chord":
            out = _apply_extra_boosts_chord_relative(out, chord_pc, extras, boost, normalize=True)
        else:
            out = _apply_extra_boosts_key_relative(out, key_root_pc, extras, boost, normalize=True)

    # ブルース度ノブ（非ダイアト底上げ）
    if blues_lv > 0.0:
        out = _apply_blues_knob(out, key_root_pc, canon_mode, blues_lv, normalize=True)

    return out


def resolve_mask_from_section(
    section: Dict[str, Any],
    default_key: str,
    chord_symbol: Optional[str] = None,
    chord_root_pc: Optional[int] = None
) -> Optional[List[float]]:
    """
    section(dict) から優先順位に従ってマスクを解決。
    
    優先順位: preset > mode > key
    
    Args:
        section: セクション辞書
            {
                "key_hint": "D",
                "mode": "lydian",
                "preset": "ionian_citypop",
                "code_offsets_mode": "chord",
                "blues": 0.2
            }
        default_key: key_hint が無い場合のデフォルトキー
        chord_symbol: コードシンボル (preset使用時)
        chord_root_pc: コードルートPC (preset使用時)
    
    Returns:
        12要素マスク、またはNone（NO-OP）
    """
    key = section.get("key_hint") or default_key
    if not key:
        return None
    
    if "preset" in section:
        return mask_for_preset(
            key, section["preset"],
            chord_symbol=chord_symbol,
            chord_root_pc=chord_root_pc,
            code_offsets_mode=section.get("code_offsets_mode"),
            blues=section.get("blues")
        )
    
    if "mode" in section:
        # mode だけ指定時でも blues ノブを効かせる
        base = mask_for_key_mode(key, section["mode"])
        if section.get("blues", 0) > 0:
            key_pc, canon_mode = _parse_key_mode_text(key, section["mode"])
            return _apply_blues_knob(list(base), key_pc, canon_mode, float(section["blues"]), normalize=True)
        return base
    
    # mode/preset 無し → key ベース
    return mask_for_key(key)


# =========================
# Preset Tests
# =========================
if __name__ == "__main__":
    print("\n[Test 5] Preset: lydian_shimmer")
    mask5 = mask_for_preset("D", "lydian_shimmer")
    print(f"  Result: {[round(m, 2) for m in mask5]}")
    
    print("\n[Test 6] Preset: ionian_citypop (chord-relative)")
    mask6 = mask_for_preset("D", "ionian_citypop", chord_symbol="Gmaj7", blues=0.2)
    print(f"  Result: {[round(m, 2) for m in mask6]}")
    
    print("\n[Test 7] Available presets:")
    print(f"  {list_presets()}")
    
    print("\n[Test 8] Describe 'aeolian_dream':")
    desc = describe_preset("aeolian_dream")
    print(f"  Mode: {desc.get('mode')}, Profile: {desc.get('profile')}, Blues: {desc.get('blues')}")
    
    print("\n✅ All tests completed!")

