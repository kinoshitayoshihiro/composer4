"""
LAMDA公式CHORDSデータ → chordmap.json 変換アダプタ

独自エンコーディング形式を吸収し、標準的な chordmap.json に正規化します。
- 入力: LAMDa_CHORDS_DATA_*.pickle (任意のエンコーディング)
- 出力: {"unit": "ql", "events": [...]} 形式
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, Union
import pickle
import json
from pathlib import Path

# 音名定義
ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 質的コードマッピング（デフォルト）
QUAL_MAP_DEFAULT = {
    0: "maj",
    1: "min",
    2: "dim",
    3: "aug",
    4: "7",
    5: "maj7",
    6: "m7",
    7: "sus2",
    8: "sus4",
}


def _to_root_name(x: Any) -> str:
    """任意の入力をルート音名に変換"""
    if isinstance(x, int):
        return ROOTS[x % 12]
    s = str(x).strip().upper().replace("B", "A#")
    return s if s in ROOTS else "N"


def _normalize_symbol(sym: str) -> str:
    """コード記号を正規化"""
    s = sym.replace(":", "").replace(" ", "")
    s = s.replace("maj", "maj").replace("min", "m")
    s = s.replace("M7", "maj7").replace("min7", "m7")
    return s


def decode_chord_token(
    tok: Any, qual_map: Dict[Any, str] = QUAL_MAP_DEFAULT
) -> str:
    """
    任意のコードトークンを文字列に変換
    
    入力例:
      - "C:maj7" / "Am" / "N"
      - {"root": "D", "quality": "m7"}
      - (0, 5)  # (root_int, quality_code) → (C, maj7)
    """
    # dict形式
    if isinstance(tok, dict):
        root = tok.get("root") or tok.get("r") or "N"
        q = tok.get("quality") or tok.get("q") or ""
        ext = tok.get("ext") or tok.get("e") or ""
        root = _to_root_name(root)
        if root == "N":
            return "N"
        lab = f"{root}{q}{ext}"
        return _normalize_symbol(lab)
    
    # tuple/list形式
    if isinstance(tok, (tuple, list)) and len(tok) >= 2:
        root = _to_root_name(tok[0])
        if root == "N":
            return "N"
        qcode = tok[1]
        q = qual_map.get(qcode, "")
        lab = f"{root}{q}"
        return _normalize_symbol(lab)
    
    # string形式
    if isinstance(tok, str):
        s = tok.strip()
        if s.upper() in ("N", "NC", "N.C.", "NOCHORD"):
            return "N"
        return _normalize_symbol(s)
    
    # fallback
    return "N"


def _sec_to_ql(sec: float, bpm: float) -> float:
    """秒 → QL (quarter length) 変換"""
    return float(sec) * float(bpm) / 60.0


def _ticks_to_ql(ticks: int, tpq: int) -> float:
    """ticks → QL 変換"""
    return float(ticks) / float(max(1, tpq))


def decode_lamda_chord_sequence(chord_seq: List[List[int]]) -> List[Dict[str, Any]]:
    """
    LAMDA独自エンコーディング → 時系列イベント
    
    エンコーディング推測:
    [delta_time, duration, ?, pitch, velocity, ...]
    
    例: [0, 39, 0, 66, 96, 39, 0, 62, 96, ...]
         ↑   ↑  ↑   ↑   ↑
         dt  dur ?  pitch vel
    """
    events = []
    time = 0.0
    
    for chord in chord_seq:
        if not chord:
            continue
        
        # delta_time を加算
        delta_time = chord[0] if len(chord) > 0 else 0
        time += delta_time
        
        # ノート抽出 (5つ飛ばしでpitch取得: インデックス3, 8, 13, ...)
        notes = []
        for i in range(3, len(chord), 5):
            if i < len(chord):
                notes.append(chord[i])
        
        if notes:
            events.append({
                "time": time,
                "notes": notes,
                "chord": _notes_to_chord_symbol(notes),
            })
    
    return events


def _notes_to_chord_symbol(notes: List[int]) -> str:
    """ノート配列からコード記号を推定（簡易版）"""
    if not notes:
        return "N"
    
    # ルート音（最低音）
    root_pitch = min(notes)
    root_name = ROOTS[root_pitch % 12]
    
    # 音程解析（簡易）
    pitches = sorted(set(n % 12 for n in notes))
    root_pc = root_pitch % 12
    intervals = sorted((p - root_pc) % 12 for p in pitches)
    
    # パターンマッチング
    if intervals == [0, 4, 7]:
        return f"{root_name}maj"
    elif intervals == [0, 3, 7]:
        return f"{root_name}m"
    elif intervals == [0, 4, 7, 10]:
        return f"{root_name}7"
    elif intervals == [0, 4, 7, 11]:
        return f"{root_name}maj7"
    elif intervals == [0, 3, 7, 10]:
        return f"{root_name}m7"
    elif intervals == [0, 3, 6]:
        return f"{root_name}dim"
    elif intervals == [0, 4, 8]:
        return f"{root_name}aug"
    else:
        # その他は単純にルート音のみ
        return root_name


def build_chordmap_from_timeseries(
    series: List[Dict[str, Any]],
    timebase: str = "beats",
    *,
    bpm: float = 120.0,
    tpq: int = 480,
    min_step_ql: float = 2.0,
) -> Dict[str, Any]:
    """
    時系列コードイベント → chordmap.json
    
    Args:
        series: [{"time": ..., "chord": ...}, ...]
        timebase: "beats" | "ticks" | "sec" | "bar_index"
        bpm: テンポ（sec変換時に使用）
        tpq: ticks per quarter（ticks変換時に使用）
        min_step_ql: 同一コードの最小間隔（2.0QL未満は間引き）
    
    Returns:
        {"unit": "ql", "events": [...]}
    """
    events: List[Dict[str, Any]] = []
    
    def to_ql(t: Any) -> float:
        if timebase == "beats":
            return float(t)
        if timebase == "ticks":
            return _ticks_to_ql(int(t), tpq)
        if timebase == "sec":
            return _sec_to_ql(float(t), bpm)
        if timebase == "bar_index":
            return float(t) * 4.0
        return float(t)  # safe fallback
    
    # 1) decode → QL
    for it in series:
        if "time" in it:
            ql = to_ql(it["time"])
            chord_sym = it.get("chord") or it.get("token") or it.get("label") or "N"
        else:
            ql = to_ql(it.get("start", 0.0))
            chord_sym = it.get("chord") or it.get("token") or it.get("label") or "N"
        
        # コード記号を解析
        if chord_sym == "N":
            root, qual = "N", ""
        else:
            # 簡易パース（例: "Cmaj7" → root="C", qual="maj7"）
            root = chord_sym[0] if chord_sym else "N"
            qual = chord_sym[1:] if len(chord_sym) > 1 else ""
        
        events.append({
            "time": float(ql),
            "root": root,
            "quality": qual,
            "confidence": 0.5,  # LAMDA由来は信頼度0.5（要調整）
        })
    
    # 2) 時間順・同一コードの短距離重複を間引き
    events.sort(key=lambda e: e["time"])
    filtered: List[Dict[str, Any]] = []
    
    for e in events:
        if not filtered:
            filtered.append(e)
            continue
        
        prev = filtered[-1]
        same_chord = (e["root"] == prev["root"] and e["quality"] == prev["quality"])
        too_close = (e["time"] - prev["time"]) < min_step_ql
        
        if same_chord and too_close:
            continue  # 間引き
        
        filtered.append(e)
    
    return {"unit": "ql", "events": filtered}


def load_lamda_chords_pickle(
    pkl_path: str, field: Optional[str] = None
) -> List[List[int]]:
    """
    LAMDa_CHORDS_DATA_*.pickle を読み込み
    
    Returns:
        List[List[int]]: コードシーケンス配列
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    # LAMDA形式: List[[file_id, chord_sequence]]
    if isinstance(data, list) and data and isinstance(data[0], list):
        if len(data[0]) >= 2:
            # 最初のエントリのコードシーケンスを返す（デモ用）
            return data[0][1]
    
    return []


def convert_lamda_pickle_to_chordmap(
    pkl_path: str,
    output_json: Optional[str] = None,
    bpm: float = 120.0,
    min_step_ql: float = 2.0,
) -> Dict[str, Any]:
    """
    LAMDA pickle → chordmap.json 一括変換
    
    Args:
        pkl_path: LAMDa_CHORDS_DATA_*.pickle のパス
        output_json: 出力先（Noneなら辞書のみ返す）
        bpm: テンポ
        min_step_ql: 最小間隔
    """
    # 1) LAMDA pickleを読み込み
    chord_seq = load_lamda_chords_pickle(pkl_path)
    
    # 2) デコード
    events = decode_lamda_chord_sequence(chord_seq)
    
    # 3) chordmap化
    chordmap = build_chordmap_from_timeseries(
        events, timebase="beats", bpm=bpm, min_step_ql=min_step_ql
    )
    
    # 4) 保存（任意）
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(chordmap, f, ensure_ascii=False, indent=2)
    
    return chordmap


if __name__ == "__main__":
    # テスト実行
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python lamda_chords_to_chordmap.py <pickle_path> [output.json]")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    out_json = sys.argv[2] if len(sys.argv) > 2 else None
    
    chordmap = convert_lamda_pickle_to_chordmap(pkl_path, out_json)
    
    print(f"✅ Converted {pkl_path}")
    print(f"   Events: {len(chordmap.get('events', []))}")
    if out_json:
        print(f"   Saved to: {out_json}")
    else:
        print(json.dumps(chordmap, indent=2))
