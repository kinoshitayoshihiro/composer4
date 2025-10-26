from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import yaml
from adapters.chord_whitelist import validate_events

# ---- helpers ----
def _notes_from_block(block: List[int]) -> List[int]:
    # 21..108 を「音高」とみなす（GMレンジ基準の簡易抽出）
    return [x for x in block if 21 <= int(x) <= 108]

def _split_blocks(seq: List[int]) -> Tuple[int,int,List[List[int]]]:
    """
    フラット配列を [dt, dur, <payload...>] とみなし、payload を4～8個程度で分割（ヒューリスティック）
    """
    if len(seq) < 2: return 0, 0, []
    dt, dur = int(seq[0]), int(seq[1])
    rest = [int(v) for v in seq[2:]]
    blocks: List[List[int]] = []
    cur: List[int] = []
    for v in rest:
        cur.append(v)
        if len(cur) >= 4:
            blocks.append(cur); cur = []
    if cur: blocks.append(cur)
    return dt, dur, blocks

def _apply_root_alias(name: str, aliases: Dict[str,str]) -> str:
    s = name.strip()
    return aliases.get(s, s)

def _root_from_int(n: int, int_to_name: List[str]) -> str:
    return int_to_name[int(n) % 12]

def _quality_from_code(q: Any, code_map: Dict[Any,str], alias: Dict[str,str]) -> str:
    if isinstance(q, int):
        return code_map.get(q, "")
    s = str(q).strip()
    return alias.get(s, s)

def _label_from_pitches(pitches: List[int]) -> Tuple[str,str,float]:
    """
    music21 が利用可能ならそれで figure を得る。失敗時は三和音ヒューリスティック。
    """
    try:
        from music21 import chord, pitch as m21p, harmony
        if not pitches:
            return "N","",0.0
        c = chord.Chord(pitches)
        cs = harmony.chordSymbolFromChord(c)  # music21の推定
        sym = cs.figure
        i = 1
        if len(sym) >= 2 and sym[1] in ("#","-"):
            i = 2
        return sym[:i], sym[i:], 0.9
    except Exception:
        if not pitches: return "N","",0.0
        pcs = sorted({p % 12 for p in pitches})
        root = min(pcs)
        m3 = (root+3)%12; M3=(root+4)%12
        qual = "m" if m3 in pcs and M3 not in pcs else "maj"
        NAME = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        return NAME[root], qual, 0.5

# ---- main ----
def decode_chord_seq_to_events(
    chord_seq: List[Any],
    tpq: int = 480,
    min_step_ql: float = 2.0,
    token_map_yaml: Optional[str] = None
) -> Dict[str,Any]:
    """
    chord_seq: LAMDAの1曲ぶんコード系列（代表例: 各chordが [dt, dur, ...payload...]）
               もしくは dict形式 {"root":..., "q":..., "dt":..., "dur":...} の配列にも対応
    """
    # YAML 読み込み（任意）
    roots_map = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    root_alias: Dict[str,str] = {}
    qual_codes: Dict[Any,str] = {}
    qual_alias: Dict[str,str] = {}
    if token_map_yaml:
        try:
            with open(token_map_yaml, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
            roots_map = y.get("roots",{}).get("int_to_name", roots_map)
            root_alias = y.get("roots",{}).get("aliases", {}) or {}
            qual_codes = y.get("qualities",{}).get("code_map", {}) or {}
            qual_alias = y.get("qualities",{}).get("aliases", {}) or {}
            min_step_ql = float(y.get("fallbacks",{}).get("min_step_ql", min_step_ql))
        except Exception:
            pass

    events: List[Dict[str,Any]] = []
    cur_ticks = 0

    for item in chord_seq:
        # dictケース：{"dt":..,"dur":..,"root":(int|str),"q":(int|str)}
        if isinstance(item, dict):
            dt = int(item.get("dt", 0))
            cur_ticks += dt
            root = item.get("root", "N")
            q = item.get("q", item.get("quality",""))
            # root
            if isinstance(root, int):
                rname = _root_from_int(root, roots_map)
            else:
                rname = _apply_root_alias(str(root), root_alias)
            if rname.upper() in ("N","NC","N.C.","NOCHORD"):
                events.append({"time": cur_ticks/max(1,tpq), "root":"N", "quality":"", "confidence":0.0})
                continue
            # quality
            qname = _quality_from_code(q, qual_codes, qual_alias)
            events.append({"time": cur_ticks/max(1,tpq), "root": rname, "quality": qname, "confidence": 0.6})
            continue

        # 配列ブロックケース：先頭2つが dt, dur 想定
        if isinstance(item, list) and len(item) >= 2:
            dt, dur, blocks = _split_blocks([int(v) for v in item])
            cur_ticks += dt
            notes: List[int] = []
            for b in blocks: notes += _notes_from_block(b)
            notes = sorted(set(notes))
            r, q, conf = _label_from_pitches(notes)
            events.append({"time": cur_ticks/max(1,tpq), "root": r, "quality": q, "confidence": conf})
            continue

        # それ以外は無視
        continue

    # 2QL未満の同一コードを間引き
    events.sort(key=lambda e: float(e["time"]))
    merged: List[Dict[str,Any]] = []
    for e in events:
        if not merged:
            merged.append(e); continue
        prev = merged[-1]
        if e["root"]==prev["root"] and e["quality"]==prev["quality"] and (e["time"]-prev["time"]) < float(min_step_ql):
            continue
        merged.append(e)

    out = {"unit":"ql","events": merged}

    # music21 ホワイトリスト検証（正規化）
    try:
        cleaned, stats = validate_events(out["events"])
        out["events"] = cleaned
        out["validation"] = stats
    except Exception:
        pass

    return out
