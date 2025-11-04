#!/usr/bin/env python3
"""
Suno AI stem分離WAVから自動アレンジを生成

Usage:
    python scripts/suno_stem_arranger.py \
        --input data/suno_ai/suno_themesong/song_001/stemswav_001 \
        --output data/arranged_midi \
        --tempo 120 \
        --emotion energetic \
        --bars 16
"""

from __future__ import annotations

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

# パス設定
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from music21 import stream, tempo as m21tempo, instrument as m21instr, meter as m21meter
    from music21 import pitch as m21pitch
except ImportError:
    raise ImportError("music21 required: pip install music21")

# Generator imports
from generator.drums_generator_stage2 import DrumsGeneratorStage2

# Bass統合対象（既存ジェネレーター本体は改変しない）
try:
    from generator.bass_generator import BassGenerator
    from utilities.config_loader import load_main_cfg

    HAVE_BASS = True
except Exception:
    HAVE_BASS = False

try:
    from generator.piano_generator import PianoGenerator

    HAVE_PIANO = True
except Exception:
    HAVE_PIANO = False

try:
    from generator.guitar_generator import GuitarGenerator

    HAVE_GUITAR = True
except Exception:
    HAVE_GUITAR = False

try:
    from generator.strings_generator import StringsGenerator

    HAVE_STRINGS = True
except Exception:
    HAVE_STRINGS = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SunoStemArranger:
    """Suno AI stem分離WAVから自動アレンジ"""

    def __init__(self, main_cfg: Optional[Dict[str, Any]] = None):
        self.main_cfg = main_cfg or self._default_config()

        # ジェネレーター初期化
        self.generators = {"drums": DrumsGeneratorStage2()}

        # 各パート: 存在すれば登録（初期化失敗はスキップ）
        if HAVE_BASS:
            bass_gen = self._init_bass_generator()
            if bass_gen is not None:
                self.generators["bass"] = bass_gen

        if HAVE_PIANO:
            pg = self._init_simple_part("piano")
            if pg is not None:
                self.generators["piano"] = pg

        if HAVE_GUITAR:
            gg = self._init_simple_part("guitar")
            if gg is not None:
                self.generators["guitar"] = gg

        if HAVE_STRINGS:
            sg = self._init_simple_part("strings")
            if sg is not None:
                self.generators["strings"] = sg

    def _default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "tempo": 120,
            "time_signature": "4/4",
            "key_tonic": "C",
            "key_mode": "major",
        }

    # --- compose系を呼ぶための薄いアダプタ類 ---
    def _build_section_data(
        self,
        chords: List[str],
        tempo: float,
        emotion: str,
        section_name: str = "Verse",
    ) -> Dict[str, Any]:
        """
        BassGenerator など compose(section_data=...) 形式のための最小データ。
        将来、必要キーが増えたときもここだけ拡張すればよい。

        Piano/Guitrarには以下の追加キーが必要:
        - chord_symbol_for_voicing: セクション全体の代表コード
        - q_length: セクション長（四分音符単位）
        """
        # 1小節=4拍前提の安全なダミー（既存仕様を壊さない最小値）
        # Piano用: 最初のコードをセクション全体の代表コードとして使用
        first_chord = chords[0] if chords else "C"
        section_length = len(chords) * 4  # 1コード=4拍

        return {
            "section_name": section_name,
            "processed_chord_events": [{"symbol": c, "beats": 4} for c in chords],
            "musical_intent": {"emotion": emotion, "tempo_bpm": tempo},
            "part_params": {},
            # Piano/Guitar 必須パラメータ
            "chord_symbol_for_voicing": first_chord,
            "q_length": section_length,
            "absolute_offset": 0,  # セクション開始位置
        }

    def _extract_onsets_ql(self, part: "stream.Part", dedupe_eps: float = 0.02) -> List[float]:
        """
        パートから onset(QL=四分音符=1.0)を抽出。
        - dedupe_eps: 近接統合しきい値(QL)
        強化オプション(YAMLから渡す想定):
          _onset_cfg = {
            "min_note_ql": 0.0,          # 最小音価(短すぎるノート無視)
            "min_rest_ql": 0.0,          # 前回採用オンセットからの最小間隔
            "velocity_threshold": null,   # これ未満のVelを無視(Noneなら無効)
            "quantize_grid": null,        # 量子化グリッド(例: 0.25=16分) Noneで無効
            "max_per_quarter": null,      # 四分内での最大オンセット数(多重ヒット抑制)
            "octave_collapse_eps_ql": 0.0,# 同一ピッチクラスの連続オクターブを統合
            "degree_weights": {           # 度数重み付け(根音/五度/ダイアトニック等)
              "tonic": 1.0, "fifth": 1.0, "diatonic": 1.0, "non_diatonic": 1.0, "vel_pow": 0.0,
              "pc_weights": null,         # 相対度数(pc:0..11)ごとの重み上書き(例: {"0":1.4,"7":1.2})
              "min_note_ql_by_degree": null  # 相対度数ごとの最小音価(例: {"0":0.1,"7":0.1})
            },
            "key_segments": null          # [{bar:int, tonic:str|int, mode:str}] (モジュレーション分節)
          }
        """
        if part is None:
            return []
        try:
            # YAMLから渡される可能性のある設定を取得(無ければ空でNO-OP)
            _onset_cfg: Dict[str, Any] = getattr(self, "_onset_cfg", {}) or {}
            min_note_ql = float(_onset_cfg.get("min_note_ql", 0.0))
            min_rest_ql = float(_onset_cfg.get("min_rest_ql", 0.0))
            vel_thr = _onset_cfg.get("velocity_threshold", None)
            vel_thr = None if vel_thr is None else int(vel_thr)
            qgrid = _onset_cfg.get("quantize_grid", None)
            qgrid = None if qgrid in (None, 0, 0.0) else float(qgrid)
            max_per_q = _onset_cfg.get("max_per_quarter", None)
            max_per_q = None if max_per_q in (None, 0) else int(max_per_q)
            oct_eps = float(_onset_cfg.get("octave_collapse_eps_ql", 0.0))
            dw = _onset_cfg.get("degree_weights", {}) or {}
            w_tonic = float(dw.get("tonic", 1.0))
            w_fifth = float(dw.get("fifth", 1.0))
            w_diat = float(dw.get("diatonic", 1.0))
            w_non = float(dw.get("non_diatonic", 1.0))
            vel_pow = float(dw.get("vel_pow", 0.0))

            # --- 調式&キー分節(モジュレーション対応) ---
            key_tonic = (self.main_cfg or {}).get("key_tonic", "C")
            key_mode = (self.main_cfg or {}).get("key_mode", "major")
            segs = _onset_cfg.get("key_segments") or []  # [{bar, tonic, mode}]

            def _pc_from_tonic(tn: "str|int") -> int:
                if isinstance(tn, int):
                    return tn % 12
                try:
                    return m21pitch.Pitch(str(tn)).pitchClass
                except Exception:
                    return 0

            def _mode_diatonic_pcs(tonic_pc: int, mode: str) -> set[int]:
                m = (mode or "major").lower()
                # メジャー/モード/ハーモニック/メロディック対応
                scale_map = {
                    "ionian": "0,2,4,5,7,9,11",
                    "major": "0,2,4,5,7,9,11",
                    "dorian": "0,2,3,5,7,9,10",
                    "phrygian": "0,1,3,5,7,8,10",
                    "lydian": "0,2,4,6,7,9,11",
                    "mixolydian": "0,2,4,5,7,9,10",
                    "aeolian": "0,2,3,5,7,8,10",
                    "minor": "0,2,3,5,7,8,10",
                    "locrian": "0,1,3,5,6,8,10",
                    "harmonic_minor": "0,2,3,5,7,8,11",
                    "melodic_minor": "0,2,3,5,7,9,11",
                }
                steps = scale_map.get(m, scale_map["major"])
                return {(tonic_pc + int(s)) % 12 for s in steps.split(",")}

            # defaultセグメント(全小節に適用)
            try:
                default_tpc = _pc_from_tonic(key_tonic)
            except Exception:
                default_tpc = 0
            default_seg = [{"bar": 0, "tonic_pc": default_tpc, "mode": str(key_mode)}]
            segs_pc = [
                {
                    "bar": int(s.get("bar", 0)),
                    "tonic_pc": _pc_from_tonic(s.get("tonic", default_tpc)),
                    "mode": str(s.get("mode", key_mode)),
                }
                for s in (segs or default_seg)
            ]
            segs_pc.sort(key=lambda d: d["bar"])

            def _context_for_bar(bar_idx: int) -> tuple[int, set[int], int, str]:
                cur = segs_pc[0]
                for s in segs_pc:
                    if bar_idx >= s["bar"]:
                        cur = s
                tpc = cur["tonic_pc"]
                m = cur["mode"]
                di = _mode_diatonic_pcs(tpc, m)
                return tpc, di, (tpc + 7) % 12, m

            pcw_map = dw.get("pc_weights") or None
            mn_by_deg = dw.get("min_note_ql_by_degree") or None  # {"0":0.1, ...}

            def _score(pc: int, vel: int, bar_idx: int) -> float:
                tonic_pc, diatonic, fifth_pc, _ = _context_for_bar(bar_idx)
                s = 1.0
                if pcw_map and str((pc - tonic_pc) % 12) in pcw_map:
                    s *= float(pcw_map[str((pc - tonic_pc) % 12)])
                else:
                    if pc == tonic_pc:
                        s *= w_tonic
                    if pc == fifth_pc:
                        s *= w_fifth
                    s *= w_diat if pc in diatonic else w_non
                if vel_pow != 0.0:
                    s *= max(0.1, (vel or 64) / 64.0) ** vel_pow
                return float(s)

            # 候補抽出(音価/Velしきい値+度数別 最小音価フィルタ)
            # cand: List[Tuple[offset, pitchClass, midi, velocity, bar_idx]]
            cand: List[tuple[float, int, int, int, int]] = []
            ts = next((e for e in part.recurse().getElementsByClass(m21meter.TimeSignature)), None)
            ql_per_bar = float(ts.numerator) if ts else 4.0
            for n in part.recurse().notes:
                try:
                    ql = float(getattr(n, "duration").quarterLength)
                except Exception:
                    ql = 0.0
                if ql < min_note_ql:
                    continue
                if vel_thr is not None:
                    try:
                        v = int(getattr(n.volume, "velocity", 64) or 64)
                    except Exception:
                        v = 64
                    if v < vel_thr:
                        continue
                off = float(n.offset)
                bar_idx = int(off // ql_per_bar)
                try:
                    midi = int(getattr(n.pitch, "midi"))
                    pc = midi % 12
                except Exception:
                    midi, pc = 60, 0
                try:
                    vel = int(getattr(n.volume, "velocity", 64) or 64)
                except Exception:
                    vel = 64
                # 度数別の最小音価があれば適用
                if mn_by_deg:
                    tpc, _, _, _ = _context_for_bar(bar_idx)
                    deg = str((pc - tpc) % 12)
                    min_ql_deg = float(mn_by_deg.get(deg, 0.0))
                    if ql < min_ql_deg:
                        continue
                cand.append((off, pc, midi, vel, bar_idx))
            if not cand:
                return []
            # 量子化(任意)
            if qgrid:
                cand = [
                    (round(off / qgrid) * qgrid, pc, midi, vel, b)
                    for (off, pc, midi, vel, b) in cand
                ]
            cand.sort(key=lambda t: t[0])

            # 連続オクターブ縮退: 近接(oct_eps)かつ同一PCなら最初の1つを残す
            if oct_eps > 0.0:
                collapsed: List[tuple[float, int, int, int, int]] = []
                last_off: Optional[float] = None
                last_pc: Optional[int] = None
                for off, pc, midi, vel, b in cand:
                    if last_off is not None and last_pc is not None:
                        if abs(off - last_off) <= oct_eps and pc == last_pc:
                            # 同一PCのオクターブ跳躍を縮退(スキップ)
                            continue
                    collapsed.append((off, pc, midi, vel, b))
                    last_off, last_pc = off, pc
                cand = collapsed

            # 近接統合(dedupe)と最小休符間隔の適用
            #   重み付けが無い場合は従来通り"先着優先"、重み付けがあればスコア最大を選ぶ
            merged: List[tuple[float, int, int, int, int]] = []
            grp: List[tuple[float, int, int, int, int]] = []

            def _flush_group():
                if not grp:
                    return
                # 最小休符間隔チェックは"採択候補"に対して行う
                candidate = None
                if (w_tonic, w_fifth, w_diat, w_non, vel_pow) == (
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                ) and not pcw_map:
                    candidate = min(grp, key=lambda t: t[0])  # 先着
                else:
                    candidate = max(grp, key=lambda t: _score(t[1], t[3], t[4]))  # スコア最大
                if not merged or (candidate[0] - merged[-1][0]) >= min_rest_ql:
                    merged.append(candidate)
                grp.clear()

            last_anchor: Optional[float] = None
            for off, pc, midi, vel, b in cand:
                if last_anchor is None or abs(off - last_anchor) <= dedupe_eps:
                    grp.append((off, pc, midi, vel, b))
                    last_anchor = off if last_anchor is None else last_anchor
                else:
                    _flush_group()
                    grp.append((off, pc, midi, vel, b))
                    last_anchor = off
            _flush_group()

            # 四分あたりの上限(任意)
            if max_per_q is not None and max_per_q > 0:
                buckets: Dict[int, List[tuple[float, int, int, int, int]]] = {}
                for off, pc, midi, vel, b in merged:
                    q = int(off // 1.0)  # その四分のインデックス
                    buckets.setdefault(q, []).append((off, pc, midi, vel, b))
                trimmed: List[tuple[float, int, int, int, int]] = []
                for q, arr in buckets.items():
                    if (w_tonic, w_fifth, w_diat, w_non, vel_pow) == (
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                    ) and not pcw_map:
                        arr_sorted = sorted(arr, key=lambda t: t[0])
                    else:
                        arr_sorted = sorted(arr, key=lambda t: (-_score(t[1], t[3], t[4]), t[0]))
                    trimmed.extend(arr_sorted[:max_per_q])
                merged = sorted(trimmed, key=lambda t: t[0])
            # 選ばれたオンセットの"度数強度"を記録(DrumsでVel調整に使用可)
            strengths = {}
            for off, pc, midi, vel, b in merged:
                strengths[off] = _score(pc, vel, b)  # 基本1.0ベース
            self._onset_strengths_latest = strengths
            # 追加: 選ばれたオンセットのPCも保持(ユニゾンの度数判定に利用)
            pcs = {}
            for off, pc, _, __, ___ in merged:
                pcs[off] = int(pc)
            self._onset_pcs_latest = pcs
            return [off for (off, _, __, ___, ____) in merged]
        except Exception as e:
            logger.warning("extract_onsets failed: %s", e)
            return []

    def _derive_key_segments_from_chords(
        self,
        chords: List[str],
        bars: int,
        window_bars: int = 4,
        min_seg_bars: int = 4,
        mode_infer: str = "by_quality",
    ) -> List[Dict[str, Any]]:
        """
        ざっくりキー分節を推定: bar単位でルート投票し、windowで多数派→セグメント化。
        - mode_infer: "by_quality"('m'含む→minor/それ以外→major) or 固定モード名
        返値: [{"bar":start_bar, "tonic": "C", "mode": "major"}, ...]
        """
        if not chords:
            return []

        def root_of(sym: str) -> str:
            if not sym:
                return "C"
            s = sym.strip()
            r = []
            for ch in s:
                if ch.upper() in "ABCDEFG" or ch in "#b":
                    r.append(ch)
                else:
                    break
            return "".join(r) or "C"

        bars = int(bars)
        window = max(1, int(window_bars))
        votes = []
        for i in range(min(bars, len(chords))):
            votes.append(root_of(chords[i]))
        segs = []
        cur_root = None
        cur_mode = None
        cur_start = 0
        for i in range(0, len(votes), window):
            win = votes[i : i + window]
            if not win:
                break
            # 最頻出ルート
            root = max(set(win), key=win.count)
            if mode_infer == "by_quality":
                raw = "".join(chords[i : i + window])
                mode = "minor" if "m" in raw and "maj" not in raw else "major"
            else:
                mode = str(mode_infer)
            if cur_root is None:
                cur_root, cur_mode, cur_start = root, mode, i
            elif root != cur_root or mode != cur_mode:
                if (i - cur_start) >= max(1, int(min_seg_bars)):
                    segs.append({"bar": cur_start, "tonic": cur_root, "mode": cur_mode})
                    cur_root, cur_mode, cur_start = root, mode, i
        segs.append({"bar": cur_start, "tonic": cur_root or "C", "mode": cur_mode or "major"})
        return segs

    def _group_phrases(self, onsets_ql: List[float], gap_ql: float = 1.0) -> List[List[float]]:
        """
        Vocal等のオンセット列から、ギャップでフレーズ区間を抽出。
        返値: [[start_ql, end_ql], ...]
        """
        if not onsets_ql:
            return []
        onsets = sorted(float(x) for x in onsets_ql)
        phrases = []
        s = onsets[0]
        prev = onsets[0]
        for x in onsets[1:]:
            if (x - prev) > max(0.0, float(gap_ql)):
                phrases.append([s, prev])
                s = x
            prev = x
        phrases.append([s, prev])
        return phrases

    def _derive_sections_from_profile(
        self, profile: Dict[str, Any], emotion: str, bars: int
    ) -> List[Dict[str, Any]]:
        """
        emotion_profile.yaml の structure_markers を取得（無ければ [] を返す）。
        形式: [{bar:int, label:str}, ...]  ※ barは0始まり
        """
        try:
            em = (profile.get("emotions") or {}).get(emotion) or {}
            secs = em.get("structure_markers") or []
            out = []
            for s in secs:
                b = int(s.get("bar", 0))
                lab = str(s.get("label", "") or "")
                if 0 <= b < int(bars) and lab:
                    out.append({"bar": b, "label": lab})
            return sorted(out, key=lambda d: d["bar"])
        except Exception:
            return []

    def _extract_pitch_events(
        self,
        part: "stream.Part",
        min_note_ql: float = 0.0,
        quantize_grid: Optional[float] = None,
        dedupe_eps: float = 0.02,
    ) -> List[List[float]]:
        """
        Vocal等から [offset_ql, midi] を抽出。量子化/重複統合は任意。
        戻り: [[off, midi], ...]（offはQL）
        """
        if part is None:
            return []
        try:
            ev = []
            for n in part.recurse().notes:
                try:
                    ql = float(n.duration.quarterLength)
                except Exception:
                    ql = 0.0
                if ql < float(min_note_ql):
                    continue
                try:
                    off = float(n.offset)
                    midi = int(n.pitch.midi)
                except Exception:
                    continue
                ev.append([off, midi])
            if not ev:
                return []
            if quantize_grid:
                qg = float(quantize_grid)
                ev = [[round(off / qg) * qg, midi] for (off, midi) in ev]
            ev.sort(key=lambda t: t[0])
            # 近接統合
            out = []
            last = None
            for off, midi in ev:
                if last is None or abs(off - last) > float(dedupe_eps):
                    out.append([off, midi])
                    last = off
            return out
        except Exception:
            return []

    def _load_phoneme_events_csv(self, csv_path: str) -> List[List[Any]]:
        """
        CSV形式（ヘッダ任意）: offset_ql, class
          例) 12.50,sibilant
        戻り: [[off, "class"], ...]
        """
        import csv, os

        ev = []
        try:
            if not csv_path or not os.path.exists(csv_path):
                return ev
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                rdr = csv.reader(f)
                for row in rdr:
                    if not row or row[0].strip().lower().startswith("offset"):
                        # ヘッダ行はスキップ
                        continue
                    try:
                        off = float(row[0])
                        cls = str(row[1]).strip().lower()
                        ev.append([off, cls])
                    except Exception:
                        continue
        except Exception:
            return []
        return sorted(ev, key=lambda t: t[0])

    def _load_energy_csv(self, csv_path: str) -> List[List[float]]:
        """
        CSV形式（ヘッダ任意）: offset_ql, energy(0..1)
          例) 0.00,0.12
        戻り: [[off, energy], ...]
        """
        import csv, os

        arr = []
        try:
            if not csv_path or not os.path.exists(csv_path):
                return arr
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                rdr = csv.reader(f)
                for row in rdr:
                    if not row or row[0].strip().lower().startswith("offset"):
                        continue
                    try:
                        off = float(row[0])
                        en = max(0.0, min(1.0, float(row[1])))
                        arr.append([off, en])
                    except Exception:
                        continue
        except Exception:
            return []
        return sorted(arr, key=lambda t: t[0])

    def _label_and_instrument(self, part_name: str, part: stream.Part) -> str:
        """
        パートに楽器とパート名を設定

        Args:
            part_name: パート名（bass, piano, guitar, strings等）
            part: music21 Part

        Returns:
            設定したパート名
        """
        # 楽器設定
        instr_map = {
            "bass": m21instr.ElectricBass(),
            "piano": m21instr.Piano(),
            "guitar": m21instr.AcousticGuitar(),
            "strings": m21instr.StringEnsemble(),
            "drums": m21instr.UnpitchedPercussion(),
        }

        if part_name in instr_map:
            part.insert(0, instr_map[part_name])

        # パート名設定
        part.partName = part_name.capitalize()

        return part_name

    def _init_bass_generator(self):
        """
        BassGenerator の初期化を安全に行う。
        - 失敗時は None を返し、ドラムのみ継続。
        - 既存 main_config を使うが、見つからなければ最小設定で試行。
        """
        try:
            cfg_path = Path("configs/main_config.yaml")

            # main_config.yaml読み込み（なければ最小設定生成）
            if cfg_path.exists():
                main_cfg = load_main_cfg(cfg_path)
            else:
                logger.warning("Bass: configs/main_config.yaml not found, using minimal config")
                main_cfg = {
                    "global_settings": {
                        "tempo_bpm": 120,
                        "time_signature": "4/4",
                        "key_tonic": "C",
                        "key_mode": "major",
                    },
                    "part_defaults": {
                        "bass": {
                            "role": "bass",
                            "part_parameters": {},
                        }
                    },
                }

            return BassGenerator(
                global_settings=main_cfg.get("global_settings", {}),
                main_cfg=main_cfg,
                part_name="bass",
                default_instrument=m21instr.ElectricBass(),
            )
        except Exception as e:
            logger.warning("Bass initialization failed; continue without bass: %s", e)
            return None

    def _init_simple_part(self, part_name: str):
        """
        Piano/Guitar/Strings など、コンストラクタが軽量なジェネレーターを安全初期化。
        - 失敗時は None。
        - 追加の設定が必要なら将来この関数だけを拡張。
        """
        try:
            # 楽器マップ
            instr_map = {
                "piano": m21instr.Piano(),
                "guitar": m21instr.AcousticGuitar(),
            }

            # 最小限の設定
            minimal_cfg = {
                "global_settings": {
                    "tempo_bpm": 120,
                    "time_signature": "4/4",
                    "key_tonic": "C",
                    "key_mode": "major",
                },
                "part_defaults": {},
            }

            # Piano/Guitar: 最小パラメータで初期化試行
            if part_name in ["piano", "guitar"]:
                try:
                    GenClass = PianoGenerator if part_name == "piano" else GuitarGenerator
                    return GenClass(
                        global_settings=minimal_cfg["global_settings"],
                        main_cfg=minimal_cfg,
                        default_instrument=instr_map[part_name],
                        part_name=part_name,
                    )
                except Exception as e:
                    logger.warning(
                        "%s initialization with params failed: %s", part_name.capitalize(), e
                    )
                    return None

            # Strings: 最小パラメータで初期化（dict返却を_insert_part_or_partsで処理）
            if part_name == "strings":
                try:
                    return StringsGenerator(
                        global_settings=minimal_cfg["global_settings"],
                        main_cfg=minimal_cfg,
                        part_name=part_name,
                    )
                except Exception as e:
                    logger.warning("Strings initialization failed: %s", e)
                    return None

        except Exception as e:
            logger.warning("%s initialization failed; skip: %s", part_name.capitalize(), e)
        return None

    def _compose_with_chord_sections(
        self, gen: Any, chords: List[str], tempo: float, emotion: str, part_name: str
    ) -> Optional[Any]:
        """
        Piano/Guitar用: 各コードを個別セクションとして処理し、結合する

        Args:
            gen: ジェネレーター（PianoGenerator/GuitarGenerator）
            chords: コード進行リスト
            tempo: テンポ
            emotion: 感情パラメータ
            part_name: パート名

        Returns:
            dict[str, Part] (Piano) or Part (Guitar)
        """
        # 各コードごとにセクション生成
        sections_results = []

        for i, chord in enumerate(chords):
            section_data = {
                "section_name": f"Section_{i}",
                "processed_chord_events": [{"symbol": chord, "beats": 4}],
                "musical_intent": {"emotion": emotion, "tempo_bpm": tempo},
                "part_params": {},
                "chord_symbol_for_voicing": chord,
                "q_length": 4.0,  # 1コード = 1小節 = 4拍
                "absolute_offset": i * 4,  # セクション開始位置
            }

            try:
                result = gen.compose(section_data=section_data)
                sections_results.append(result)
            except Exception as e:
                logger.warning(f"{part_name} section {i} ({chord}) failed: {e}")
                # 失敗したセクションは空のRestを挿入
                if part_name == "piano":
                    from music21 import stream, note

                    rh = stream.Part(id="piano_rh")
                    lh = stream.Part(id="piano_lh")
                    rh.insert(0, note.Rest(quarterLength=4.0))
                    lh.insert(0, note.Rest(quarterLength=4.0))
                    sections_results.append({"piano_rh": rh, "piano_lh": lh})
                else:
                    from music21 import stream, note

                    p = stream.Part(id=part_name)
                    p.insert(0, note.Rest(quarterLength=4.0))
                    sections_results.append(p)

        if not sections_results:
            return None

        # Piano: dict形式の結合
        if isinstance(sections_results[0], dict):
            merged = {}
            for key in sections_results[0].keys():
                from music21 import stream

                merged_part = stream.Part(id=key)
                offset = 0.0
                for section_dict in sections_results:
                    if key in section_dict:
                        for element in section_dict[key].flatten():
                            # Instrumentは最初の1回だけ挿入
                            from music21 import instrument

                            if isinstance(element, instrument.Instrument):
                                if offset == 0.0:  # 最初のセクションのみ
                                    import copy

                                    merged_part.insert(0, copy.deepcopy(element))
                            else:
                                import copy

                                merged_part.insert(offset + element.offset, copy.deepcopy(element))
                        offset += section_dict[key].duration.quarterLength
                merged[key] = merged_part
            return merged

        # Guitar: Part形式の結合
        else:
            from music21 import stream, instrument

            merged_part = stream.Part(id=part_name)
            offset = 0.0
            for section_part in sections_results:
                for element in section_part.flatten():
                    # Instrumentは最初の1回だけ挿入
                    if isinstance(element, instrument.Instrument):
                        if offset == 0.0:
                            import copy

                            merged_part.insert(0, copy.deepcopy(element))
                    else:
                        import copy

                        merged_part.insert(offset + element.offset, copy.deepcopy(element))
                offset += section_part.duration.quarterLength
            return merged_part

    def _render_part(
        self, name: str, gen: Any, chords: List[str], tempo: float, emotion: str, bars: int
    ) -> Optional[Any]:  # stream.Part or List[stream.Part] or dict
        """
        汎用レンダラー：
        1) compose(section_data=...) があればそれを使用
        2) 無ければ generate(bars, chords, tempo, emotion) を試す
        どちらも無ければ None。

        返却値:
          - 単一Part: そのまま返す
          - dict[str, Part]: List[Part]に変換して返す (Piano/Strings対応)
          - None: 生成失敗

        Piano/Guitar の場合、コード進行を複数セクションに分割して
        より豊かなアレンジを生成する。
        """
        result = None

        # compose 優先（Piano/Guitar用にコード分割処理）
        try:
            # Piano/Guitar: 各コードを個別セクションとして処理
            if name in ["piano", "guitar"] and hasattr(gen, "compose"):
                result = self._compose_with_chord_sections(gen, chords, tempo, emotion, name)
            else:
                # 他の楽器: 通常処理
                sd = self._build_section_data(chords=chords, tempo=tempo, emotion=emotion)
                result = gen.compose(section_data=sd)
        except AttributeError:
            pass
        except Exception as e:
            logger.exception("%s compose failed: %s", name.capitalize(), e)
            return None

        # compose が無ければ generate にフォールバック
        if result is None:
            try:
                result = gen.generate(bars=bars, chords=chords, tempo=tempo, emotion=emotion)
            except AttributeError:
                logger.warning("%s has neither compose nor generate; skip", name.capitalize())
            except Exception as e:
                logger.exception("%s generate failed: %s", name.capitalize(), e)

        # dict返却の場合（Piano/Strings）→ List[Part]に変換
        if isinstance(result, dict):
            logger.info(
                f"{name.capitalize()} returned dict with {len(result)} parts: {list(result.keys())}"
            )
            return list(result.values())

        return result

    def _render_part_with_intent(
        self,
        name: str,
        gen: Any,
        chords: List[str],
        tempo: float,
        emotion: str,
        bars: int,
        extra_intent: Dict[str, Any],
    ) -> Optional[Any]:
        """
        extra_intent付きで_render_partを実行

        Args:
            name: パート名
            gen: ジェネレーター
            chords: コード進行
            tempo: テンポ
            emotion: 感情
            bars: 小節数
            extra_intent: density_multipliers等の追加パラメータ

        Returns:
            Part or List[Part] or None
        """
        # Piano/Guitar: コード分割処理でextra_intentを各セクションに適用
        if name in ["piano", "guitar"] and hasattr(gen, "compose"):
            return self._compose_with_chord_sections_and_intent(
                gen, chords, tempo, emotion, name, extra_intent
            )
        else:
            # 通常の_render_part
            return self._render_part(name, gen, chords, tempo, emotion, bars)

    def _compose_with_chord_sections_and_intent(
        self,
        gen: Any,
        chords: List[str],
        tempo: float,
        emotion: str,
        part_name: str,
        extra_intent: Dict[str, Any],
    ) -> Optional[Any]:
        """
        Piano/Guitar用: 各コードを個別セクションとして処理し、extra_intentを適用

        Args:
            gen: ジェネレーター
            chords: コード進行
            tempo: テンポ
            emotion: 感情
            part_name: パート名
            extra_intent: density_multipliers等

        Returns:
            dict[str, Part] (Piano) or Part (Guitar)
        """
        sections_results = []

        for i, chord in enumerate(chords):
            section_data = {
                "section_name": f"Section_{i}",
                "processed_chord_events": [{"symbol": chord, "beats": 4}],
                "musical_intent": {
                    "emotion": emotion,
                    "tempo_bpm": tempo,
                    "extra_intent": extra_intent,  # 追加インテントを差し込み
                },
                "part_params": {},
                "chord_symbol_for_voicing": chord,
                "q_length": 4.0,
                "absolute_offset": i * 4,
            }

            try:
                result = gen.compose(section_data=section_data)
                sections_results.append(result)
            except Exception as e:
                logger.warning(f"{part_name} section {i} ({chord}) failed: {e}")
                # 失敗時は空のRestを挿入
                if part_name == "piano":
                    from music21 import stream, note

                    rh = stream.Part(id="piano_rh")
                    lh = stream.Part(id="piano_lh")
                    rh.insert(0, note.Rest(quarterLength=4.0))
                    lh.insert(0, note.Rest(quarterLength=4.0))
                    sections_results.append({"piano_rh": rh, "piano_lh": lh})
                else:
                    from music21 import stream, note

                    p = stream.Part(id=part_name)
                    p.insert(0, note.Rest(quarterLength=4.0))
                    sections_results.append(p)

        if not sections_results:
            return None

        # 結合処理（既存の_compose_with_chord_sectionsと同じ）
        if isinstance(sections_results[0], dict):
            merged = {}
            for key in sections_results[0].keys():
                from music21 import stream, instrument

                merged_part = stream.Part(id=key)
                offset = 0.0
                for section_dict in sections_results:
                    if key in section_dict:
                        for element in section_dict[key].flatten():
                            if isinstance(element, instrument.Instrument):
                                if offset == 0.0:
                                    import copy

                                    merged_part.insert(0, copy.deepcopy(element))
                            else:
                                import copy

                                merged_part.insert(offset + element.offset, copy.deepcopy(element))
                        offset += section_dict[key].duration.quarterLength
                merged[key] = merged_part
            return merged
        else:
            from music21 import stream, instrument

            merged_part = stream.Part(id=part_name)
            offset = 0.0
            for section_part in sections_results:
                for element in section_part.flatten():
                    if isinstance(element, instrument.Instrument):
                        if offset == 0.0:
                            import copy

                            merged_part.insert(0, copy.deepcopy(element))
                    else:
                        import copy

                        merged_part.insert(offset + element.offset, copy.deepcopy(element))
                offset += section_part.duration.quarterLength
            return merged_part

    def _apply_humanize(
        self,
        part: Any,  # stream.Part
        tempo_bpm: float,
        seed: Optional[int],
        timing_ms: float = 8.0,
        vel_sigma: float = 5.0,
    ):
        """
        Humanize機能: タイミングとベロシティにランダムなゆらぎを追加

        Args:
            part: music21 Part
            tempo_bpm: テンポ（BPM）
            seed: 乱数シード（再現性用）
            timing_ms: タイミングのゆらぎ（±ms）
            vel_sigma: ベロシティの標準偏差
        """
        import random
        import hashlib

        # パート固有の決定的RNG（seedがNoneなら非決定）
        if seed is not None:
            part_tag = getattr(part, "id", getattr(part, "partName", "part"))
            h = hashlib.md5(f"{seed}:{part_tag}".encode()).hexdigest()
            rng = random.Random(int(h[:8], 16))
        else:
            rng = random

        try:
            notes = list(part.flatten().notes)

            # タイミングのゆらぎ（ms → quarter length変換）
            # 1拍 = 60000 / tempo_bpm (ms)
            ms_per_quarter = 60000.0 / tempo_bpm
            timing_ql = timing_ms / ms_per_quarter

            for n in notes:
                # タイミングのゆらぎ
                if hasattr(n, "offset"):
                    offset_shift = rng.uniform(-timing_ql, timing_ql)
                    new_off = n.offset + offset_shift
                    # 負のオフセット回避
                    n.offset = new_off if new_off >= 0.0 else 0.0

                # ベロシティのゆらぎ
                if hasattr(n, "volume") and hasattr(n.volume, "velocity"):
                    vel_shift = int(rng.gauss(0, vel_sigma))
                    new_vel = max(1, min(127, n.volume.velocity + vel_shift))
                    n.volume.velocity = new_vel

        except Exception as e:
            logger.warning(f"humanize failed on part {getattr(part, 'partName', '?')}: {e}")

    def _apply_swing_eighths(self, part: Any, swing_ratio: float, tempo_bpm: float):
        """
        8分裏をswing_ratioだけ後ろへ（0.0～0.15程度を想定）

        Args:
            part: music21 Part
            swing_ratio: スウィング量（0.0=無変更、0.04=軽いスウィング）
            tempo_bpm: テンポ（BPM）
        """
        if not swing_ratio or swing_ratio <= 0.0:
            return

        try:
            # 4/4想定：四分=1.0QL → 八分=0.5QL / 「裏」は各拍+0.5QL
            eighth = 0.5
            push = swing_ratio * (eighth * 0.5)  # 裏を"半分の半分"だけ遅らせる

            for n in list(part.flatten().notes):
                pos = n.offset / eighth
                # 位置が "…+0.5（裏）" に近いもの
                if abs((pos % 1.0) - 0.5) < 1e-6:
                    n.offset += push
                    if n.offset < 0.0:
                        n.offset = 0.0

        except Exception as e:
            logger.warning(f"swing apply failed on {getattr(part, 'partName', '?')}: {e}")

    def analyze_stems(self, stem_dir: Path) -> Dict[str, Any]:
        """
        Stem WAVファイルを分析

        Returns:
            {
                'drums': Path,
                'bass': Path,
                'guitar': Path,
                'vocals': Path,
                ...
            }
        """
        stem_files = {}

        # ファイル名からstem種別を推定
        for wav_file in stem_dir.glob("*.wav"):
            name_lower = wav_file.stem.lower()

            if "drum" in name_lower or "percussion" in name_lower:
                stem_files["drums"] = wav_file
            elif "bass" in name_lower:
                stem_files["bass"] = wav_file
            elif "guitar" in name_lower:
                stem_files["guitar"] = wav_file
            elif "piano" in name_lower or "keyboard" in name_lower:
                stem_files["piano"] = wav_file
            elif "string" in name_lower:
                stem_files["strings"] = wav_file
            elif "vocal" in name_lower:
                stem_files["vocals"] = wav_file
            elif "synth" in name_lower:
                stem_files["synth"] = wav_file

        logger.info(f"Found {len(stem_files)} stems: {list(stem_files.keys())}")
        return stem_files

    # -------- Tempo Map読み込み --------
    def _load_tempo_map(self, tempo_map_path: Path) -> List[List[float]]:
        """
        Tempo MapをJSONから読み込み

        Args:
            tempo_map_path: Tempo MapファイルパスJSON (例: analysis/tempo_map.json or sections.json)

        Returns:
            Tempo Map [[bar, bpm], [bar, bpm], ...]

        Raises:
            ValueError: 無効なTempo Map形式
            FileNotFoundError: ファイルが存在しない
        """
        import json

        if not tempo_map_path.exists():
            raise FileNotFoundError(f"Tempo map file not found: {tempo_map_path}")

        logger.info(f"Loading tempo map from: {tempo_map_path}")

        try:
            with open(tempo_map_path) as f:
                data = json.load(f)

            # sections.json形式対応（{"tempo_map": [[bar, bpm], ...]}）
            if isinstance(data, dict) and "tempo_map" in data:
                tempo_map = data["tempo_map"]
                logger.info(f"Extracted tempo_map from sections.json: {len(tempo_map)} entries")
                return tempo_map

            # tempo_map.json形式対応（[[bar, bpm], ...]）
            if isinstance(data, list):
                logger.info(f"Loaded tempo_map.json: {len(data)} entries")
                return data

            raise ValueError(
                f"Invalid tempo map format: expected list or dict with 'tempo_map' key, got {type(data)}"
            )

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from {tempo_map_path}: {e}")

    # -------- Emotion Profile 読み込みと適用値の取得 --------
    def _load_emotion_profile(self, path: Optional[str]) -> Dict[str, Any]:
        """
        Emotion ProfileのYAMLを読み込み

        Args:
            path: YAMLファイルパス（Noneまたは存在しない場合は空辞書）

        Returns:
            プロファイル辞書（読み込み失敗時は{}）
        """
        if not path:
            return {}
        try:
            p = Path(path)
            if not p.exists():
                logger.warning(f"emotion_profile not found: {path}")
                return {}
            return yaml.safe_load(p.read_text()) or {}
        except Exception as e:
            logger.warning(f"failed to load emotion_profile: {e}")
            return {}

    def _emotion_params(self, emotion: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        profileからemotionに対応するパラメータを取り出し、最低限の既定値で補完

        Args:
            emotion: 感情名（例: "energetic", "melancholic"）
            profile: _load_emotion_profileで読み込んだプロファイル

        Returns:
            {
              "humanize": {"timing_ms": 8.0, "vel_sigma": 5.0},
              "density_multipliers": {...},
              "velocity_shift": {...},
              "swing": {...}
            }
        """
        e = (profile.get("emotions") or {}).get(emotion) or {}
        h = e.get("humanize") or {}
        timing_ms = float(h.get("timing_ms", 8.0))
        vel_sigma = float(h.get("vel_sigma", 5.0))
        return {
            "humanize": {"timing_ms": timing_ms, "vel_sigma": vel_sigma},
            "density_multipliers": e.get("density_multipliers"),
            "velocity_shift": e.get("velocity_shift"),
            "swing": e.get("swing"),
        }

    def extract_chords_from_stems(self, stem_files: Dict[str, Path], bars: int = 16) -> List[str]:
        """
        Stem WAVからコード進行を推定

        TODO: 実装が必要
        - Piano/Guitar stemからコード推定
        - 現在は仮のコード進行を返す
        """
        # 簡易実装: 仮のコード進行
        basic_progression = ["C", "G", "Am", "F"]
        full_progression = basic_progression * (bars // len(basic_progression))

        logger.warning("Chord extraction not fully implemented - using C-G-Am-F")
        return full_progression[:bars]

    def _insert_part_or_parts(
        self, score: Any, result: Any, part_name: str  # stream.Score  # Part or List[Part]
    ) -> int:
        """
        Part または List[Part] を Score に挿入

        Args:
            score: 挿入先のスコア
            result: 単一Part または List[Part]
            part_name: パート名（ログ用）

        Returns:
            挿入したノート数の合計
        """
        if result is None:
            logger.warning(f"{part_name.capitalize()} returned None; continue.")
            return 0

        # List[Part]の場合
        if isinstance(result, list):
            total_notes = 0
            for i, part in enumerate(result):
                score.insert(0, part)
                note_count = len(list(part.flatten().notes))
                total_notes += note_count
                logger.info(
                    f"  ✅ {part_name.capitalize()} part {i+1}/{len(result)}: {note_count} notes"
                )
            logger.info(
                f"✅ {part_name.capitalize()}: {total_notes} notes total ({len(result)} parts)"
            )
            return total_notes

        # 単一Partの場合
        else:
            score.insert(0, result)
            note_count = len(list(result.flatten().notes))
            logger.info(f"✅ {part_name.capitalize()}: {note_count} notes")
            return note_count

    def arrange_with_generators(
        self,
        chords: List[str],
        tempo: float = 120,
        tempo_map: Optional[List[List[float]]] = None,
        emotion: str = "energetic",
        bars: int = 16,
        seed: Optional[int] = None,
        humanize: bool = True,
        emotion_profile_path: Optional[str] = None,
    ) -> Any:  # stream.Score
        """
        5つのジェネレーターでアレンジ生成

        Args:
            chords: コード進行リスト
            tempo: テンポ（BPM）- tempo_mapが指定されていない場合の固定テンポ
            tempo_map: Tempo Map [[bar, bpm], ...] (オプション)
            emotion: 感情表現（energetic, melancholic, calm, aggressive, romantic）
            bars: 小節数
            seed: 乱数シード（Humanize再現性用）
            humanize: Humanize機能の有効化
            emotion_profile_path: Emotion ProfileのYAMLパス

        Returns:
            生成されたstream.Score
        """
        score = stream.Score()

        # Emotion Profile を読み込み → 当該 emotion パラメータを確定
        profile = self._load_emotion_profile(emotion_profile_path)
        eparams = self._emotion_params(emotion, profile)

        # section_data に差し込む追加インテント（各ジェネレーターが任意参照）
        extra_intent = {
            "density_multipliers": eparams.get("density_multipliers"),
            "velocity_shift": eparams.get("velocity_shift"),
            "swing": eparams.get("swing"),
        }

        logger.info(
            f"Emotion '{emotion}': humanize timing_ms={eparams['humanize']['timing_ms']}, vel_sigma={eparams['humanize']['vel_sigma']}"
        )

        # テンポ設定（Tempo Map対応）
        if tempo_map:
            logger.info(f"Using tempo map with {len(tempo_map)} tempo changes")
            for bar, bpm in tempo_map:
                offset_ql = bar * 4.0  # 4/4拍子想定
                score.insert(offset_ql, m21tempo.MetronomeMark(number=bpm))
        else:
            logger.info(f"Using fixed tempo: {tempo} BPM")
            metronome = m21tempo.MetronomeMark(number=tempo)
            score.insert(0, metronome)

        # ---- 事前にBassを"プレビュー生成"してオンセット抽出（失敗してもスルー）----
        bass_preview = None
        bass_onsets: List[float] = []
        onset_strengths: Dict[float, float] = {}
        onset_pcs: Dict[float, int] = {}
        key_segments_used: List[Dict[str, Any]] = []
        if "bass" in self.generators:
            logger.info("Previewing bass for kick-bass unison onsets...")
            try:
                sd_prev = self._build_section_data(chords=chords, tempo=tempo, emotion=emotion)
                if isinstance(sd_prev.get("musical_intent"), dict):
                    sd_prev["musical_intent"]["extra_intent"] = extra_intent
                try:
                    bass_preview = self.generators["bass"].compose(section_data=sd_prev)
                except AttributeError:
                    bass_preview = self.generators["bass"].generate(
                        bars=bars, chords=chords, tempo=tempo, emotion=emotion
                    )
                # 人間味はまだ適用しない：Drumsのユニゾン基準がズレないように生オフセットで抽出
                if bass_preview is not None:
                    # Emotion YAMLの抽出器設定（任意・無ければNO-OPデフォルト）
                    onset_cfg = (
                        ((profile.get("emotions") or {}).get(emotion) or {}).get("mix_context")
                        or {}
                    ).get("onset_extractor") or {}
                    # キー分節の自動推定（任意）
                    kd = onset_cfg.get("key_detection") or {}
                    if bool(kd.get("enable", False)):
                        segs = self._derive_key_segments_from_chords(
                            chords=chords,
                            bars=bars,
                            window_bars=int(kd.get("window_bars", 4)),
                            min_seg_bars=int(kd.get("min_segment_bars", 4)),
                            mode_infer=str(kd.get("mode_infer", "by_quality")),
                        )
                        onset_cfg = dict(onset_cfg)  # shallow copy
                        onset_cfg["key_segments"] = segs
                        key_segments_used = segs
                    else:
                        key_segments_used = onset_cfg.get("key_segments") or []
                    # 抽出器に一時的に設定を渡す（安全な一時プロパティ）
                    self._onset_cfg = onset_cfg
                    bass_onsets = self._extract_onsets_ql(
                        bass_preview, dedupe_eps=float(onset_cfg.get("dedupe_eps", 0.02))
                    )
                    onset_strengths = getattr(self, "_onset_strengths_latest", {}) or {}
                    onset_pcs = getattr(self, "_onset_pcs_latest", {}) or {}
                    self._onset_cfg = {}
            except Exception as e:
                logger.exception("Bass preview generation failed: %s", e)

        # 1) Drums生成
        logger.info("Generating drums...")
        drum_part = None

        try:
            # 追加: emotion_profile の係数に "mix_context.bass_onsets_ql" を合流して伝える
            drums_overrides = {
                "density_multipliers": extra_intent.get("density_multipliers"),
                "velocity_shift": extra_intent.get("velocity_shift"),
                "swing": extra_intent.get("swing"),
                "drums_params": (profile.get("emotions", {}).get(emotion, {}) or {}).get(
                    "drums_params"
                ),
                # 追加：スタイルプリセット／ドラマー個性プリセットも透過
                "drums_style": (profile.get("emotions", {}).get(emotion, {}) or {}).get(
                    "drums_style"
                ),
                "drummer_profile": (profile.get("emotions", {}).get(emotion, {}) or {}).get(
                    "drummer_profile"
                ),
            }

            mix_ctx: Dict[str, Any] = {}
            if bass_onsets:
                strengths_list = [[float(k), float(v)] for k, v in sorted(onset_strengths.items())]
                pc_list = [[float(k), int(v)] for k, v in sorted(onset_pcs.items())]
                mix_ctx.update(
                    {
                        "bass_onsets_ql": bass_onsets,
                        "bass_onset_strengths": strengths_list,
                        "bass_onset_pcs": pc_list,
                    }
                )
            if key_segments_used:
                mix_ctx["key_segments"] = key_segments_used

            # 追加: Vocalプレビュー（あれば）→ オンセット&フレーズ区間を共有
            vocal_onsets: List[float] = []
            vocal_phrases: List[List[float]] = []
            vocal_pitches: List[List[float]] = []
            vocal_phonemes: List[List[Any]] = []
            vocal_energy: List[List[float]] = []
            if "vocal" in self.generators:
                try:
                    logger.info("Previewing vocal for conflict-aware drums...")
                    sd_v = self._build_section_data(chords=chords, tempo=tempo, emotion=emotion)
                    if isinstance(sd_v.get("musical_intent"), dict):
                        sd_v["musical_intent"]["extra_intent"] = extra_intent
                    try:
                        vocal_prev = self.generators["vocal"].compose(section_data=sd_v)
                    except AttributeError:
                        vocal_prev = self.generators["vocal"].generate(
                            bars=bars, chords=chords, tempo=tempo, emotion=emotion
                        )
                    if vocal_prev is not None:
                        vcfg = (
                            ((profile.get("emotions") or {}).get(emotion) or {}).get("mix_context")
                            or {}
                        ).get("vocal_extractor") or {}
                        self._onset_cfg = vcfg
                        vocal_onsets = self._extract_onsets_ql(
                            vocal_prev, dedupe_eps=float(vcfg.get("dedupe_eps", 0.03))
                        )
                        self._onset_cfg = {}
                        vocal_phrases = self._group_phrases(
                            vocal_onsets, gap_ql=float(vcfg.get("phrase_gap_ql", 1.0))
                        )
                        # ピッチイベントも抽出（最小音価/量子化はvocal_extractorを流用）
                        vocal_pitches = self._extract_pitch_events(
                            vocal_prev,
                            min_note_ql=float(vcfg.get("min_note_ql", 0.0)),
                            quantize_grid=vcfg.get("quantize_grid", None),
                            dedupe_eps=float(vcfg.get("dedupe_eps", 0.03)),
                        )
                        # 任意：子音CSV / エネルギーCSVのロード
                        vmeta = ((profile.get("emotions") or {}).get(emotion) or {}).get(
                            "mix_context"
                        ) or {}
                        ph_csv = (vmeta.get("vocal_phonemes") or {}).get("csv_path")
                        en_csv = (vmeta.get("vocal_energy") or {}).get("csv_path")
                        if ph_csv:
                            vocal_phonemes = self._load_phoneme_events_csv(str(ph_csv))
                        if en_csv:
                            vocal_energy = self._load_energy_csv(str(en_csv))
                except Exception:
                    pass
            if vocal_onsets:
                mix_ctx.update({"vocal_onsets_ql": vocal_onsets, "vocal_phrases": vocal_phrases})
            if vocal_pitches:
                mix_ctx["vocal_pitch_events"] = vocal_pitches  # [[off, midi], ...]
            if vocal_phonemes:
                mix_ctx["vocal_phonemes"] = vocal_phonemes  # [[off, "class"], ...]
            if vocal_energy:
                mix_ctx["vocal_energy"] = vocal_energy  # [[off, energy], ...]

            # セクションマーカー（emotion_profile 由来・任意）
            sections = self._derive_sections_from_profile(profile, emotion, bars)
            if sections:
                mix_ctx["sections"] = sections
            if mix_ctx:
                drums_overrides["mix_context"] = mix_ctx

            # 係数・シードをDrumsに伝達（例外安全）
            try:
                self.generators["drums"].set_overrides(drums_overrides)
            except Exception:
                pass
            try:
                self.generators["drums"].set_seed(seed)
            except Exception:
                pass
            drum_part = self.generators["drums"].generate(
                bars=bars,
                chords=chords,
                tempo=tempo,
                emotion=emotion,
            )
            score.insert(0, drum_part)
            if humanize:
                self._apply_humanize(drum_part, tempo_bpm=tempo, seed=seed)
        except Exception as e:
            logger.exception("Drums generation failed: %s", e)

        # 2) Bass生成（すでにプレビュー生成済みならそれを採用）
        if "bass" in self.generators:
            logger.info("Generating bass...")
            try:
                if bass_preview is not None:
                    bass_part = bass_preview
                else:
                    sd = self._build_section_data(chords=chords, tempo=tempo, emotion=emotion)
                    if isinstance(sd.get("musical_intent"), dict):
                        sd["musical_intent"]["extra_intent"] = extra_intent
                    try:
                        bass_part = self.generators["bass"].compose(section_data=sd)
                    except AttributeError:
                        bass_part = self.generators["bass"].generate(
                            bars=bars, chords=chords, tempo=tempo, emotion=emotion
                        )
                if bass_part is None:
                    logger.warning("Bass returned None; continue.")
                else:
                    self._label_and_instrument("bass", bass_part)
                    score.insert(0, bass_part)
                    if humanize:
                        self._apply_humanize(
                            bass_part,
                            tempo_bpm=tempo,
                            seed=seed,
                            timing_ms=eparams["humanize"]["timing_ms"],
                            vel_sigma=eparams["humanize"]["vel_sigma"],
                        )
            except Exception as e:
                logger.exception("Bass generation failed: %s", e)

        # 3) Piano生成
        if "piano" in self.generators:
            logger.info("Generating piano...")
            try:
                sd = self._build_section_data(chords=chords, tempo=tempo, emotion=emotion)
                if isinstance(sd.get("musical_intent"), dict):
                    sd["musical_intent"]["extra_intent"] = extra_intent
                try:
                    piano_part = self.generators["piano"].compose(section_data=sd)
                except AttributeError:
                    piano_part = self.generators["piano"].generate(
                        bars=bars, chords=chords, tempo=tempo, emotion=emotion
                    )
                if piano_part is None:
                    logger.warning("Piano returned None; continue.")
                else:
                    self._label_and_instrument("piano", piano_part)
                    score.insert(0, piano_part)
                    if humanize:
                        self._apply_humanize(
                            piano_part,
                            tempo_bpm=tempo,
                            seed=seed,
                            timing_ms=eparams["humanize"]["timing_ms"],
                            vel_sigma=eparams["humanize"]["vel_sigma"],
                        )
            except Exception as e:
                logger.exception("Piano generation failed: %s", e)

        # 4) Guitar生成
        if "guitar" in self.generators:
            logger.info("Generating guitar...")
            try:
                sd = self._build_section_data(chords=chords, tempo=tempo, emotion=emotion)
                if isinstance(sd.get("musical_intent"), dict):
                    sd["musical_intent"]["extra_intent"] = extra_intent
                try:
                    guitar_part = self.generators["guitar"].compose(section_data=sd)
                except AttributeError:
                    guitar_part = self.generators["guitar"].generate(
                        bars=bars, chords=chords, tempo=tempo, emotion=emotion
                    )
                if guitar_part is None:
                    logger.warning("Guitar returned None; continue.")
                else:
                    self._label_and_instrument("guitar", guitar_part)
                    score.insert(0, guitar_part)
                    if humanize:
                        self._apply_humanize(
                            guitar_part,
                            tempo_bpm=tempo,
                            seed=seed,
                            timing_ms=eparams["humanize"]["timing_ms"],
                            vel_sigma=eparams["humanize"]["vel_sigma"],
                        )
            except Exception as e:
                logger.exception("Guitar generation failed: %s", e)

        # 5) Strings生成
        if "strings" in self.generators:
            logger.info("Generating strings...")
            try:
                sd = self._build_section_data(chords=chords, tempo=tempo, emotion=emotion)
                if isinstance(sd.get("musical_intent"), dict):
                    sd["musical_intent"]["extra_intent"] = extra_intent
                try:
                    strings_part = self.generators["strings"].compose(section_data=sd)
                except AttributeError:
                    strings_part = self.generators["strings"].generate(
                        bars=bars, chords=chords, tempo=tempo, emotion=emotion
                    )
                if strings_part is None:
                    logger.warning("Strings returned None; continue.")
                else:
                    self._label_and_instrument("strings", strings_part)
                    score.insert(0, strings_part)
                    if humanize:
                        self._apply_humanize(
                            strings_part,
                            tempo_bpm=tempo,
                            seed=seed,
                            timing_ms=eparams["humanize"]["timing_ms"],
                            vel_sigma=eparams["humanize"]["vel_sigma"],
                        )
            except Exception as e:
                logger.exception("Strings generation failed: %s", e)

        return score

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        tempo: float = 120,
        tempo_map: Optional[List[List[float]]] = None,
        emotion: str = "energetic",
        bars: int = 16,
        seed: Optional[int] = None,
        humanize: bool = True,
        emotion_profile_path: str = "configs/emotion_profile.yaml",
    ) -> Path:
        """
        メイン処理

        Args:
            input_dir: Suno stem WAVディレクトリ
            output_dir: 出力MIDIディレクトリ
            tempo: テンポ (BPM) - tempo_mapが指定されていない場合の固定テンポ
            tempo_map: Tempo Map [[bar, bpm], ...] (オプション)
            emotion: 感情プロファイル
            bars: 小節数
            seed: 乱数シード（再現性用）
            humanize: Humanize機能の有効/無効
            emotion_profile_path: 感情プロファイルYAMLパス

        Returns:
            出力MIDIファイルパス
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Stem解析
        logger.info(f"Analyzing stems in: {input_dir}")
        stem_files = self.analyze_stems(input_dir)

        if not stem_files:
            raise ValueError(f"No stem files found in {input_dir}")

        # 2. コード進行推定
        logger.info("Extracting chord progression...")
        chords = self.extract_chords_from_stems(stem_files, bars=bars)
        logger.info(f"Chords: {chords}")

        # 3. アレンジ生成
        logger.info("Generating arrangement with 5 generators...")
        score = self.arrange_with_generators(
            chords=chords,
            tempo=tempo,
            tempo_map=tempo_map,
            emotion=emotion,
            bars=bars,
            seed=seed,
            humanize=humanize,
            emotion_profile_path=emotion_profile_path,
        )

        # 4. MIDI出力
        output_file = output_dir / f"{input_dir.name}_arranged.mid"
        score.write("midi", fp=output_file)
        logger.info(f"💾 Saved to: {output_file}")

        return output_file


def main():
    parser = argparse.ArgumentParser(description="Suno AI stem分離WAVから自動アレンジ")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Suno stem WAVディレクトリ",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/arranged_midi"),
        help="出力MIDIディレクトリ",
    )
    parser.add_argument(
        "--tempo",
        type=float,
        default=120,
        help="テンポ (BPM) - tempo-mapが指定されていない場合の固定テンポ",
    )
    parser.add_argument(
        "--tempo-map",
        type=Path,
        default=None,
        help="Tempo MapファイルパスJSON (例: analysis/tempo_map.json or sections.json)",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default="energetic",
        choices=["energetic", "calm", "melancholic", "hopeful", "intense"],
        help="感情プロファイル",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=16,
        help="生成小節数",
    )
    parser.add_argument(
        "--emotion-profile",
        type=str,
        default="configs/emotion_profile.yaml",
        help="感情プロファイルYAMLファイルパス",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="乱数シード（再現性用）",
    )
    parser.add_argument(
        "--humanize",
        action="store_true",
        default=True,
        help="Humanize機能を有効化（デフォルト: 有効）",
    )
    parser.add_argument(
        "--no-humanize",
        action="store_false",
        dest="humanize",
        help="Humanize機能を無効化",
    )

    args = parser.parse_args()

    # SunoStemArranger初期化
    arranger = SunoStemArranger()

    # Tempo Map読み込み
    tempo_map = None
    if args.tempo_map:
        tempo_map = arranger._load_tempo_map(args.tempo_map)
        logger.info(f"Loaded tempo map from: {args.tempo_map}")

    # 実行
    output_file = arranger.run(
        input_dir=args.input,
        output_dir=args.output,
        tempo_map=tempo_map,
        tempo=args.tempo,
        emotion=args.emotion,
        bars=args.bars,
        seed=args.seed,
        humanize=args.humanize,
        emotion_profile_path=args.emotion_profile,
    )

    print(f"\n{'='*60}")
    print(f"✅ Arrangement complete!")
    print(f"📁 Output: {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
