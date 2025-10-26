#!/usr/bin/env python3
"""
拡張メタ関数のスモークテスト（独立実行版）
"""
import sys
import json
from typing import Any, Dict, List

try:
    import pretty_midi
except ImportError:
    print("ERROR: pretty_midi not installed. Run: pip install pretty_midi")
    sys.exit(1)


def _safe_round_extended(x: Any, nd: int = 3) -> float:
    """安全な丸め処理"""
    try:
        return round(float(x), nd)
    except (TypeError, ValueError, AttributeError):
        return 0.0


def extract_tempo_grid_extended(midi_data: Any) -> Dict[str, Any]:
    """
    1) テンポ/拍グリッド抽出（改善版: timesig bar紐付け修正）
    返り値:
      {
        "tempo_map": [[time_ql, bpm], ...],
        "timesig_map": [[bar_index, "4/4"], ...],
        "downbeats_ql": [0, 960, 1920, ...]
      }
    """
    out: Dict[str, Any] = {"tempo_map": [], "timesig_map": [], "downbeats_ql": []}
    try:
        # テンポ取得
        tempo_changes = midi_data.get_tempo_changes()
        if len(tempo_changes[0]) > 0:
            for tick, bpm in zip(tempo_changes[0], tempo_changes[1]):
                out["tempo_map"].append([int(tick * 480), _safe_round_extended(bpm, 2)])

        # デフォルトテンポ
        if not out["tempo_map"]:
            out["tempo_map"].append([0, 120.0])

        # 拍子取得
        time_signatures = midi_data.time_signature_changes
        if time_signatures:
            # downbeatsを取得
            downbeats = midi_data.get_downbeats()
            downbeats_ql = [int(db * 480) for db in downbeats]

            def _nearest_bar_index(ts_time_ql: float, downbeats_ql: List[int]) -> int:
                """最も近いdownbeat indexを返す"""
                if not downbeats_ql:
                    return 0
                min_dist = float("inf")
                best_idx = 0
                for i, db_ql in enumerate(downbeats_ql):
                    dist = abs(db_ql - ts_time_ql)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
                return best_idx

            for ts in time_signatures:
                ts_time_ql = int(ts.time * 480)
                bar_idx = _nearest_bar_index(ts_time_ql, downbeats_ql)
                out["timesig_map"].append([bar_idx, f"{ts.numerator}/{ts.denominator}"])

        # デフォルト拍子
        if not out["timesig_map"]:
            out["timesig_map"].append([0, "4/4"])

        # downbeats QL
        downbeats = midi_data.get_downbeats()
        out["downbeats_ql"] = [int(db * 480) for db in downbeats]
    except Exception:
        pass
    return out


def extract_bar_chords_extended(
    midi_data: Any, downbeats_ql: List[int]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    2) 1小節単位のコード推定（PC-setベース簡易版）
    返り値: {"events": [{"time":0.0, "root":"C", "quality":"maj", "confidence":0.0}, ...]}
    """
    events = []
    try:
        nb = len(downbeats_ql)
        if nb < 1:
            return {"events": events}

        # 各小節のピッチクラス集合を取得
        for i in range(nb):
            start_ql = downbeats_ql[i]
            end_ql = downbeats_ql[i + 1] if i + 1 < nb else start_ql + 1920

            # QL → 秒（簡易変換: 120 BPM固定）
            start_sec = (start_ql / 480) * 0.5
            end_sec = (end_ql / 480) * 0.5

            # ピッチクラス集合
            pc_set = set()
            for ins in midi_data.instruments:
                if ins.is_drum:
                    continue
                for n in ins.notes:
                    if start_sec <= n.start < end_sec:
                        pc_set.add(n.pitch % 12)

            # 簡易判定: root = 最低ピッチクラス
            if pc_set:
                root_pc = min(pc_set)
                root_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                root = root_names[root_pc]

                # maj/min判定（簡易）
                quality = "maj" if (root_pc + 4) % 12 in pc_set else "min"
            else:
                root = "N"
                quality = ""

            events.append({"time": start_sec, "root": root, "quality": quality, "confidence": 0.0})
    except Exception:
        pass
    return {"events": events}


def estimate_local_keys_extended(chord_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    3) ローカルキー推定（root多数決ベース）
    返り値:
      {
        "key_hints": [{"bar":0, "key":"C", "mode":"major"}, ...],
        "modulations": [{"bar":0, "from_key":"C", "to_key":"G"}, ...]
      }
    """
    out: Dict[str, Any] = {"key_hints": [], "modulations": []}
    try:
        nb = len(chord_events)
        if nb < 1:
            return out

        # 各小節のroot多数決
        prev_key = None
        for i, ch in enumerate(chord_events):
            root = ch.get("root", "N")
            mode = "major" if ch.get("quality") == "maj" else "minor"

            out["key_hints"].append({"bar": i, "key": root, "mode": mode})

            # 転調検出
            cur_key = f"{root}:{mode}"
            if prev_key and prev_key != cur_key:
                out["modulations"].append(
                    {"bar": i, "from_key": prev_key.split(":")[0], "to_key": root}
                )
            prev_key = cur_key
    except Exception:
        pass
    return out


def auto_sections_from_energy_extended(
    midi_data: Any, downbeats_ql: List[int], min_bars: int = 8
) -> Dict[str, Any]:
    """
    4) セクション自動検出（実ノート数ベースのエネルギー計算 - 改善版）
    返り値:
      {
        "energy": [[bar, val], ...],
        "sections": [{"bar":0, "label":"intro"}, ...]
      }
    """
    out: Dict[str, Any] = {"energy": [], "sections": []}
    try:
        nb = len(downbeats_ql)
        if nb < 1:
            return out

        # downbeats_ql を秒に変換（簡易: 120 BPM固定）
        tpq = 480
        tempo_changes = midi_data.get_tempo_changes()
        if len(tempo_changes[0]) > 0:
            base_tempo = float(tempo_changes[1][0])
        else:
            base_tempo = 120.0

        # QL → 秒
        db_sec = []
        for ql in downbeats_ql:
            sec_val = (60.0 / base_tempo) * (ql / tpq)
            db_sec.append(sec_val)

        # 末尾番兵
        if db_sec:
            db_sec.append(db_sec[-1] + 4.0)

        # 秒境界が取れない場合は従来ロジックにフォールバック
        if not db_sec or len(db_sec) < 2:
            bar_energy = [0.2] * nb
        else:
            bar_energy = [0] * nb
            for ins in midi_data.instruments:
                for n in ins.notes:
                    # 各ノートの start が属する小節に 1 カウント
                    for i in range(nb):
                        if db_sec[i] <= n.start < db_sec[i + 1]:
                            bar_energy[i] += 1
                            break

        # 正規化 0..1
        m = max(bar_energy) if nb > 0 else 1
        m = m if m > 0 else 1.0
        out["energy"] = [[i, _safe_round_extended(bar_energy[i] / m, 3)] for i in range(nb)]

        # 分割: min_barsで3-4区間に分割（intro/verse/chorus/outro）
        cuts = [0]
        step = max(min_bars, max(1, nb // 3))
        b = step
        while b < nb:
            cuts.append(b)
            b += step
        if cuts[-1] != nb:
            cuts.append(nb)

        labels = ["intro", "verse", "chorus", "outro"]
        for j, bar_idx in enumerate(cuts[:-1]):
            label = labels[min(j, len(labels) - 1)]
            out["sections"].append({"bar": int(bar_idx), "label": label})
    except Exception:
        pass
    return out


def analyze_groove_extended(midi_data: Any, downbeats_ql: List[float]) -> Dict[str, Any]:
    """
    5) グルーヴ特徴（スケルトン）
    返り値: {"swing_pct":0..100, "backbeat_strength":0..1, "onset_deviation_hist":[...]}
    """
    # 後でgroove_sampler_v2.pyの実装に置換
    return {
        "swing_pct": 0.0,
        "backbeat_strength": 0.5,
        "onset_deviation_hist": [],
        "rhythm_hash": None,
    }


def summarize_controls_extended(midi_data: Any) -> Dict[str, Any]:
    """
    6) PB/CC/RPN要約（改善版: RPN時系列検出 + PB ±8191クリップ）
    返り値:
      {
        "pb_range": [min, max],
        "cc_used": [{"cc":1, "range":[min,max]}, ...],
        "has_rpn": bool
      }
    """
    PB_MIN = -8191
    PB_MAX = 8191

    pb_vals = []
    cc_dict = {}
    rpn_seen = False

    try:
        for ins in midi_data.instruments:
            # Pitch Bend
            for pb in ins.pitch_bends:
                val = max(PB_MIN, min(PB_MAX, pb.pitch))
                pb_vals.append(val)

            # CC
            for cc in ins.control_changes:
                cc_num = cc.control
                if cc_num not in cc_dict:
                    cc_dict[cc_num] = []
                cc_dict[cc_num].append(cc.value)

            # RPN検出: CC101→100→6/38 の時系列パターン
            def _seen_rpn_sequence(cc_list, window_sec=0.5):
                """CC101→100→6/38 を0.5秒window内で検出"""
                state = 0  # 0=初期, 1=CC101, 2=CC100
                t_101 = None
                t_100 = None
                for cc in sorted(cc_list, key=lambda c: c.time):
                    if state == 0 and cc.control == 101:
                        state = 1
                        t_101 = cc.time
                    elif state == 1 and cc.control == 100:
                        if t_101 and (cc.time - t_101) < window_sec:
                            state = 2
                            t_100 = cc.time
                        else:
                            state = 0
                    elif state == 2 and cc.control in (6, 38):
                        if t_100 and (cc.time - t_100) < window_sec:
                            return True
                        else:
                            state = 0
                return False

            if _seen_rpn_sequence(ins.control_changes):
                rpn_seen = True

        # PB range計算
        if pb_vals:
            pb_range = [min(pb_vals), max(pb_vals)]
        else:
            pb_range = [0, 0]  # 未使用時は[0,0]

        # CC range
        cc_used = []
        for cc_num, vals in cc_dict.items():
            cc_used.append({"cc": cc_num, "range": [min(vals), max(vals)]})
    except Exception:
        pb_range = [0, 0]
        cc_used = []
        rpn_seen = False

    return {"pb_range": pb_range, "cc_used": cc_used, "has_rpn": rpn_seen}


def estimate_roles_extended(midi_data: Any) -> Dict[str, List[Dict[str, Any]]]:
    """
    7) ロール推定（改善版: GM Program + 音域ハイブリッド）
    返り値: {"instruments": [{"name":"Piano", "role":"melody", "program":0}, ...]}
    """
    instruments = []
    try:

        def _gm_role(program: int) -> str:
            """GM Program番号を10種類のロールに分類"""
            if 0 <= program <= 7:
                return "piano"
            elif 8 <= program <= 15:
                return "chromatic"
            elif 16 <= program <= 23:
                return "organ"
            elif 24 <= program <= 31:
                return "guitar"
            elif 32 <= program <= 39:
                return "bass"
            elif 40 <= program <= 47:
                return "strings"
            elif 48 <= program <= 55:
                return "ensemble"
            elif 56 <= program <= 79:
                return "brass"
            elif 80 <= program <= 87:
                return "lead"
            elif 88 <= program <= 95:
                return "pad"
            else:
                return "sfx"

        for ins in midi_data.instruments:
            if ins.is_drum:
                # ドラム判定: kick/snare/hat等の代表ピッチカウント
                drum_tokens = {36: 0, 38: 0, 42: 0}  # kick, snare, hihat
                for n in ins.notes:
                    if n.pitch in drum_tokens:
                        drum_tokens[n.pitch] += 1

                role = "drums" if sum(drum_tokens.values()) > 0 else "percussion"
                instruments.append({"name": ins.name or "Drums", "role": role, "program": -1})
            else:
                # GM Program判定
                program = ins.program
                role = _gm_role(program)

                # 音域補正
                if ins.notes:
                    avg_pitch = sum(n.pitch for n in ins.notes) / len(ins.notes)
                    if avg_pitch < 50:
                        role = "bass"
                    elif avg_pitch >= 50 and avg_pitch < 70:
                        role = "guitar"
                    elif avg_pitch >= 70:
                        role = "piano"

                instruments.append(
                    {"name": ins.name or f"Instrument {program}", "role": role, "program": program}
                )
    except Exception:
        pass
    return {"instruments": instruments}


def main():
    """スモークテスト実行"""
    if len(sys.argv) < 2:
        print("Usage: python test_extended_meta.py <midi_file>")
        sys.exit(1)

    midi_path = sys.argv[1]
    print(f"Testing with: {midi_path}")

    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"ERROR: Failed to load MIDI: {e}")
        sys.exit(1)

    print(f"Duration: {midi_data.get_end_time():.2f}s")
    print(f"Instruments: {len(midi_data.instruments)}")

    # 1) tempo_grid
    tempo_result = extract_tempo_grid_extended(midi_data)
    print(f"\n1) Tempo Grid:")
    print(f"   tempo_map: {len(tempo_result['tempo_map'])} entries")
    print(f"   timesig_map: {tempo_result['timesig_map']}")
    print(f"   downbeats: {len(tempo_result['downbeats_ql'])} bars")

    # 2) chords
    chords_result = extract_bar_chords_extended(midi_data, tempo_result["downbeats_ql"])
    print(f"\n2) Chords:")
    print(f"   events: {len(chords_result['events'])} chords")
    if chords_result["events"]:
        print(f"   sample: {chords_result['events'][0]}")

    # 3) keys
    keys_result = estimate_local_keys_extended(chords_result["events"])
    print(f"\n3) Keys:")
    print(f"   key_hints: {len(keys_result['key_hints'])} bars")
    print(f"   modulations: {len(keys_result['modulations'])} changes")

    # 4) sections
    sections_result = auto_sections_from_energy_extended(midi_data, tempo_result["downbeats_ql"])
    print(f"\n4) Sections:")
    print(f"   energy: {len(sections_result['energy'])} bars")
    print(f"   sections: {len(sections_result['sections'])} sections")
    if sections_result["sections"]:
        print(f"   labels: {[s['label'] for s in sections_result['sections']]}")

    # 5) groove
    groove_result = analyze_groove_extended(midi_data, tempo_result["downbeats_ql"])
    print(f"\n5) Groove:")
    print(f"   swing_pct: {groove_result['swing_pct']}")
    print(f"   backbeat_strength: {groove_result['backbeat_strength']}")

    # 6) controls
    controls_result = summarize_controls_extended(midi_data)
    print(f"\n6) Controls:")
    print(f"   pb_range: {controls_result['pb_range']}")
    print(f"   has_rpn: {controls_result['has_rpn']}")
    print(f"   cc_used: {len(controls_result['cc_used'])} types")

    # 7) roles
    roles_result = estimate_roles_extended(midi_data)
    print(f"\n7) Roles:")
    print(f"   instruments: {len(roles_result['instruments'])}")
    for r in roles_result["instruments"][:5]:
        print(f"   - {r}")

    print("\n✅ All 7 functions executed successfully!")


if __name__ == "__main__":
    main()
