#!/usr/bin/env python3
# ops/stage2_batch_export.py
"""
Suno stems → mix_context → Stage2 Generators を一括実行してMIDIを書き出す最小バッチ。
- 未設定はNO-OP(安全)
- 既存 Generators の apply をそのまま利用
- pretty_midi でノート/CC11/PB14 を出力(RPNはメタ保持のみ)
- スタイルプリセット自動ロード対応
- v4.1: chordmap統一スキーマ対応
"""
import argparse
import json
import random
import sys
import copy
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pretty_midi as pm
import yaml

# v4.1: スキーマ統一コンバータ
try:
    from ops.chordmap_unify import unify_chordmap_dict
    _HAS_UNIFY = True
except ImportError:
    _HAS_UNIFY = False

# PYTHONPATH を自動調整(スクリプトの親ディレクトリをパスに追加)
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ---- 必要なジェネレーターを柔軟に import（無いものはスキップ） ----
GENS = {}


def _try_import():
    try:
        from generator.piano_params_stage2 import PianoParamsStage2
        GENS["piano"] = PianoParamsStage2
    except Exception as e:
        print(f"[DEBUG] Piano import failed: {e}", file=sys.stderr)
    try:
        from generator.guitar_params_stage2 import GuitarParamsStage2
        GENS["guitar"] = GuitarParamsStage2
    except Exception as e:
        print(f"[DEBUG] Guitar import failed: {e}", file=sys.stderr)
    try:
        from generator.strings_params_stage2 import StringsParamsStage2
        GENS["strings"] = StringsParamsStage2
    except Exception as e:
        print(f"[DEBUG] Strings import failed: {e}", file=sys.stderr)
    try:
        from generator.bass_params_stage2 import BassParamsStage2
        GENS["bass"] = BassParamsStage2
    except Exception as e:
        print(f"[DEBUG] Bass import failed: {e}", file=sys.stderr)
    try:
        from generator.drums_params_stage2 import DrumsParamsStage2
        GENS["drums"] = DrumsParamsStage2
    except Exception as e:
        print(f"[DEBUG] Drums import failed: {e}", file=sys.stderr)


# ============================================================
# スタイルプリセート自動ロード ユーティリティ
# ============================================================

def _seek_role_style_file(role: str, hint: Optional[str] = None) -> Optional[Path]:
    """
    役割ごとのスタイルプリセットYAMLを探す
    
    検索優先順位:
    1. hint パスが指定されている場合はそれを使用
    2. data/{role}_style_presets.yaml
    3. presets/{role}_style_presets.yaml
    
    Returns:
        Path | None: 発見したYAMLパス、未発見時はNone
    """
    candidates = []
    
    # hint が指定されている場合
    if hint:
        hint_path = Path(hint)
        if hint_path.is_dir():
            # ディレクトリの場合は {role}_style_presets.yaml を探す
            candidates.append(hint_path / f"{role}_style_presets.yaml")
        elif hint_path.suffix in ['.yaml', '.yml']:
            # ファイルの場合はそのまま使用
            candidates.append(hint_path)
    
    # 標準的な検索パス
    candidates.extend([
        _project_root / "data" / f"{role}_style_presets.yaml",
        _project_root / "presets" / f"{role}_style_presets.yaml",
    ])
    
    for p in candidates:
        if p and p.exists():
            print(f"[INFO] Found style preset: {p}", file=sys.stderr)
            return p
    
    print(f"[WARN] No style preset file found for role={role}", file=sys.stderr)
    return None


def _load_role_style(role: str, style: str, hint: Optional[str] = None) -> Dict[str, Any]:
    """
    YAMLからstyle名のプリセートを取り出す
    
    対応する3パターン:
    1. styles.{style} キー
    2. 直下 {style} キー
    3. default フォールバック
    
    Args:
        role: 楽器役割 (piano, guitar, bass, drums, strings)
        style: スタイル名 (simple, moderate, complex, intense)
        hint: YAMLファイルパスまたはディレクトリのヒント
    
    Returns:
        Dict[str, Any]: プリセート設定、未発見時は空辞書
    """
    path = _seek_role_style_file(role, hint)
    if not path:
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}", file=sys.stderr)
        return {}
    
    # パターン1: styles.{style}
    if "styles" in data and style in data["styles"]:
        print(f"[INFO] Loaded preset: styles.{style} from {path.name}", file=sys.stderr)
        return data["styles"][style]
    
    # パターン2: 直下 {style}
    if style in data:
        print(f"[INFO] Loaded preset: {style} from {path.name}", file=sys.stderr)
        return data[style]
    
    # パターン3: default フォールバック
    if "styles" in data and "default" in data["styles"]:
        print(f"[WARN] Style '{style}' not found, using default from {path.name}", file=sys.stderr)
        return data["styles"]["default"]
    if "default" in data:
        print(f"[WARN] Style '{style}' not found, using default from {path.name}", file=sys.stderr)
        return data["default"]
    
    print(f"[WARN] No matching style '{style}' or default in {path.name}", file=sys.stderr)
    return {}


def _fallback_minimal(role: str) -> Dict[str, Any]:
    """
    プリセート未発見時の最小既定値
    Phase 11/12 が必ず動作するように最低限の設定を返す
    
    Args:
        role: 楽器役割
    
    Returns:
        Dict[str, Any]: 最小既定プリセート
    """
    base = {
        "phase11": {"enable": True},
        "phase12": {"enable": True},
        "density": {"min": 4, "max": 10},
        "register": {"lo": 48, "hi": 76},
        "rhythm": {"grid": 4, "swing": 0.0},
    }
    
    # 楽器別の調整
    if role == "bass":
        base["register"] = {"lo": 36, "hi": 60}
        base["density"] = {"min": 2, "max": 6}
    elif role == "drums":
        base["register"] = {"lo": 35, "hi": 51}
        base["density"] = {"min": 8, "max": 16}
    elif role == "guitar":
        base["register"] = {"lo": 40, "hi": 72}
        base["density"] = {"min": 4, "max": 12}
    elif role == "strings":
        base["register"] = {"lo": 48, "hi": 84}
        base["density"] = {"min": 4, "max": 10}
    
    print(f"[INFO] Using fallback minimal preset for {role}", file=sys.stderr)
    return base


# ============================================================
# セクション・データ生成
# ============================================================

def _make_section(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": meta.get("label", "verse"),
        "bar": int(meta.get("bar", 0)),
        "beat": int(meta.get("beat", 0)),
        "tempo": float(meta.get("tempo", 120.0)),
        "ql_per_bar": float(meta.get("ql_per_bar", 4.0)),
        "index": int(meta.get("index", 1)),
        "chordmap": meta.get("chordmap", {})
    }


def _run_role(role: str, section: Dict[str, Any], mix_ctx: Dict[str, Any], params: Dict[str, Any], seed: int = 1234) -> Dict[str, Any]:
    Gen = GENS[role]
    
    # プリセートを抽出してGeneratorに渡す
    # style_presets辞書を構築: {style_name: preset_config}
    style_name = params.get("style", "moderate")
    style_config = {k: v for k, v in params.items() if k not in ["style", "export"]}
    
    # Generatorを初期化（style_presetsを渡す）
    gen_style_presets = {style_name: style_config}
    g = Gen(style_presets=gen_style_presets)  # instrument_name引数は不要（各クラスで固定）
    
    # apply メソッドを呼び出し
    try:
        # 新しいシグネチャ: apply(part, section_meta, mix_context, overrides)
        from music21 import stream
        part = stream.Part()
        
        print(f"[DEBUG] Before apply: section={section.get('label')}, bar={section.get('bar')}", file=sys.stderr)
        print(f"[DEBUG] Generator.style_presets: {list(g.style_presets.keys())}", file=sys.stderr)
        print(f"[DEBUG] Params: {list(params.keys())}", file=sys.stderr)
        print(f"[DEBUG] mix_ctx.activity keys: {list(mix_ctx.get('activity', {}).keys())}", file=sys.stderr)
        
        # overridesにstyle名を渡す
        overrides = {"style": style_name, **params}
        part = g.apply(part, section, mix_ctx, overrides, seed=seed)
        
        print(f"[DEBUG] After apply: part.notes={len(list(part.flatten().notes))}", file=sys.stderr)
        
    except Exception as e:
        print(f"[DEBUG] Apply failed for {role}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        # フォールバック: 空のパート
        from music21 import stream
        part = stream.Part()
    
    # music21.stream.Part の場合は辞書に変換
    try:
        from music21 import stream
        if isinstance(part, stream.Part):
            # music21.Part から notes と controls を抽出
            notes_list = []
            for n in part.flatten().notes:
                notes_list.append({
                    "pitch": n.pitch.midi,
                    "vel": n.volume.velocity if hasattr(n.volume, 'velocity') else 64,
                    "on_ql": float(n.offset),
                    "dur_ql": float(n.duration.quarterLength)
                })
            
            # comment から export_name を抽出
            export_name = ""
            if hasattr(part, 'comment') and part.comment:
                for token in part.comment.split('|'):
                    if token.startswith('export_name='):
                        export_name = token.split('=', 1)[1]
                        break
            
            return {
                "notes": notes_list,
                "controls": {
                    "meta": {"export_name": export_name},
                    "cc": {},
                    "pb14": []
                }
            }
    except ImportError:
        pass
    
    if isinstance(part, dict):
        return part
    return {"notes": getattr(part, "notes", []), "controls": getattr(part, "controls", {})}


def _ql_to_sec(ql: float, bpm: float) -> float:
    # 1拍=四分音符=1 QL とみなし、秒 = 60/bpm * ql
    return float(ql) * (60.0 / max(1e-6, bpm))


def _export_part_to_midi(part: Dict[str, Any], out_path: Path, bpm: float, program: int = 0, is_drum: bool = False):
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0 if is_drum else program, is_drum=is_drum, name=str(out_path.stem))
    notes = part.get("notes") or []
    for n in notes:
        pitch = int(n.get("pitch", 60))
        vel = int(n.get("vel", 64))
        on_ql = float(n.get("on_ql", n.get("off_ql", 0.0)))
        dur_ql = float(n.get("dur_ql", 0.25))
        start = _ql_to_sec(on_ql, bpm)
        end = max(start + _ql_to_sec(max(0.0, dur_ql), bpm), start + 1e-3)
        inst.notes.append(pm.Note(velocity=max(1, min(127, vel)), pitch=pitch, start=start, end=end))
    
    # CC11 / PB14
    ctrl = (part.get("controls") or {})
    cc = (ctrl.get("cc") or {})
    cc11 = cc.get(11) or ctrl.get("cc11") or []
    for ev in cc11:
        t = _ql_to_sec(float(ev.get("time", 0.0)), bpm)
        v = int(ev.get("value", 0))
        inst.control_changes.append(pm.ControlChange(number=11, value=max(0, min(127, v)), time=t))
    
    pb = ctrl.get("pb14") or ctrl.get("pitchbend") or []
    for ev in pb:
        t = _ql_to_sec(float(ev.get("time", 0.0)), bpm)
        v = int(ev.get("value", 0))
        inst.pitch_bends.append(pm.PitchBend(pitch=v, time=t))  # ±8191 スケール対応
    
    midi.instruments.append(inst)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(out_path))


def main():
    _try_import()
    ap = argparse.ArgumentParser(description="Stage2 Batch Export: mix_context + sections → MIDI files")
    ap.add_argument("--mix", required=True, help="mix_context JSON path")
    ap.add_argument("--sections", required=True, help="sections JSON path")
    ap.add_argument("--roles", default="piano,guitar,strings,bass,drums", help="Comma-separated roles to export")
    ap.add_argument("--style", default="moderate", help="Style preset name (simple/moderate/complex/intense)")
    ap.add_argument("--style-presets", default="", help="Path to style preset YAML file or directory (auto-search if empty)")
    ap.add_argument("--outdir", default="out_midi", help="Output directory for MIDI files")
    ap.add_argument("--name-fmt", default="{date}_{seq}_{project}_{role}_{section}_{style}", help="Name format template")
    ap.add_argument("--project", default="", help="Project tag for naming")
    ap.add_argument("--date-fmt", default="%Y%m%d", help="Date format for naming")
    ap.add_argument("--seq-width", type=int, default=2, help="Sequence number width (zero-padded)")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed")
    args = ap.parse_args()

    mix_ctx = json.loads(Path(args.mix).read_text(encoding="utf-8"))
    sections = json.loads(Path(args.sections).read_text(encoding="utf-8"))
    
    # --- 追加: 正規化（辞書配列→タプル配列 / chordmapキー数値化）---
    def _norm_activity(m):
        """activity: [{bar, energy}] → [(bar, energy)] に正規化"""
        act = (m.get("activity") or {})
        out = {}
        for role, arr in (act.items()):
            norm = []
            for it in (arr or []):
                # {bar:.., energy:..} or {bar:.., level:..} or (bar, level)
                if isinstance(it, dict):
                    b = int(it.get("bar", 0))
                    v = float(it.get("energy", it.get("level", 0.0)))
                    norm.append((b, v))
                elif isinstance(it, (list, tuple)) and len(it) >= 2:
                    norm.append((int(it[0]), float(it[1])))
            out[role] = norm
        m["activity"] = out
        return m

    def _norm_sections(arr):
        """chordmap: {"0.0": ...} → {0.0: ...} に正規化 + v4.1統一スキーマ対応"""
        out = []
        for s in (arr or []):
            s2 = copy.deepcopy(s)
            cm = s2.get("chordmap") or {}
            
            # v4.1: スキーマ統一（秒表記/配列/辞書ゆれを吸収）
            if _HAS_UNIFY:
                try:
                    cm_unified = unify_chordmap_dict(
                        cm,
                        to_unit="ql",
                        snap_ql=0.25,  # 16分音符グリッド
                        merge_N=True,
                        min_N_ql=2.0,  # 最小2QL（8分音符）
                        glue_same_root=True,
                    )
                    # events形式を旧形式（QL辞書）に変換して互換性維持
                    if "events" in cm_unified:
                        cm2 = {}
                        for e in cm_unified["events"]:
                            t = float(e["time"])
                            root = e["root"]
                            qual = e["quality"]
                            # root/qualityをchord記号に再結合
                            chord_sym = root if qual == "" or qual == "maj" else f"{root}{qual}"
                            cm2[t] = chord_sym
                        s2["chordmap"] = cm2
                    else:
                        s2["chordmap"] = cm
                except Exception as ex:
                    print(f"[WARN] chordmap unify failed: {ex}", file=sys.stderr)
                    # フェイルセーフ：旧処理
                    if isinstance(cm, dict):
                        cm2 = {}
                        for k, v in cm.items():
                            try:
                                cm2[float(k)] = v
                            except Exception:
                                continue
                        s2["chordmap"] = cm2
            else:
                # unify未導入時の旧処理
                if isinstance(cm, dict):
                    cm2 = {}
                    for k, v in cm.items():
                        try:
                            cm2[float(k)] = v
                        except Exception:
                            continue
                    s2["chordmap"] = cm2
            
            out.append(s2)
        return out

    mix_ctx = _norm_activity(mix_ctx)
    sections = _norm_sections(sections)
    
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]

    seq = 0
    for smeta in sections:
        section = _make_section(smeta)
        bpm = float(section["tempo"])
        for role in roles:
            if role not in GENS:
                print(f"[SKIP] {role} - generator not available")
                continue
            
            # ============================================================
            # スタイルプリセート自動ロード
            # ============================================================
            style_cfg = _load_role_style(role, args.style, args.style_presets or None)
            
            # プリセート未発見時はフォールバック最小既定を使用
            if not style_cfg:
                style_cfg = _fallback_minimal(role)
            
            # params をプリセートベースで構築
            params = {
                **style_cfg,  # プリセート設定を展開
                "style": args.style,
                "export": {
                    "name_fmt": args.name_fmt,
                    "date_fmt": args.date_fmt,
                    "seq_width": args.seq_width,
                    "project_tag": args.project,
                    "style_tag": args.style
                }
            }
            
            print(f"[DEBUG] {role} params keys: {list(params.keys())}", file=sys.stderr)
            
            part = _run_role(role, section, mix_ctx, params, seed=args.seed)
            
            # export_name が meta に入っている想定（Phase28対応）
            meta = ((part.get("controls") or {}).get("meta") or {})
            
            # 自己完結（万一未設定でもフォールバック）
            seq += 1
            seq_str = f"{seq:0{args.seq_width}d}"
            from datetime import datetime
            name = meta.get("export_name") or args.name_fmt.format(
                idx=int(section["index"]), role=role,
                section=str(section["label"]).lower(),
                seq=seq_str, date=datetime.now().strftime(args.date_fmt),
                project=args.project, style=args.style
            )
            
            out = Path(args.outdir) / f"{name}.mid"
            _export_part_to_midi(part, out, bpm, program=0, is_drum=(role == "drums"))
            print(f"[EXPORT] {out}")


if __name__ == "__main__":
    main()
