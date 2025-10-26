#!/usr/bin/env python3
"""
LOCAL_SIGNATURES_DATA.pickle ビルダー

ステムMIDIから拍子情報を抽出してSIGNATURES互換形式で生成。
公式SIGNATURESと同じAPI（lamda_sources）で読める。

出力形式:
[file_id, [
    [signature_id, count],
    ...
]]

signature_id: signature_id_map.yamlで変換（例: 4/4 -> 211）
不明な拍子は "unknown:<sig>" として記録
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
from collections import Counter
from pathlib import Path

import pretty_midi as pm
import yaml

# 除外ディレクトリ（temp/quarantine/shards等）
EXCLUDE_DIRS = {"temp", "quarantine", "shards", ".git", "__pycache__"}


def iter_midis(root: Path):
    """除外ディレクトリをスキップしてMIDIファイルを収集"""
    for p in root.rglob("*"):
        if p.suffix.lower() not in (".mid", ".midi"):
            continue
        if any(excl in p.parts for excl in EXCLUDE_DIRS):
            continue
        yield p


def make_id(p: str) -> str:
    """安定したLOCAL IDを生成（4資源共通）"""
    h = hashlib.sha1()
    h.update((p + "|" + str(os.path.getsize(p))).encode("utf-8"))
    return "LOCAL:" + h.hexdigest()[:20]


def sig_str(ts: pm.TimeSignature) -> str:
    """TimeSignatureオブジェクトを文字列化"""
    return f"{ts.numerator}/{ts.denominator}"


def rescue_1_4_signature(counter: Counter) -> Counter:
    """
    1/4救済: 1/4が存在し4/4も存在する場合、1/4を4/4に統合。
    1/4のみの場合は変更しない（本物の1/4曲）。
    
    目標: 1/4比率を0.5%未満に削減
    """
    if "1/4" not in counter:
        return counter
    
    # 4/4が存在する場合のみ救済
    if "4/4" in counter:
        # 1/4のカウントを4/4に統合
        counter["4/4"] += counter["1/4"]
        del counter["1/4"]
    
    # 1/4のみの場合は本物の1/4曲として保持
    return counter


def main():
    ap = argparse.ArgumentParser(
        description="LOCAL_SIGNATURES_DATA builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--midi-root",
        required=True,
        help="MIDI root directory (recursive search)",
    )
    ap.add_argument(
        "--out-pickle",
        required=True,
        help="Output pickle path (LOCAL_SIGNATURES_DATA.pickle)",
    )
    ap.add_argument(
        "--sig-map-yaml",
        default=None,
        help="signature_id_map.yaml path (optional)",
    )
    args = ap.parse_args()

    # signature_id_map.yamlをロード
    sig_map = {}
    if args.sig_map_yaml and os.path.exists(args.sig_map_yaml):
        with open(args.sig_map_yaml, "r", encoding="utf-8") as f:
            sig_map = yaml.safe_load(f) or {}
        print(f"📚 Signature map loaded: {len(sig_map)} signatures")

    # MIDI収集（除外ディレクトリをスキップ）
    midi_root = Path(args.midi_root)
    midi_files = sorted(set(iter_midis(midi_root)))
    print(f"📂 Found {len(midi_files)} MIDI files (excluding temp/quarantine/shards)")

    out = []
    for i, mp in enumerate(midi_files, 1):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(midi_files)}...")

        try:
            m = pm.PrettyMIDI(str(mp))
            ts = m.time_signature_changes

            if not ts:
                # デフォルト4/4
                c = Counter({"4/4": 1})
            else:
                c = Counter([sig_str(x) for x in ts])
            
            # 1/4救済を適用
            c = rescue_1_4_signature(c)

            rows = []
            for s, count in c.items():
                # signature_id_map.yamlで変換
                sid = sig_map.get(s, f"unknown:{s}")
                rows.append([sid, int(count)])

            out.append([make_id(str(mp)), rows])

        except Exception as e:
            print(f"[skip] {mp}: {e}")

    with open(args.out_pickle, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✅ Wrote {len(out)} entries -> {args.out_pickle}")


if __name__ == "__main__":
    main()
