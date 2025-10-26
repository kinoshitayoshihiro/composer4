#!/usr/bin/env python3
"""
LOCAL LAMDA バンドル健全性チェック

混入・ID重複・1/4比率・形状・Stage2合流の5軸でQA。

Usage:
    python scripts/local_lamda/qa_local_lamda.py \
      --stage1-root output/stage1 \
      --local-dir data/LOCAL_LAMDA \
      --stage2-json-dir output/stage2_local_test/json
"""
import argparse
import pickle
import json
import glob
import csv
from pathlib import Path
from collections import Counter, defaultdict


def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser(
        description="QA for LOCAL LAMDA bundles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--stage1-root", required=True, help="Stage1 root directory")
    ap.add_argument(
        "--local-dir",
        required=True,
        help="Directory containing LOCAL_*.pickle and LOCAL_ID_MAP.csv",
    )
    ap.add_argument(
        "--stage2-json-dir",
        required=False,
        help="Stage2 JSON output directory (optional)",
    )
    args = ap.parse_args()

    local_dir = Path(args.local_dir)

    print("\n" + "=" * 60)
    print("🔎 LOCAL LAMDA Bundle QA")
    print("=" * 60)

    # 必須ファイル存在チェック
    required_files = [
        "LOCAL_KILO_CHORDS_DATA.pickle",
        "LOCAL_SIGNATURES_DATA.pickle",
        "LOCAL_TOTALS.pickle",
        "LOCAL_ID_MAP.csv",
    ]

    print("\n1️⃣  File Existence Check")
    for fname in required_files:
        exists = (local_dir / fname).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {fname}")
        if not exists and fname != "LOCAL_ID_MAP.csv":  # ID_MAPは後で追加予定
            print(f"     ⚠️  Missing required file: {fname}")

    # METAは複数シャードの可能性
    meta_files = list(local_dir.glob("LOCAL_META_DATA*.pickle"))
    if meta_files:
        print(f"  ✅ LOCAL_META_DATA*.pickle ({len(meta_files)} shard(s))")
    else:
        print("  ❌ LOCAL_META_DATA*.pickle (missing)")

    # ID map 健全性
    print("\n2️⃣  ID Map Integrity")
    id_map_path = local_dir / "LOCAL_ID_MAP.csv"
    if id_map_path.exists():
        id_rows = list(csv.DictReader(open(id_map_path)))
        # カラム名は "relative_path" または "rel_path"
        rels = [r.get("relative_path") or r.get("rel_path", "") for r in id_rows]
        ids = [r["local_id"] for r in id_rows]

        dupe_rel = [k for k, v in Counter(rels).items() if v > 1]
        dupe_id = [k for k, v in Counter(ids).items() if v > 1]

        if dupe_rel:
            print(f"  ❌ Duplicated rel_path: {len(dupe_rel)} ({dupe_rel[:5]})")
        else:
            print("  ✅ No duplicated rel_path")

        if dupe_id:
            print(f"  ❌ Duplicated local_id: {len(dupe_id)} ({dupe_id[:5]})")
        else:
            print("  ✅ No duplicated local_id")

        # temp/quarantine混入チェック
        bad = [
            r
            for r in rels
            if any(x in r.split("/") for x in ("temp", "quarantine", ".cache", ".trash"))
        ]
        if bad:
            print(f"  ❌ Mixed temp/quarantine: {len(bad)} files")
            for b in bad[:5]:
                print(f"     - {b}")
        else:
            print("  ✅ No temp/quarantine mixed")
    else:
        print("  ⚠️  LOCAL_ID_MAP.csv not found (will be generated in fixed version)")

    # KILO 形状
    print("\n3️⃣  KILO Shape Check")
    kilo_path = local_dir / "LOCAL_KILO_CHORDS_DATA.pickle"
    if kilo_path.exists():
        try:
            kilo = load_pickle(kilo_path)
            if isinstance(kilo, list):
                # [[file_id, payload], ...] 形式
                if kilo:
                    some_id, some_payload = kilo[0]
                    tokens = some_payload.get("tokens", [])
                    print(f"  ✅ KILO entries: {len(kilo)}")
                    print(f"  ✅ Sample bars: {len(tokens)} for {some_id[:16]}...")
            elif isinstance(kilo, dict):
                # {file_id: payload} 形式
                some_id, some_payload = next(iter(kilo.items()))
                if isinstance(some_payload, dict):
                    tokens = some_payload.get("tokens", [])
                else:
                    tokens = some_payload if isinstance(some_payload, list) else []
                print(f"  ✅ KILO entries: {len(kilo)}")
                print(f"  ✅ Sample bars: {len(tokens)} for {some_id[:16]}...")
        except Exception as e:
            print(f"  ❌ KILO load error: {e}")
    else:
        print("  ❌ KILO file not found")

    # SIGNATURES & 1/4救済確認
    print("\n4️⃣  Signatures & 1/4 Rescue Check")
    sig_path = local_dir / "LOCAL_SIGNATURES_DATA.pickle"
    if sig_path.exists():
        try:
            sig = load_pickle(sig_path)
            sig_vals = []

            if isinstance(sig, list):
                # [[file_id, [[sig_id, count], ...]], ...]
                for item in sig:
                    if len(item) >= 2:
                        sig_list = item[1]
                        for s_item in sig_list:
                            if isinstance(s_item, list) and len(s_item) >= 2:
                                sig_vals.append(str(s_item[0]))
            elif isinstance(sig, dict):
                # {file_id: [[sig_id, count], ...]}
                for v in sig.values():
                    if isinstance(v, list):
                        for s_item in v:
                            if isinstance(s_item, list) and len(s_item) >= 2:
                                sig_vals.append(str(s_item[0]))

            c = Counter(sig_vals)
            n_one_four = sum(v for k, v in c.items() if "1/4" in str(k) or k == "1/4")
            n_total = sum(c.values())
            ratio = 0.0 if n_total == 0 else n_one_four / n_total

            print(f"  📏 Total signatures: {n_total}")
            print(f"  📏 '1/4' ratio: {ratio:.4f} (target < 0.005)")
            if ratio >= 0.005:
                print("     ⚠️  High 1/4 ratio - consider applying 1/4 rescue patch")
            else:
                print("     ✅ 1/4 ratio is acceptable")

            # 上位5つの拍子
            print("  📊 Top 5 signatures:")
            for sig_id, count in c.most_common(5):
                print(f"     - {sig_id}: {count} ({count/n_total*100:.1f}%)")

        except Exception as e:
            print(f"  ❌ SIGNATURES load error: {e}")
    else:
        print("  ❌ SIGNATURES file not found")

    # TOTALS 形状
    print("\n5️⃣  TOTALS Shape Check")
    totals_path = local_dir / "LOCAL_TOTALS.pickle"
    if totals_path.exists():
        try:
            totals = load_pickle(totals_path)
            # 期待形式: {"pitch_hist_256": [...], "dur_hist_256": [...], "vel_hist_256": [...]}
            expected_keys = ["pitch_hist_256", "dur_hist_256", "vel_hist_256"]
            ok = all(k in totals for k in expected_keys)
            if ok:
                ok = all(len(totals[k]) == 256 for k in expected_keys)

            if ok:
                print("  ✅ TOTALS shape 256 OK")
                # 統計表示
                for k in expected_keys:
                    total_count = sum(totals[k])
                    print(f"     - {k}: {total_count} total counts")
            else:
                print("  ❌ TOTALS shape mismatch")
                for k in expected_keys:
                    if k in totals:
                        print(f"     - {k}: len={len(totals[k])}")
                    else:
                        print(f"     - {k}: missing")
        except Exception as e:
            print(f"  ❌ TOTALS load error: {e}")
    else:
        print("  ❌ TOTALS file not found")

    # META の代表値チェック
    print("\n6️⃣  META Entries Check")
    metas = []
    for pkl in meta_files:
        if pkl.exists():
            try:
                metas.append(load_pickle(pkl))
            except Exception as e:
                print(f"  ⚠️  Failed to load {pkl.name}: {e}")

    n_meta = sum(len(m) for m in metas)
    print(f"  ✅ META entries: {n_meta}")

    # Stage2 JSON との合流チェック（任意）
    if args.stage2_json_dir:
        print("\n7️⃣  Stage2 Integration Check")
        json_dir = Path(args.stage2_json_dir)
        if json_dir.exists():
            used = Counter()
            sample_size = 0
            for jp in glob.glob(str(json_dir / "*.json"))[:5000]:
                try:
                    with open(jp) as f:
                        d = json.load(f)
                    src = d.get("lamda_source", "none")
                    if src == "none":
                        # 後方互換: chordmap_external があれば origin
                        if d.get("chordmap_external"):
                            src = "origin"
                    used[src] += 1
                    sample_size += 1
                except Exception:
                    continue

            print(f"  🔎 lamda_source histogram (first {sample_size} files):")
            for src, count in used.most_common():
                pct = count / sample_size * 100 if sample_size > 0 else 0
                print(f"     - {src}: {count} ({pct:.1f}%)")

            # 評価
            local_count = used.get("local", 0)
            if local_count > 0:
                print("  ✅ LOCAL sources are being used")
            else:
                print("  ⚠️  No LOCAL sources detected in Stage2 output")
        else:
            print(f"  ⚠️  Stage2 JSON directory not found: {json_dir}")

    print("\n" + "=" * 60)
    print("✅ QA Finished")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
