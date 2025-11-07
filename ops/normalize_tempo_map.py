#!/usr/bin/env python3
"""
normalize_tempo_map.py - ワンショットtempo_map.json正規化ツール

既存のtempo_map.jsonをv2形式（points配列）に正規化します。

使用例:
    python ops/normalize_tempo_map.py \\
        data/suno_ai/suno_themesong/song_002/tempo_map.json \\
        data/suno_ai/suno_themesong/song_002/tempo_map.json  # 上書き

推奨スキーマ（v2形式）:
    {
      "points": [
        {"bar": 0, "beat": 0.0, "bpm": 89.3}
      ],
      "ppq": 480
    }
"""
import json
import sys
from pathlib import Path


def normalize_tempo_map(src_path: Path, dst_path: Path):
    """tempo_map.jsonを正規化してv2形式で保存"""
    data = json.loads(src_path.read_text(encoding="utf-8"))

    # points配列チェック
    points = data.get("points") or data.get("events") or data.get("map")

    # tempo_points形式（[time_sec, bpm]のリスト）チェック
    tempo_points_list = data.get("tempo_points")
    if tempo_points_list and isinstance(tempo_points_list, list):
        if tempo_points_list and isinstance(tempo_points_list[0], list):
            # [[time_sec, bpm], ...] 形式
            # 平均BPMを計算（簡易処理）
            bpms = [float(tp[1]) for tp in tempo_points_list if len(tp) >= 2]
            avg_bpm = sum(bpms) / len(bpms) if bpms else 120.0
            points = [{"bar": 0, "beat": 0.0, "bpm": avg_bpm}]
            print(f"tempo_points形式検出: {len(tempo_points_list)}点 → 平均BPM {avg_bpm:.1f}")

    if not points:
        # グローバルBPM→1点にフォールバック
        bpm = data.get("bpm") or data.get("tempo_bpm") or data.get("qpm") or 120.0
        points = [{"bar": 0, "beat": 0.0, "bpm": float(bpm)}]
        print(f"グローバルBPM検出: {bpm:.1f} → 1点")

    # 正規化
    norm_points = []
    global_bpm = data.get("bpm") or data.get("tempo_bpm") or data.get("qpm") or 120.0
    for p in points:
        if isinstance(p, dict):
            norm_points.append(
                {
                    "bar": int(p.get("bar", 0)),
                    "beat": float(p.get("beat", p.get("start_beat", 0.0))),
                    "bpm": float(p.get("bpm", p.get("tempo_bpm", p.get("qpm", global_bpm)))),
                }
            )

    # v2形式で保存
    out = {"points": norm_points, "ppq": data.get("ppq", 480)}
    dst_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ 正規化完了: {len(norm_points)}点 → {dst_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ops/normalize_tempo_map.py <src_path> <dst_path>")
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    if not src.exists():
        print(f"❌ 入力ファイル未検出: {src}")
        sys.exit(1)

    normalize_tempo_map(src, dst)
