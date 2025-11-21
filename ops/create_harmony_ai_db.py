#!/usr/bin/env python3
"""
Harmony AI Database Generator
song_001のchordmapデータから和声進行学習データを生成
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import uuid

# プロジェクトルートを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive_learning import ProgressionLearner


def extract_progressions_from_chordmap(chordmap_path: Path) -> list[list[str]]:
    """chordmap.jsonからコード進行を抽出"""
    with open(chordmap_path) as f:
        chordmap = json.load(f)

    events = chordmap.get("events", [])

    # 4小節ごとにコード進行を抽出
    progressions = []
    current_progression = []

    for i, event in enumerate(events):
        symbol = event.get("symbol", "")
        if symbol:
            current_progression.append(symbol)

        # 4コードごとに区切る
        if len(current_progression) >= 4:
            progressions.append(current_progression[:4])
            current_progression = current_progression[4:]

    # 残りがあれば追加
    if len(current_progression) >= 2:
        progressions.append(current_progression)

    return progressions


def analyze_emotion_from_progression(progression: list[str]) -> str:
    """コード進行から感情を推測"""
    # 簡易的な感情分類
    prog_str = " ".join(progression)

    # マイナーコードの割合
    minor_count = sum(1 for c in progression if "m" in c.lower() and "maj" not in c.lower())

    # sus, add9などの複雑なコード
    complex_count = sum(1 for c in progression if any(x in c for x in ["sus", "add", "maj7", "7"]))

    # 感情分類ロジック
    if minor_count >= len(progression) * 0.5:
        return "melancholic"
    elif complex_count >= len(progression) * 0.5:
        return "complex"
    elif any(c.startswith(("C", "G", "D", "A")) for c in progression):
        return "bright"
    else:
        return "calm"


def main():
    """メイン処理"""
    if len(sys.argv) < 2:
        print("Usage: python ops/create_harmony_ai_db.py <song_dir>")
        print("Example: python ops/create_harmony_ai_db.py data/suno_ai/suno_themesong/song_001")
        sys.exit(1)

    song_dir = Path(sys.argv[1])
    chordmap_path = song_dir / "chordmap.json"

    if not chordmap_path.exists():
        print(f"❌ chordmap.json not found: {chordmap_path}")
        sys.exit(1)

    print("=" * 60)
    print("Harmony AI Database Generator")
    print("=" * 60)

    # ProgressionLearner初期化
    db_path = Path("usage_history.db")
    learner = ProgressionLearner(db_path)
    print(f"✅ Database initialized: {db_path}")

    # コード進行抽出
    progressions = extract_progressions_from_chordmap(chordmap_path)
    print(f"✅ Extracted {len(progressions)} progressions from chordmap")

    # 学習データ作成
    session_id = str(uuid.uuid4())

    emotion_counts = defaultdict(int)

    for i, progression in enumerate(progressions):
        emotion = analyze_emotion_from_progression(progression)
        emotion_counts[emotion] += 1

        # 使用履歴として記録（高評価で記録）
        learner.record_usage(
            session_id=session_id,
            emotion=emotion,
            section=f"section_{i}",
            progression=progression,
            source="lamda",
            rating=4,  # 4/5の高評価
            metadata={"song": "song_001", "bar_index": i * 4, "auto_generated": True},
        )

    print("\n" + "=" * 60)
    print("Learning Summary")
    print("=" * 60)

    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {count} progressions")

    print(f"\n✅ Total progressions learned: {len(progressions)}")
    print(f"✅ Database saved: {db_path}")

    # 学習結果のテスト
    print("\n" + "=" * 60)
    print("Test Recommendations")
    print("=" * 60)

    for emotion in emotion_counts.keys():
        recommendations = learner.get_learned_recommendations(emotion, limit=3)
        print(f"\n{emotion}:")
        for i, rec in enumerate(recommendations[:3], 1):
            prog_str = " → ".join(rec["progression"])
            print(f"  {i}. {prog_str}")
            print(f"     (used: {rec['usage_count']}x, rating: {rec['avg_rating']:.1f})")

    print("\n" + "=" * 60)
    print("🎉 Harmony AI Database Created Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
