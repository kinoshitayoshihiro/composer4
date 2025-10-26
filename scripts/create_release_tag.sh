#!/bin/bash
# Git タグ作成 & プッシュ（本番投入準備）

set -e

echo "=== v3-guitar-ml-proba1.0 リリース準備 ==="

# 1. 変更をステージング
git add data/ab_v3_best.yaml
git add V3_EVALUATION_FINAL_REPORT.md
git add RELEASE_v3_GUITAR_ML.md
git add ml/simple_pattern_recommender.py
git add scripts/ab_test_guitar_v3.py
git add scripts/add_metadata_by_rhythm.py

# 2. コミット
git commit -m "Release: v3-guitar-ml-proba1.0

- v3単独評価への完全移行（v1退役）
- threshold=0.0（常時ML採用）
- 低確率セーフティ実装（0.15未満でsafe-kit）
- 50曲スモークテスト全KPI PASS（Accent 91.91%, Chord 83.59%, ML 100%）
- SHA256固定化（b4dbb87c...）

本番投入可能と判定。
"

# 3. タグ作成
git tag -a v3-guitar-ml-proba1.0 -m "Guitar Stage2 v3 Production Release

KPI Results (50 songs):
- Accent Score: 91.91% (target ≥65%)
- Chord Fit: 83.59% (target ≥60%)
- ML Usage: 100.00% (target ≥70%)
- Density Abs: 0.00 (target ≤1.0)

Model: stage2_guitar_v3_meta.pickle
SHA256: b4dbb87cef6a0b4bbabcc806ae0c3a796dcee9c363819d0a24b6e5e2e828c117

Status: ✓ Production Ready
"

# 4. 確認
echo ""
echo "✓ コミット完了"
echo "✓ タグ作成完了: v3-guitar-ml-proba1.0"
echo ""
echo "次のステップ:"
echo "  1. git push origin main"
echo "  2. git push origin v3-guitar-ml-proba1.0"
echo "  3. GitHub Releaseページでリリースノート公開"
echo ""
