#!/bin/bash
# GitHub Release 作成ガイド（手動実行用）

echo "=== GitHub Release 作成手順 ==="
echo ""
echo "1. GitHub Releaseページを開く:"
echo "   https://github.com/kinoshitayoshihiro/composer4/releases/new?tag=v3-guitar-ml-proba1.0"
echo ""
echo "2. リリース情報を入力:"
echo "   - Tag: v3-guitar-ml-proba1.0 (自動選択済み)"
echo "   - Release title: 🎸 Guitar Stage2 v3 ML-Direct Production Release"
echo "   - Description: GITHUB_RELEASE_v3_GUITAR_ML.md の内容をコピー&ペースト"
echo ""
echo "3. リリースノートのコピー:"
echo "   cat GITHUB_RELEASE_v3_GUITAR_ML.md | pbcopy"
echo "   (macOS: 自動的にクリップボードにコピーされます)"
echo ""
echo "4. オプション設定:"
echo "   [ ] Set as a pre-release (チェックしない)"
echo "   [x] Set as the latest release (チェック推奨)"
echo ""
echo "5. 'Publish release' ボタンをクリック"
echo ""
echo "========================================="
echo ""

# リリースノートをクリップボードにコピー（macOS）
if command -v pbcopy &> /dev/null; then
    cat GITHUB_RELEASE_v3_GUITAR_ML.md | pbcopy
    echo "✓ リリースノートをクリップボードにコピーしました"
    echo ""
fi

# ブラウザで開く（オプション）
read -p "ブラウザでGitHub Releaseページを開きますか？ (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v open &> /dev/null; then
        open "https://github.com/kinoshitayoshihiro/composer4/releases/new?tag=v3-guitar-ml-proba1.0"
        echo "✓ ブラウザでGitHub Releaseページを開きました"
    else
        echo "⚠️  'open'コマンドが見つかりません。手動でURLを開いてください:"
        echo "   https://github.com/kinoshitayoshihiro/composer4/releases/new?tag=v3-guitar-ml-proba1.0"
    fi
fi

echo ""
echo "========================================="
echo "GitHub Release作成後の次のステップ:"
echo ""
echo "1. Canary Deployment (Week 1)"
echo "   - 10曲フル生成 & ヒアリング確認"
echo "   - KPIモニタリング開始"
echo ""
echo "2. KPIダッシュボード構築"
echo "   - Grafana/Prometheus連携"
echo "   - リアルタイム異常検知"
echo ""
echo "3. Shadow Testing (Week 2)"
echo "   - v3 vs v1 並行運用"
echo "   - 遅延監視（p95 < 100ms）"
echo ""
echo "========================================="
