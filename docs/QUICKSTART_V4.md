# v4.0 クイックスタートガイド

## 🚀 3つの主要機能

### 1. コード認識（統一フォーマット）

#### 基本的な使用
```bash
# 7th版（最速・推奨）
python ops/stem_harmony_7th.py \
    --stems data/suno_ai/suno_themesong/song_001/stemswav_001 \
    --out output/chordmap.json \
    --exclude Vocals \
    --force-key C

# 7th Enhanced版（転調対応）
python ops/stem_harmony_7th_v2.py \
    --stems data/suno_ai/suno_themesong/song_001/stemswav_001 \
    --out output/chordmap.json \
    --exclude Vocals \
    --gamma-local 0.30

# 拡張和音版（sus4/add9/6th）
python ops/stem_harmony_extended.py \
    --stems data/suno_ai/suno_themesong/song_001/stemswav_001 \
    --out output/chordmap.json \
    --exclude Vocals
```

#### 出力例（全版統一）
```json
{
  "unit": "ql",
  "events": [
    {"time": 0.0, "root": "B", "quality": "min7"},
    {"time": 47.0, "root": "E", "quality": "min7"},
    {"time": 83.0, "root": "B", "quality": "min7"}
  ]
}
```

---

### 2. Stage1パイプライン（一括生成）

#### 全JSONを一度に生成
```bash
python scripts/generate_stage1_jsons.py \
    --song-dir data/suno_ai/suno_themesong/song_001 \
    --use-enhanced \
    --exclude Vocals \
    --force-key C
```

#### 生成されるファイル
```
song_001/analysis/
├── chordmap.json          # コード進行（統一形式）
├── sections.json          # セクション構造
├── lyric_anchors.json     # 歌詞アンカー
└── mix_context.json       # ミックスコンテキスト
```

#### オプション
```bash
# 拡張和音版を使用
python scripts/generate_stage1_jsons.py \
    --song-dir ... \
    --use-extended

# 特定のJSONをスキップ
python scripts/generate_stage1_jsons.py \
    --song-dir ... \
    --skip-lyrics \
    --skip-mix
```

---

### 3. 並列処理（バッチテスト）

#### 複数songを並列処理
```bash
python scripts/batch_chord_test_parallel.py \
    --base data/suno_ai/suno_themesong \
    --output results/parallel_test \
    --workers 3 \
    --use-7th \
    --force-key C
```

#### 実行結果
```
[INFO] Found 3 songs to test
[INFO] Using 3 parallel workers

[Phase 1/2] Running chord recognition...
Recognition: 100%|██████| 3/3 [04:30<00:00, 90.32s/it]
[INFO] Recognition completed: 3/3 successful

[Phase 2/2] Evaluating accuracy...
Evaluation: 100%|██████| 3/3 [00:00<00:00, 373.92it/s]

[OK] Results saved to: results/parallel_test
```

---

## 📊 バージョン比較

| バージョン | 状態数 | キャッシュ | 速度（2回目） | 用途 |
|-----------|--------|----------|--------------|------|
| **7th版** | 48 | ✅ | **0.17秒** | Jazz/R&B（推奨） |
| **7th Enhanced** | 48 | ✅ | **0.77秒** | 転調あり |
| **拡張版** | 72 | ❌ | 220秒 | Folk/Acoustic |

---

## 🎯 ジャンル別推奨

### Jazz/R&B → 7th版
```bash
python ops/stem_harmony_7th.py --stems ... --out chordmap.json --force-key C
```

### Folk/Acoustic → 拡張版
```bash
python ops/stem_harmony_extended.py --stems ... --out chordmap.json
```

### 転調が多い楽曲 → 7th Enhanced
```bash
python ops/stem_harmony_7th_v2.py --stems ... --out chordmap.json
```

---

## 💡 Tip: キャッシュ活用

### 初回実行（遅い）
```bash
python ops/stem_harmony_7th.py --stems ... --out output1.json
# 220秒
```

### 2回目以降（超高速）
```bash
python ops/stem_harmony_7th.py --stems ... --out output2.json
[CACHE] Chroma: HIT
# 0.17秒 🚀
```

### キャッシュ無効化
```bash
python ops/stem_harmony_7th.py --stems ... --out output.json --no-cache
```

### カスタムキャッシュディレクトリ
```bash
python ops/stem_harmony_7th.py --stems ... --cache-dir /path/to/cache
```

---

## 🔧 トラブルシューティング

### エラー: "No usable audio files"
→ `--exclude`オプションで全ファイルを除外している可能性

```bash
# 修正前（全除外）
--exclude Vocals --exclude Bass --exclude Drums --exclude Guitar ...

# 修正後（Vocalsのみ除外）
--exclude Vocals
```

### エラー: "sections.json not found"
→ sectionsファイルが存在しない（オプションなので問題なし）

```bash
# sectionsなしで実行（問題なし）
python ops/stem_harmony_7th.py --stems ... --out chordmap.json
```

### エラー: "Timeout"
→ 長い楽曲で初回実行時にタイムアウト

```bash
# キャッシュ作成後に再実行
python ops/stem_harmony_7th.py --stems ... --out chordmap.json
# 2回目は超高速
```

---

## 📚 関連ドキュメント

- `V4_FINAL_REPORT.md`: v4.0最終実装報告
- `V4_UNIFIED_FORMAT_COMPLETE.md`: 統一フォーマット完了報告
- `V4_IMPLEMENTATION_COMPLETE.md`: v4.0実装完了報告

---

**作成日**: 2025年10月20日  
**バージョン**: v4.0 Final  
