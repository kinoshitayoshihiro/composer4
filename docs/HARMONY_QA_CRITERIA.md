# Harmony QA Criteria (和声品質保証基準)

**Version**: 1.0 (2025-11-06)

---

## KPI Gates (品質ゲート)

### 1. Cadence Score (終止感評価)

**目標**: ≥ 0.70 (70%)

**定義**: セクション境界での終止進行の完成度

**評価方法**:
- セクション末尾3小節に dominant function (V7, 7th, sus4→7)
- 次セクション冒頭に tonic function (I, 6, maj9, add9)
- 両方揃えば 1.0、dominant のみで 0.7、tonic のみで 0.5

**合格ライン**:
- ✅ ≥ 0.70: 良好（大半のセクションで終止感あり）
- ⚠️ 0.50-0.69: 要改善（ii-V-I/ii-V-i 追加推奨）
- ❌ < 0.50: 不合格（終止進行が不明瞭）

**実装例**:
```
# Verse終止 (Ab major)
bars 28-31: Bbm7 → Eb7 → Abmaj9 → Ab6/9  # ii-V-I-I6/9
# Score: 1.0 (perfect cadence)

# Pre-chorus終止 (B minor)
bars 60-63: Em7 → F#7(b9) → Bmadd9 → Bm  # iv-V7-i-i
# Score: 1.0 (perfect cadence)
```

---

### 2. Anchor Near-Change (歌詞タイミング精度)

**目標**: ≥ 0.20 (20%)

**定義**: 歌詞アンカーが小節/拍の「変わり目」(±0.15拍以内)に位置する割合

**合格ライン**:
- ✅ ≥ 0.20: 良好（歌詞が拍/小節に自然に乗る）
- ⚠️ 0.10-0.19: 要改善（タイミング微調整推奨）
- ❌ < 0.10: 不合格（歌詞が裏拍にずれている）

---

### 3. Key Confidence (調性信頼度)

**目標**: ≥ 0.15

**定義**: key_timeline での各セグメントの confidence の平均値

**合格ライン**:
- ✅ ≥ 0.25: 優秀（調性が明確）
- ✅ 0.15-0.24: 良好（拡張和音多用でも安定）
- ⚠️ 0.10-0.14: 要改善（調性判定が曖昧）
- ❌ < 0.10: 不合格（調性不明瞭）

**Note**: Tension 40-60% の場合、0.20-0.30 が妥当な範囲

---

### 4. Tension Usage (拡張和音使用率)

**目標**: 10% ≤ tension ≤ 60%

**定義**: 拡張和音（7th, 9th, 11th, 13th, sus, alt等）の使用割合

**合格ライン**:
- ✅ 40-60%: 最適（配信向けモダンサウンド）
- ✅ 20-39%: 良好（適度な拡張）
- ⚠️ 10-19%: 許容（シンプル志向）
- ⚠️ 61-80%: 許容（ジャズ/フュージョン志向）
- ❌ < 10%: 単調（拡張不足）
- ❌ > 80%: 過剰（複雑すぎる）

---

## 理論的整合性チェック

### 5. Enharmonic Consistency (エンハーモニック一貫性)

**目標**: 100%

**Flat Keys (F, Bb, Eb, Ab, Db, Gb)**:
- ✅ 使用: C, D, Eb, F, G, Ab, Bb
- ❌ 禁止: C#, D#, F#, G#, A# → Db, Eb, Gb, Ab, Bb に変換

**Sharp Keys (G, D, A, E, B, F#, C#)**:
- ✅ 使用: C, D, E, F#, G, A, B
- ❌ 禁止: Db, Eb, Gb, Ab, Bb → C#, D#, F#, G#, A# に変換

**実行**:
```bash
python ops/normalize_enharmonic.py \
    --chordmap analysis/chordmap.json \
    --key-center "Ab" \
    --backup
```

---

### 6. Quality-Symbol Alignment (quality↔symbol 完全一致)

**目標**: 100%

**検証例**:
- ✅ quality="maj9", symbol="Cmaj9" ← 正しい
- ❌ quality="maj9", symbol="Cmaj7" ← 要修正

---

## 使用方法

```bash
# 監査実行
python ops/deep_harmony_audit.py data/suno_ai/suno_themesong/song_003

# エンハーモニック正規化
python ops/normalize_enharmonic.py \
    --chordmap data/suno_ai/suno_themesong/song_003/analysis/chordmap.json \
    --key-center "Ab" \
    --backup
```

---

## song_003 達成結果 (2025-11-06)

- ✅ Cadence: **82.0%** (目標 ≥70%)
- ✅ Anchor near-change: **21.8%** (目標 ≥20%)
- ✅ Key confidence: **0.282** (目標 ≥0.15)
- ✅ Tension usage: **50.0%** (目標 10-60%)
- ✅ Enharmonic normalized: **41 events** (G#→Ab, A#→Bb, D#→Eb, C#→Db)

**全指標合格 (5/5 = 100%)** 🎉
