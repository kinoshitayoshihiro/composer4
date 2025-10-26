# ChordMap自動生成 + Scale/Mode統合ワークフロー

## 🎯 概要

このワークフローは、**Mode/Scale制約を前段・後段で二段階適用**することで、
楽曲の調性を保ちながら高品質なMIDI生成を実現します。

```
前段（ChordMap推定）: Scale Prior で誘導（α=0.25）→ 安定化
後段（Stage2生成）  : Phase31で最終ガード → 外れ音修正
```

---

## 📁 実行例

### 1. ChordMap自動生成（完了✅）

```bash
docker run --rm -v "$(pwd)":/app -w /app composer2 python generate_chordmap_with_scale.py \
  --stems data/suno_ai/suno_themesong/song_001/stemswav_001 \
  --sections data/suno_ai/suno_themesong/song_001/analysis/sections.json \
  --output data/suno_ai/suno_themesong/song_001/analysis/chordmap_auto.json \
  --genre j-pop \
  --alpha 0.25 \
  --hop-length 512
```

**結果:**
- ✅ 149小節のchordmap生成
- ✅ J-POPプリセット自動割当（intro: lydian_shimmer, chorus: ionian_citypop など）
- ✅ マルチステム（Keyboard, Guitar, Strings, Bass, Synth）から高精度chromagram抽出
- ✅ Scale Prior ブレンド（α=0.25）で転調境界を安定化

---

## 🎛️ ジャンル別プリセット

### J-POP（実行済み）
- **intro**: `lydian_shimmer` (blues=0.08)
- **verse**: `ionian_vintage` (blues=0.10)
- **chorus**: `ionian_citypop` (blues=0.12, chord-relative)
- **post_chorus**: `dorian_soul` (blues=0.10)
- **bridge**: `lydian_shimmer` (blues=0.08)
- **outro**: `aeolian_cinematic` (blues=0.10)

### Ballad/Acoustic
```bash
--genre ballad
```
- **verse**: `aeolian_dream` (blues=0.15)
- **chorus**: `dorian_soul` (blues=0.18, chord-relative)
- **bridge**: `phrygian_spice` (blues=0.10)

### J-Rock
```bash
--genre j-rock
```
- **intro**: `mixolydian_blues` (blues=0.20)
- **chorus**: `dorian_gospel` (blues=0.25, chord-relative)
- **bridge**: `phrygian_spice` (blues=0.15)

### 演歌（Enka）
```bash
--genre enka
```
- **verse**: `aeolian_dream` (blues=0.22)
- **chorus**: `aeolian_dream` (blues=0.20, chord-relative)

### City Pop
```bash
--genre citypop
```
- **intro**: `lydian_shimmer` (blues=0.10)
- **verse**: `ionian_citypop` (blues=0.10, chord-relative)
- **chorus**: `ionian_citypop` (blues=0.15, chord-relative)
- **bridge**: `lydian_shimmer` (blues=0.12, chord-relative)

---

## 🔧 パラメータ調整

### Alpha（Scale Prior ブレンド比率）

- `--alpha 0.20`: 控えめ（Chromagramをより尊重）
- `--alpha 0.25`: **推奨**（バランス型）
- `--alpha 0.30`: 強め（スケール誘導を優先）
- `--alpha 0.40`: 非常に強い（実験的）

### Hop Length（時間解像度）

- `--hop-length 256`: 高解像度（遅い、メモリ消費大）
- `--hop-length 512`: **推奨**（バランス型）
- `--hop-length 1024`: 低解像度（速い、メモリ節約）

---

## 📊 生成結果の確認

```bash
# 統計情報
docker run --rm -v "$(pwd)":/app -w /app composer2 python -c "
import json
with open('data/suno_ai/suno_themesong/song_001/analysis/chordmap_auto.json', 'r') as f:
    data = json.load(f)
print(f'Total bars: {len(data[\"chords\"])}')
print(f'Genre: {data[\"meta\"][\"genre\"]}')
print(f'Method: {data[\"meta\"][\"method\"]}')
print('\\nSample chords:')
for c in data['chords'][:10]:
    print(f'  Bar {c[\"bar\"]}: {c[\"chord\"]}')
"
```

---

## 🎼 次のステップ

### 2. ChatGPT編集（手動）

chordmap_auto.json を開いて：
- 歌詞の物語に合わせてコード差し替え
- テンション追加（9th, 11th, 13th）
- 転調タイミング調整
- セクション別のプリセット微調整

保存先: `chordmap.json`（v1版）

### 3. sections.json に preset 追記（オプション）

自動割当を上書きしたい場合：

```json
{
  "bar": 43,
  "label": "chorus",
  "key_hint": "D",
  "preset": "dorian_gospel",
  "blues": 0.28,
  "code_offsets_mode": "chord"
}
```

### 4. Stage2 MIDI生成（次回実装）

```bash
# 未実装：Stage2でのScale Mask適用
docker run --rm -v "$(pwd)":/app -w /app composer2 python modular_composer.py \
  --sections data/suno_ai/suno_themesong/song_001/analysis/sections.json \
  --chordmap data/suno_ai/suno_themesong/song_001/analysis/chordmap.json \
  --anchors data/suno_ai/suno_themesong/song_001/analysis/lyric_anchors.json \
  --output output/song_001_full.mid \
  --scale-mask-alpha 0.30  # Stage2でのマスク適用強度
```

**Stage2での適用方針:**
- **メロディ**: α=0.30-0.40（強め）
- **伴奏**: α=0.20-0.25（控えめ）
- **ベース**: α=0.15（非常に弱い、Phase31無し）

### 5. Phase31 最終ガード（自動）

`InstrumentStage2Base._voice_leading_smooth()` が自動で実行：
- 生成ノートを最近接のスケール内音に修正
- ボイスリーディングと連動
- NO-OP保証（key_hint無し → スキップ）

---

## 🎨 実行結果サマリ（song_001）

### ChordMap生成
- **Total bars**: 149
- **Genre**: j-pop
- **Alpha**: 0.25
- **Method**: multi-stem+scale_prior+template_matching

### プリセット割当
| Section | Bar | Preset | Sample Chord |
|---------|-----|--------|--------------|
| intro | 0 | lydian_shimmer | G7 |
| verse | 21 | ionian_vintage | Amaj7 |
| verse | 35 | ionian_vintage | Bmin7 |
| chorus | 43 | ionian_citypop | Bmin7 |
| post_chorus | 48 | dorian_soul | Emin7 |
| verse | 63 | ionian_vintage | Amaj7 |
| bridge | 67 | lydian_shimmer | Emaj7 |
| outro | 71 | aeolian_cinematic | Dmin7 |

### ステム重み
- **Keyboard**: 1.0
- **Guitar**: 0.9
- **Strings**: 0.8
- **Bass**: 0.7
- **Synth**: 0.6

---

## 📚 関連ドキュメント

- `ops/scale_modes.py`: Mode/Scale実装（692行、10プリセット）
- `PHASE_26_31_INTEGRATION_REPORT.md`: Phase31統合レポート
- `README_MODE_SCALE_INTEGRATION.md`: Mode/Scale機能詳細

---

## 🚀 Tips

### 転調がスムーズに検出されない場合
→ `--alpha 0.30` に上げる（スケール誘導を強化）

### コードが単調すぎる場合
→ ChatGPT編集でテンション追加、またはプリセットを変更

### 特定セクションだけ調性を変えたい
→ sections.json に個別に `preset`, `blues`, `mode` を追記

### ブルース/ジャズ系の曲
→ `--genre j-rock` + `blues=0.30` で非ダイアトニック許容度UP

---

## ✅ チェックリスト

- [x] sections.json 確定（key_hint設定済み）
- [x] マルチステムchromagram抽出（Keyboard/Guitar/Strings/Bass/Synth）
- [x] ジャンル別プリセット自動割当（j-pop）
- [x] Scale Prior ブレンド（α=0.25）
- [x] ChordMap自動生成（149小節）
- [ ] ChatGPT編集（歌詞・コード微調整）
- [ ] Stage2 MIDI生成（次回実装）
- [ ] Phase31 最終ガード検証

---

**Next Action**: chordmap_auto.json を ChatGPT で編集 → chordmap.json v1 作成 🎉
