# 統合MIDIクリーニングパイプライン

**共通クリーニング → 楽器別クリーニング → 検証 → 層別分割**

---

## 📋 概要

このシステムは、LAMDa方式の重複除去を拡張し、**全楽器に対応した統合クリーニングパイプライン**を提供します。

### 🎯 設計方針

- **非破壊**: 安全な修正のみ自動適用、危険な修正は隔離
- **決定論**: SHA1ハッシュで順序固定、seed固定で再現可能
- **監査性**: 各ファイルに `.meta.json` を生成 (before/after統計、reason_codes)
- **段階ゲート**: clean → validate → gate → split の順にフィルタリング

---

## 🏗️ パイプライン全体像

```
raw_midi/
  ↓
(1) 共通クリーニング (common_clean)
  - 無効イベント除去
  - テンポ/拍子正規化
  - 範囲外Pitch検出
  ↓
(2) 楽器別クリーニング (piano/guitar/bass/strings/drums)
  - Piano: ペダル正規化、片手重複緩和
  - Guitar: ストラム検出、12弦ノイズ除去
  - Bass: モノフォニック保証、大跳躍検出
  - Strings: レガート検出、スタッカート洪水警告
  - Drums: グリッド整合性、キックオンビート率
  ↓
(3) 検証 & Quality Gates (validate_and_gate)
  - quality_gates.yaml の基準で検証
  - 違反ファイルを隔離
  ↓
(4) 層別分割 (prepare_splits)
  - SHA1決定論的分割
  - style × tempo × density で層別化
  - train / val / test に分割
```

---

## 🚀 使用方法

### Step 1: クリーニング

```bash
# Piano
python scripts/clean_midi.py \
  --in data/lamda/raw/piano \
  --out data/lamda/clean/piano \
  --instrument piano \
  --quarantine data/lamda/quarantine/piano

# Guitar
python scripts/clean_midi.py \
  --in data/lamda/raw/guitar \
  --out data/lamda/clean/guitar \
  --instrument guitar \
  --quarantine data/lamda/quarantine/guitar

# 他の楽器も同様...
```

**出力:**
- `data/lamda/clean/{instrument}/` - クリーニング済みMIDI
- `data/lamda/clean/{instrument}/*.meta.json` - メタデータ
- `data/lamda/quarantine/{instrument}/` - 隔離ファイル
- `data/lamda/{instrument}_clean_report.json` - レポート

### Step 2: 検証

```bash
python scripts/validate_and_gate.py \
  --in data/lamda/clean/piano \
  --gates configs/quality_gates/quality_gates.yaml \
  --report reports/piano_validation_report.json
```

**出力:**
- `reports/piano_validation_report.json` - 検証レポート

### Step 3: 分割

```bash
python scripts/prepare_splits.py \
  --in data/lamda/clean/piano \
  --out data/lamda/splits/piano \
  --seed 1234
```

**出力:**
- `data/lamda/splits/piano/train/` - 訓練セット (80%)
- `data/lamda/splits/piano/val/` - 検証セット (10%)
- `data/lamda/splits/piano/test/` - テストセット (10%)
- `data/lamda/splits/piano/split_summary.json` - 分割統計

---

## 📊 メタデータ構造

各MIDIファイルには `.meta.json` が生成されます:

```json
{
  "clean_actions": [
    "remove_invalid_notes:5",
    "deduped_chord_fragments:3"
  ],
  "reason_codes": [
    "pedal_excessive"
  ],
  "tempo": 120.0,
  "tempo_estimated": false,
  "time_signature": "4/4",
  "bars": 16.0,
  "notes": 256,
  "density": 8.5,
  "duration_sec": 30.2,
  
  // 楽器別メトリクス (Piano)
  "pedal_sustain_ratio": 0.65,
  "hand_separation": 18.5,
  "velocity_std": 22.3,
  "velocity_mean": 75.2
}
```

---

## ⚙️ Quality Gates設定

`configs/quality_gates/quality_gates.yaml`:

```yaml
common:
  min_duration_sec: 2.0
  max_duration_sec: 600.0
  min_notes: 8
  max_notes: 50000

piano:
  min_notes: 20
  max_pedal_sustain_ratio: 0.95
  min_hand_separation: 6

guitar:
  min_pitch: 40
  max_pitch: 84
  max_arpeggio_glitch_ratio: 0.3

bass:
  max_leap_excess_ratio: 0.4
  max_grid_off_ratio: 0.4

strings:
  max_staccato_flood_ratio: 0.8
  max_chord_spread_semitones: 30

drums:
  min_kick_on_beat_rate: 0.3
  max_grid_off_std_ms: 30
```

---

## 🔧 楽器別クリーニング詳細

### Piano (`cleaners/piano.py`)
- **ペダル正規化**: CC64の断片を統合、過剰ペダル(>90%)を警告
- **片手重複緩和**: <5ms以内の同pitch重複を最長ノートに統合
- **音域分析**: 左手/右手の分離度を計算

### Guitar (`cleaners/guitar.py`)
- **ストラム検出**: 0-60ms以内の3音以上の和音群
- **12弦ノイズ除去**: 完全8ve±5ms以内のダブリングを統合
- **過密アルペジオ**: IOI<15msが20%超で警告
- **音域チェック**: E2(40) - C7(84)

### Bass (`cleaners/bass.py`)
- **モノフォニック保証**: 同時鳴りは最長ノートを残す
- **大跳躍検出**: >12半音の連発が30%超で警告
- **グリッド整合性**: 拍±30msのずれ率を計算

### Strings (`cleaners/strings.py`)
- **レガート検出**: 隣接ノートのギャップ±20msをレガートとカウント
- **スタッカート洪水**: ≤120msのノートが70%超で警告
- **和音広がり**: 同時鳴りの音域が24半音超で警告

### Drums (`cleaners/drums.py`)
- **グリッド整合性**: 16分音符グリッドからのずれ
- **キックオンビート率**: Kick(35, 36)が拍頭±5%以内の割合
- **既存LAMDa統合**: Stage1メトリクスと互換

---

## ✅ 検証結果 (2024年10月15日)

**決定性検証**: 100%成功 🎉

### テスト実行サマリー

1. **クリーニング決定性**
   - 20ファイルサブセットで2回実行 (同一seed)
   - ✅ Fileset Hash一致: `8efad288c36b`
   - ✅ 全メタデータSHA1一致
   - ✅ `diff -r` 完全一致

2. **品質ゲート検証**
   - 14ファイル検証 (Drums)
   - ✅ 5件合格 (35.7%)
   - ✅ 9件不合格 (`min_kick_on_beat_rate_violation`)
   - ✅ 非クリティカル正常処理

3. **層別分割決定性**
   - 同一seedで2回実行
   - ✅ Train 10 / Val 0 / Test 4 (両実行で一致)
   - ✅ 層別化: 5初期層 → 3最終層
   - ✅ `diff -r` 完全一致

### 実装完了機能

| 機能 | 実装 | 検証 |
|------|------|------|
| `stable_list_midis()` | ✅ | ✅ |
| `seeded_rng()` | ✅ | ✅ |
| `atomic_write_json()` | ✅ | ✅ |
| Schema versioning | ✅ | ✅ |
| Provenance tracking | ✅ | ✅ |
| `--dry-run` | ✅ | ✅ |
| `--jobs` | ✅ | ✅ |
| `--fail-on-critical` | ✅ | ✅ |
| `--summary` | ✅ | ✅ |
| Quality gates YAML | ✅ | ✅ |

**詳細**: `MIDI_CLEANING_VALIDATION.md` 参照  
**CI/CD**: `.github/workflows/midi_cleaning_ci.yml` 参照

---

## 📈 reason_codes一覧

### Critical (強制隔離)
- `hard_fail` - 致命的エラー
- `parse_error` - MIDIパース失敗
- `tempo_change_excess` - 異常な大量テンポ変更 (>100)
- `pitch_outlier` - GM範囲外ピッチ
- `pitch_range_excessive` - 音域が7オクターブ超

### Warning (3つ以上で隔離)
- `too_short` - 小節数<1
- `too_few_notes` - ノート数<8
- `pedal_excessive` - ペダル使用率>90%
- `arpeggio_glitch` - 過密アルペジオ
- `leap_excess` - 大跳躍連発
- `grid_off_outlier` - グリッド外れ>30%
- `staccato_flood` - スタッカート洪水
- `chord_spread_excess` - 和音広がり>24半音
- `drum_program_mismatch` - ドラムプログラム矛盾

---

## 🔄 CI/CD統合

```yaml
# .github/workflows/clean_midi.yml
name: MIDI Cleaning Pipeline

on: [push, pull_request]

jobs:
  clean:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install pretty_midi pyyaml
      
      - name: Run cleaning (smoke test)
        run: |
          python scripts/clean_midi.py \
            --in tests/fixtures/piano \
            --out tests/output/clean \
            --instrument piano \
            --quarantine tests/output/quarantine
      
      - name: Validate
        run: |
          python scripts/validate_and_gate.py \
            --in tests/output/clean \
            --gates configs/quality_gates/quality_gates.yaml \
            --report tests/output/validation_report.json
```

---

## 📚 関連ドキュメント

- `docs/LAMDA_README.md` - LAMDaデータセット統合ガイド
- `docs/LAMDA_STAGE2_SPEC.md` - Stage 2仕様 (ドラム特化)
- `scripts/clean_drumloops_lamda.py` - 既存のドラム専用クリーナー

---

## 🛠️ 開発者向け

### 新しい楽器クリーナーの追加

1. `scripts/cleaners/{instrument}.py` を作成
2. `clean_{instrument}` 関数を実装 (返り値: `(pm, metadata, reason_codes)`)
3. `scripts/cleaners/__init__.py` に追加
4. `scripts/clean_midi.py` の `REGISTRY` に登録
5. `configs/quality_gates/quality_gates.yaml` にゲート基準を追加

### テスト

```bash
# ユニットテスト
pytest tests/test_cleaners.py

# スモークテスト (10ファイル)
python scripts/clean_midi.py \
  --in tests/fixtures/piano \
  --out tests/output/clean \
  --instrument piano \
  --quarantine tests/output/quarantine
```

---

**✅ すべてのタスク完了！**

新しい統合クリーニングシステムが利用可能です。🎵
