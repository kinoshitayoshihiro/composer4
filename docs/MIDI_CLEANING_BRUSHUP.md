# 統合MIDIクリーニングパイプライン - ブラッシュアップ版

**決定論・原子的IO・並列処理・CI親和性を強化**

---

## 🎯 重点改善ポイント

### 1️⃣ **決定論の強化**
- ✅ ファイル列挙を `stable_list_midis()` で FS順非依存に
- ✅ 乱数を `seeded_rng(seed)` で SHA1ベース決定論RNGに統一
- ✅ `fileset_hash` で入力集合の変更を検出

### 2️⃣ **原子的I/O**
- ✅ `.meta.json` を `atomic_write_json()` で中断耐性
- ✅ tempfile経由の `os.replace()` で書き込み途中の壊れを防止

### 3️⃣ **再入可能性 (Idempotent)**
- ✅ `--force` なしでは既存 `.meta.json` をスキップ
- ✅ `meta_index.jsonl` で処理履歴を追記
- ✅ 同一入力→同一出力を保証

### 4️⃣ **並列処理**
- ✅ `--jobs N` で ProcessPoolExecutor による並列実行
- ✅ `--dry-run` で対象件数のみ表示 (クイック確認)

### 5️⃣ **CI親和性**
- ✅ `--fail-on-critical` でクリティカル検出時に exit code 2
- ✅ `--summary reports/summary.jsonl` で1行1件の軽量集計
- ✅ `schema_version`, `provenance`, `fileset_hash` で監査性

---

## 🚀 使用方法 (改善版)

### Step 1: クリーニング (ドライラン→並列実行)

```bash
# 1) ドライラン: 件数確認
python scripts/clean_midi.py \
  --in data/lamda/raw/piano \
  --out data/lamda/clean/piano \
  --instrument piano \
  --quarantine data/lamda/quarantine/piano \
  --dry-run

# 2) 並列実行 (8コア)
python scripts/clean_midi.py \
  --in data/lamda/raw/piano \
  --out data/lamda/clean/piano \
  --instrument piano \
  --quarantine data/lamda/quarantine/piano \
  --jobs 8 \
  --seed cleaning-v1

# 3) 再実行 (既存スキップ)
# --force なしで実行すると、既に .meta.json があるファイルはスキップ
python scripts/clean_midi.py \
  --in data/lamda/raw/piano \
  --out data/lamda/clean/piano \
  --instrument piano \
  --quarantine data/lamda/quarantine/piano \
  --jobs 8
```

**出力:**
- `data/lamda/clean/piano/*.mid` - クリーニング済みMIDI
- `data/lamda/clean/piano/*.meta.json` - メタデータ (原子的保存)
- `data/lamda/clean/piano/meta_index.jsonl` - 処理インデックス
- `data/lamda/quarantine/piano/` - 隔離ファイル (階層維持)
- `data/lamda/piano_clean_report.json` - 統計レポート

### Step 2: 検証 & Quality Gates (クリティカルで失敗)

```bash
python scripts/validate_and_gate.py \
  --in data/lamda/clean/piano \
  --gates configs/quality_gates/quality_gates.yaml \
  --report reports/piano_validation.json \
  --summary reports/piano_summary.jsonl \
  --fail-on-critical
```

**動作:**
- クリティカル検出時は `exit code 2` で終了 (CI連携)
- `--summary` でJSONL追記 (軽量・可視化しやすい)

**出力:**
- `reports/piano_validation.json` - 詳細レポート (原子的保存)
- `reports/piano_summary.jsonl` - 1行1件の要約

### Step 3: 層別分割 (決定論・極小層吸収)

```bash
python scripts/prepare_splits.py \
  --in data/lamda/clean/piano \
  --out data/lamda/splits/piano \
  --seed 1234 \
  --min-bucket 3
```

**動作:**
- `--min-bucket 3`: 3件未満の層は `tempo:mid` に統合
- SHA1決定論でseed固定なら常に同一分割

**出力:**
- `data/lamda/splits/piano/train/` (80%)
- `data/lamda/splits/piano/val/` (10%)
- `data/lamda/splits/piano/test/` (10%)
- `data/lamda/splits/piano/split_summary.json` - 統計

---

## 📊 出力ファイル仕様

### `.meta.json` (共通フィールド)

```json
{
  "schema_version": "1.0",
  "fileset_hash": "a1b2c3d4e5f6",
  "provenance": {
    "tool": "cleaning-pipeline",
    "schema_version": "1.0",
    "git_commit": "abc123...",
    "git_branch": "main"
  },
  "clean_actions": ["remove_invalid_notes:5"],
  "reason_codes": ["pedal_excessive"],
  "tempo": 120.0,
  "bars": 16.0,
  "notes": 256,
  "density": 8.5,
  
  // 楽器別メトリクス
  "pedal_sustain_ratio": 0.65,
  "hand_separation": 18.5,
  "velocity_std": 22.3
}
```

### `meta_index.jsonl` (クリーニング履歴)

```jsonl
{"path": "data/raw/piano/file1.mid", "fileset_hash": "a1b2c3", "reason_codes": [], "tempo": 120, "bars": 16, "notes": 256, "density": 8.5}
{"path": "data/raw/piano/file2.mid", "fileset_hash": "a1b2c3", "reason_codes": ["pedal_excessive"], "tempo": 90, "bars": 8, "notes": 128, "density": 4.2}
```

### `summary.jsonl` (検証サマリ)

```jsonl
{"path": "clean/piano/file1.meta.json", "passed": true, "is_critical": false, "reasons": [], "violations": []}
{"path": "clean/piano/file2.meta.json", "passed": false, "is_critical": false, "reasons": ["pedal_excessive"], "violations": ["max_pedal_sustain_ratio_violation"]}
```

---

## 🔧 Quality Gates (改善版)

### `configs/quality_gates/quality_gates.yaml`

```yaml
# Severity: critical (強制隔離) | warning (3つ以上で隔離)

piano:
  min_notes: 20
  max_pedal_sustain_ratio: 0.85  # 0.95 → 0.85 に引き下げ
  min_hand_separation: 6

guitar:
  min_strum_consistency: 0.75  # 新規追加

bass:
  min_root_or_chord_tone_rate: 0.70  # ルート/コードトーン率

strings:
  min_legato_connection_rate: 0.60  # レガート接続率

drums:
  min_kick_on_beat_rate: 0.55  # 0.3 → 0.55 に引き上げ

# Reason Codes (snake_case統一)
critical_reason_codes:
  - parse_error
  - note_count_low
  - strum_inconsistent
  - nonharmonic_excess

warning_reason_codes:
  - pedal_excessive
  - velocity_variation_low
  - kick_offbeat
  - legato_insufficient
```

---

## 🧪 受け入れ基準 (QA)

### ✅ 決定論

```bash
# 2回実行して byte-level 一致
python scripts/clean_midi.py --in raw --out clean1 --instrument piano --quarantine q1 --seed 42
python scripts/clean_midi.py --in raw --out clean2 --instrument piano --quarantine q2 --seed 42

diff -r clean1 clean2
# → 出力なし (完全一致)
```

### ✅ フェイル制御

```bash
python scripts/validate_and_gate.py \
  --in clean --gates gates.yaml --report r.json \
  --fail-on-critical

echo $?
# → 2 (クリティカル検出時)
# → 0 (正常時)
```

### ✅ 層別分割の決定論

```bash
python scripts/prepare_splits.py --in clean --out splits1 --seed 1234
python scripts/prepare_splits.py --in clean --out splits2 --seed 1234

diff <(ls splits1/train) <(ls splits2/train)
# → 出力なし (ファイルリスト一致)
```

### ✅ Quarantine階層維持

```bash
# 入力: data/raw/subdir/file.mid
# 出力: data/quarantine/subdir/file.mid (階層維持)
```

---

## 🔄 CI/CD統合例

### GitHub Actions

```yaml
name: MIDI Cleaning Pipeline

on: [push, pull_request]

env:
  GIT_COMMIT: ${{ github.sha }}
  GIT_BRANCH: ${{ github.ref_name }}

jobs:
  clean-and-validate:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install pretty_midi pyyaml tqdm
      
      - name: Clean MIDI (parallel)
        run: |
          python scripts/clean_midi.py \
            --in tests/fixtures/piano \
            --out tests/output/clean \
            --instrument piano \
            --quarantine tests/output/quarantine \
            --jobs 4 \
            --seed ${{ github.sha }}
      
      - name: Validate & Gate (fail on critical)
        run: |
          python scripts/validate_and_gate.py \
            --in tests/output/clean \
            --gates configs/quality_gates/quality_gates.yaml \
            --report tests/output/validation.json \
            --summary tests/output/summary.jsonl \
            --fail-on-critical
      
      - name: Upload Artifacts
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: cleaning-reports
          path: |
            tests/output/*.json
            tests/output/*.jsonl
```

---

## 📈 新規フラグ一覧

### `clean_midi.py`

| フラグ | デフォルト | 説明 |
|--------|-----------|------|
| `--dry-run` | false | 件数のみ表示 (実行なし) |
| `--jobs` | 1 | 並列処理数 (1=直列) |
| `--force` | false | 既存 .meta.json を上書き |
| `--seed` | "cleaning-default" | 乱数シード |

### `validate_and_gate.py`

| フラグ | デフォルト | 説明 |
|--------|-----------|------|
| `--fail-on-critical` | false | クリティカル検出時に exit 2 |
| `--summary` | None | JSONL要約出力パス |

### `prepare_splits.py`

| フラグ | デフォルト | 説明 |
|--------|-----------|------|
| `--min-bucket` | 3 | 最小層サイズ (未満は統合) |
| `--seed` | "splits-default" | 分割シード |

---

## 🛠️ トラブルシューティング

### Q: 並列実行でエラーが出る

```bash
# ProcessPoolExecutor は pickle可能な関数が必要
# → process_one_file() をモジュールトップレベルに定義済み
```

### Q: 既存ファイルが再処理される

```bash
# --force を外して再実行
python scripts/clean_midi.py --in raw --out clean --instrument piano --quarantine q
# (--force なし)
```

### Q: クリティカルでCIが止まらない

```bash
# --fail-on-critical を追加
python scripts/validate_and_gate.py ... --fail-on-critical
```

---

## 📚 関連ドキュメント

- `docs/MIDI_CLEANING_PIPELINE.md` - 基本設計
- `configs/quality_gates/quality_gates.yaml` - ゲート基準
- `scripts/cleaners/common.py` - 共通ユーティリティ

---

**✅ すべてのブラッシュアップ完了！決定論・CI親和性が大幅に向上しました** 🎵✨
