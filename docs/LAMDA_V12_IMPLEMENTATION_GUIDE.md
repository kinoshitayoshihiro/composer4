# LAMDA統合 v1.2 完全実装ガイド

**更新日**: 2025年10月23日  
**バージョン**: v1.2（META正規化＋SIGNATURES自動マップ＋監査ツール統合）

---

## ✅ 新規実装コンポーネント（v1.2）

### 🔧 Phase 1: META正規化＋SIGNATURES自動マップ

| コンポーネント | ファイル | 機能 |
|------------|---------|------|
| METAキー正規化 | `adapters/meta_key_normalizer.py` | ✅実装済み |
| SIGNATURES自動マップ生成 | `scripts/build_signature_id_map.py` | ✅実装済み |
| Stage2拡張スクリプトv1.2 | `scripts/augment_stage2_with_lamda_meta.py` | ✅実装済み |
| 拍子IDマップ初期版 | `adapters/signature_id_map.yaml` | ✅実装済み |

**新機能**:
1. **META_DATAのキー表記ゆれ吸収**
   - snake_case / camelCase / synonym → 共通スキーマ
   - `normalize_meta()` で一貫したJSON構造

2. **SIGNATURES_DATA → 拍子マップ自動生成**
   - Stage2の`timesig_map`とLAMDAの`SIGNATURES_DATA`を突き合わせ
   - 多数決でID→"num/den"対応を自動構築
   - `adapters/signature_id_map.yaml`を自動更新

3. **Stage2拡張（v1.2）**
   - 正規化META統合
   - 自動生成拍子マップ適用
   - TOTALS先験統合
   - `schema_version: "lamda_v2.2"`

---

### 📊 Phase 2: 監査ツール統合

| ツール | ファイル | 機能 |
|--------|---------|------|
| KILO vs CHORDS監査 | `scripts/audit_kilo_vs_chords.py` | ✅実装済み |
| EXT vs GOLD監査 | `scripts/audit_vs_gold.py` | ✅実装済み |
| EXT vs GOLD監査（マップ対応） | `scripts/audit_vs_gold_map.py` | ✅NEW! |
| リングバッファ追記 | `scripts/ringbuffer_append.py` | ✅実装済み |
| リングバッファレポート | `scripts/ringbuffer_report.py` | ✅実装済み |
| リングバッファレポート（PNG） | `scripts/ringbuffer_report_png.py` | ✅NEW! |

**新機能**:
- **ファイル名不一致対応**（mapping CSV）
- **時系列PNG可視化**（matplotlib）
- **タグ別フィルタリング**

---

### 🎨 Phase 3: 可視化ツール

| ツール | ファイル | 機能 |
|--------|---------|------|
| 小節別差分ヒートマップ | `scripts/visualize_chord_diff.py` | ✅NEW! |
| Top-N差分サムネ一括生成 | `scripts/build_topn_thumbs.py` | ✅NEW! |
| ファイルID対応自動推定 | `scripts/auto_file_id_map.py` | ✅NEW! |

**新機能**:
1. **小節別差分ヒートマップ**
   - 1行×N列（0=一致, 1=不一致）
   - 視覚的な差分確認

2. **Top-N自動抽出＋サムネ生成**
   - 監査CSVから不一致Top-N抽出
   - 一括PNG生成（要注目ファイルの可視化）

3. **ファイルID自動マッピング**
   - ハッシュ完全一致
   - 小節数＋ハミング距離近似
   - mapping CSV自動生成

---

## 🚀 使い方（最短レシピ）

### 1. SIGNATURES自動マップ生成

**前提**: Stage2 JSONに`timesig_map: [(0,"4/4")]`が一定数入っている

```bash
python scripts/build_signature_id_map.py \
  --signatures-pickle data/Los-Angeles-MIDI/SIGNATURES_DATA/LAMDa_SIGNATURES_DATA.pickle \
  --stage2-json-dir output/stage2/test/json \
  --out-yaml adapters/signature_id_map.yaml
```

**成果**: `adapters/signature_id_map.yaml`が自動更新（ID→"4/4"等）

---

### 2. META正規化＋Stage2拡張（v1.2）

```bash
python scripts/augment_stage2_with_lamda_meta.py \
  --stage2-json output/stage2/test/json/162000.stage2.json \
  --file-id 162000 \
  --kilo-pickle data/Los-Angeles-MIDI/KILO_CHORDS_DATA/LAMDa_KILO_CHORDS_DATA.pickle \
  --meta-pickle "data/Los-Angeles-MIDI/META_DATA/LAMDa_META_DATA_*.pickle" \
  --signatures-pickle data/Los-Angeles-MIDI/SIGNATURES_DATA/LAMDa_SIGNATURES_DATA.pickle \
  --signature-map adapters/signature_id_map.yaml \
  --totals-pickle data/Los-Angeles-MIDI/TOTALS_MATRIX/LAMDa_TOTALS.pickle \
  --write-back 1
```

**追加フィールド**:
- `chordmap`: KILO由来（指定時）
- `lamda_meta`: 正規化済みメタ
- `timesig_map`: ID→拍子変換
- `global_priors`: TOTALS先験
- `schema_version: "lamda_v2.2"`

**NO-OP安全**: データ源が無くてもスキップ

---

### 3. 監査ツール（4種）

#### 3-1. KILO vs CHORDS一貫性
```bash
python scripts/audit_kilo_vs_chords.py \
  --kilo data/Los-Angeles-MIDI/KILO_CHORDS_DATA/LAMDa_KILO_CHORDS_DATA.pickle \
  --chords-dir data/Los-Angeles-MIDI/CHORDS_DATA \
  --out-csv analysis/kilo_vs_chords_audit.csv \
  --tpq 480 \
  --max-files 0
```

**CSV列**: `file_id, bars, match_rate, n_diff, diff_bars, kilo_first5, chords_first5`

#### 3-2. EXT vs GOLD（基本版）
```bash
python scripts/audit_vs_gold.py \
  --gold-dir data/GOLD_stage2_json \
  --ext-dir data/lamda_chordmaps \
  --out-csv analysis/ext_vs_gold.csv
```

#### 3-3. EXT vs GOLD（マップ対応版）
```bash
# まずファイルID対応を自動推定
python scripts/auto_file_id_map.py \
  --gold-dir data/GOLD_stage2_json \
  --ext-dir data/lamda_chordmaps \
  --out-csv mappings/file_map.csv

# マップを使って監査
python scripts/audit_vs_gold_map.py \
  --gold-dir data/GOLD_stage2_json \
  --ext-dir data/lamda_chordmaps \
  --out-csv analysis/ext_vs_gold_mapped.csv \
  --mapping-csv mappings/file_map.csv
```

**CSV列**: `file_id, bars, match_rate, n_diff, diff_bars`

#### 3-4. リング追記＋レポート
```bash
# スナップショット追記
python scripts/ringbuffer_append.py \
  --csv analysis/ext_vs_gold.csv \
  --ring analysis/consistency_ring.jsonl \
  --tag ext_vs_gold \
  --max-entries 200

# CSV＋PNG時系列レポート
python scripts/ringbuffer_report_png.py \
  --ring analysis/consistency_ring.jsonl \
  --out-csv analysis/consistency_ring.csv \
  --out-png analysis/consistency_ring.png \
  --tag ext_vs_gold
```

---

### 4. 可視化ツール（3種）

#### 4-1. 単一ファイル差分ヒートマップ
```bash
python scripts/visualize_chord_diff.py \
  --gold-json data/GOLD_stage2_json/162000.stage2.json \
  --ext-json data/lamda_chordmaps/162000.json \
  --out-png analysis/162000_diff.png
```

**出力**: 1行×N列ヒートマップ（0=一致, 1=不一致）

#### 4-2. Top-N差分サムネ一括生成
```bash
python scripts/build_topn_thumbs.py \
  --audit-csv analysis/ext_vs_gold.csv \
  --gold-dir data/GOLD_stage2_json \
  --ext-dir data/lamda_chordmaps \
  --out-dir analysis/thumbs \
  --top-n 50 \
  --mapping-csv mappings/file_map.csv
```

**成果**: `analysis/thumbs/*.png`（不一致Top-50の可視化）

#### 4-3. ファイルID自動マッピング
```bash
python scripts/auto_file_id_map.py \
  --gold-dir data/GOLD_stage2_json \
  --ext-dir data/lamda_chordmaps \
  --out-csv mappings/file_map.csv
```

**手法**:
1. ハッシュ完全一致（MD5）
2. 小節数差≤2 ＋ ハミング距離最小

---

## 📊 運用パイプライン（推奨）

### パターンA: 一貫性監視（CI/定期実行）

```bash
#!/bin/bash
# 1. KILO vs CHORDS
python scripts/audit_kilo_vs_chords.py \
  --kilo data/KILO.pickle \
  --chords-dir data/CHORDS_DATA \
  --out-csv analysis/kilo_vs_chords.csv

# 2. EXT vs GOLD（マップ対応）
python scripts/auto_file_id_map.py \
  --gold-dir data/GOLD \
  --ext-dir data/lamda_chordmaps \
  --out-csv mappings/auto_map.csv

python scripts/audit_vs_gold_map.py \
  --gold-dir data/GOLD \
  --ext-dir data/lamda_chordmaps \
  --mapping-csv mappings/auto_map.csv \
  --out-csv analysis/ext_vs_gold.csv

# 3. リング更新
python scripts/ringbuffer_append.py \
  --csv analysis/ext_vs_gold.csv \
  --ring analysis/ring.jsonl \
  --tag ext_vs_gold \
  --max-entries 200

# 4. レポート生成
python scripts/ringbuffer_report_png.py \
  --ring analysis/ring.jsonl \
  --out-csv analysis/ring.csv \
  --out-png analysis/ring.png

# 5. Top-N可視化（任意）
python scripts/build_topn_thumbs.py \
  --audit-csv analysis/ext_vs_gold.csv \
  --gold-dir data/GOLD \
  --ext-dir data/lamda_chordmaps \
  --out-dir analysis/thumbs \
  --top-n 20
```

### パターンB: GOLD生成→SILVER拡張

```bash
#!/bin/bash
# 1. SIGNATURES自動マップ生成（1回のみ）
python scripts/build_signature_id_map.py \
  --signatures-pickle data/SIGNATURES_DATA.pickle \
  --stage2-json-dir output/stage2/GOLD/json \
  --out-yaml adapters/signature_id_map.yaml

# 2. SILVER候補にMETA拡張（一括）
for json in output/stage2/SILVER_candidates/json/*.stage2.json; do
  fid=$(basename "$json" .stage2.json)
  python scripts/augment_stage2_with_lamda_meta.py \
    --stage2-json "$json" \
    --file-id "$fid" \
    --kilo-pickle data/KILO.pickle \
    --meta-pickle "data/META_DATA_*.pickle" \
    --signatures-pickle data/SIGNATURES_DATA.pickle \
    --signature-map adapters/signature_id_map.yaml \
    --totals-pickle data/TOTALS_MATRIX.pickle \
    --write-back 1
done

# 3. Teacher v1推論
python scripts/teacher_v1_infer.py \
  --model models/teacher_v1.pkl \
  --in-dir output/stage2/SILVER_candidates/json \
  --out-dir analysis/teacher_v1_pred

# 4. SILVER登録（閾値ゲート）
python scripts/tier_register.py \
  --pred-dir analysis/teacher_v1_pred \
  --tier SILVER \
  --threshold 0.75
```

---

## 🎯 品質ゲート

### A/B監査基準
| 指標 | 閾値 | 判定 |
|------|------|------|
| `match_rate >= 0.95` | GOLD | 手作業確認済み |
| `match_rate >= 0.85` | SILVER | Teacher v1 + 監査合格 |
| `0.70 <= match_rate < 0.85` | BRONZE | 要手動確認 |
| `match_rate < 0.70` | 破棄 | 品質不足 |

### リング監視（CI）
```bash
# mean_matchが閾値以下ならアラート
THRESHOLD=0.78
LATEST=$(tail -1 analysis/ring.jsonl | jq -r '.mean_match')
if (( $(echo "$LATEST < $THRESHOLD" | bc -l) )); then
  echo "⚠️  ALERT: consistency dropped to $LATEST"
  exit 1
fi
```

---

## 📈 実装完了率

| Phase | コンポーネント | ステータス |
|-------|--------------|----------|
| **Phase 1** | META正規化 | ✅ 100% |
| | SIGNATURES自動マップ | ✅ 100% |
| | Stage2拡張v1.2 | ✅ 100% |
| **Phase 2** | KILO vs CHORDS監査 | ✅ 100% |
| | EXT vs GOLD監査 | ✅ 100% |
| | マップ対応監査 | ✅ 100% |
| | リングバッファ | ✅ 100% |
| **Phase 3** | 差分ヒートマップ | ✅ 100% |
| | Top-Nサムネ | ✅ 100% |
| | 自動マッピング | ✅ 100% |

**総合**: **100%完成** 🎉

---

## 🔧 依存関係

### 必須
```bash
pip install pyyaml numpy matplotlib
```

### オプショナル
- `music21` (chordmap検証、既存実装)
- `pretty_midi` (MIDI解析、既存実装)

---

## 📚 関連ドキュメント

- `docs/LAMDA_IMPLEMENTATION_STATUS.md` - Phase 1-5実装状況
- `docs/LAMDA_INTEGRATION_PLAN.md` - 全体計画
- `STAGE2_OUTPUT_FORMATS.md` - Stage2出力仕様
- `schemas/tiered_data_schema.py` - GOLD/SILVER/BRONZE管理

---

## 🎯 次のステップ

### 即実行可能
1. ✅ ~~SIGNATURES自動マップ生成~~
2. ✅ ~~META正規化＋Stage2拡張~~
3. ✅ ~~監査ツール統合~~
4. ✅ ~~可視化ツール実装~~
5. ⏳ **GOLD 5-10k生成**（カバレッジ格子）

### 拡張候補（v1.3+）
1. **外れ値スコア実計算**
   - Stage2のMIDIパス→ピッチ/音価/ベロシティヒスト
   - `outlier_scores`フィールド追加

2. **SIGNATURES時間軸化**
   - `timesig_map`を区間配列化（転拍子対応）

3. **META→controls強化**
   - PB/CC分布詳細を`controls`に統合

4. **ティア自動ゲート**
   - `lamda_meta` + `global_priors`スコア→SILVER/BRONZE自動フラグ

---

## ✨ まとめ

**v1.2実装完了**: 全コンポーネント実装済み、即使用可能！

**主要機能**:
- ✅ META_DATA正規化（キー表記ゆれ吸収）
- ✅ SIGNATURES自動マップ（多数決ベース）
- ✅ Stage2拡張v1.2（正規化META＋拍子＋先験）
- ✅ 監査ツール4種（KILO/CHORDS/EXT/GOLD）
- ✅ 可視化ツール3種（ヒートマップ/サムネ/自動マップ）
- ✅ リングバッファ（時系列監視＋PNG）

**運用パイプライン**: CI/定期実行レディ 🚀
