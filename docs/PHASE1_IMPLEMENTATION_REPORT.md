# Phase 1 Implementation Report: Manifest Runner & Dataset Integration

**日付:** 2025年10月17日  
**ステータス:** 🚧 実装中  
**進捗:** 40% (基盤実装完了、テスト・統合待ち)

---

## 📋 実装完了項目

### 1. Manifest Runner (run_manifest.py) ✅

**ファイル:** `scripts/run_manifest.py`  
**サイズ:** ~400行  
**ステータス:** 基本実装完了（Generator統合は次フェーズ）

**実装内容:**

#### ShardWriter クラス
```python
class ShardWriter:
    - Resume検出: _detect_next_index()
    - バッファ管理: add(), flush()
    - Shard自動分割: 5,000件/shard
    - 統計出力: total_written
```

#### technique_to_params() マッピング
```python
# 楽器×奏法 → Generatorパラメータ変換
Guitar:
  - strum → {rhythm_key: "strum_basic", velocity: 64, ...}
  - arpeggio → {rhythm_key: "arpeggio_16th", velocity: 56, ...}
  - fingerpicking → {rhythm_key: "fingerpick_pattern", ...}
  - power_chord → {rhythm_key: "power_chord_8th", velocity: 80, ...}

Bass:
  - walking → {pattern: "walking_quarter", velocity: 72, ...}
  - pick → {pattern: "picked_8th", velocity: 76, ...}
  - slap → {pattern: "slap_funk", velocity: 90, ...}
  - fingerstyle → {pattern: "finger_groove", ...}

Strings:
  - legato → {style: "legato", note_overlap: 0.9, ...}
  - staccato → {style: "staccato", note_overlap: 0.2, ...}
  - spiccato → {style: "spiccato", bow_direction: "bouncing", ...}
  - sustained → {style: "sustained", note_overlap: 1.0, ...}
  - tremolo → {style: "tremolo", tremolo_rate: 8, ...}

Piano:
  - pop_comping → {style: "comping", chord_density: "medium", ...}
  - ballad → {style: "ballad", chord_density: "sparse", ...}
  - jazz_voicing → {style: "jazz", chord_density: "complex", ...}
  - arpeggio_pattern → {style: "arpeggio", ...}
  - fast_runs → {style: "runs", velocity: 72, ...}
```

#### Multi-instrument協調生成（スタブ実装）
```python
def sync_bass_drums(bass_gen, drums_gen, section):
    # TODO: ドラムキック × ベースルート音タイミング同期
    pass

def align_guitar_strings(guitar_gen, strings_gen, section):
    # TODO: ギターコード × ストリングスボイシング調和
    pass
```

#### コマンドライン引数
```bash
python scripts/run_manifest.py \
  --manifest manifests/manifest_20251017.jsonl \
  --pickle-out data/shards/hybrid \
  --shard-size 5000 \
  --emit-midi-out audits/synth_midi \  # Optional
  --resume \
  --max-jobs 100  # For testing
```

**未実装（次フェーズ）:**
- ⏸️ ModularComposer統合（実際のMIDI生成）
- ⏸️ EmotionHumanizer統合（表現制御）
- ⏸️ LAMDA metadata抽出（品質スコア）
- ⏸️ Multi-instrument協調生成の完全実装

---

### 2. Dataset Integration Guide ✅

**ファイル:** `docs/DATASET_INTEGRATION_GUIDE.md`  
**サイズ:** ~450行  
**ステータス:** 完成

**内容:**
- 📊 Critical Gaps優先度表（strings_legato 1,117件 → URMP推奨）
- 🎯 楽器別推奨データセット（GuitarSet/URMP/MAESTRO/SMD/Lakh）
- 🔧 統合ワークフロー（4 Phases）
- 📋 データセット評価基準（必須条件/推奨条件/除外基準）
- 🎯 統合目標（短期: 1,310件、中期: 2,160件）
- 📚 参考資料（論文リンク、ライセンス注意事項）

**主要データセット:**

| データセット | 楽器 | 推定ファイル数 | 優先度 | Target Gap |
|------------|------|--------------|-------|-----------|
| **GuitarSet** | guitar | 360 | 🔴 High | strum (1,554), arpeggio (1,007) |
| **URMP** | strings | 350 | 🔴 High | legato (1,117), spiccato (600) |
| **MAESTRO** | piano | 200 | 🟡 Medium | ballad (200) |
| **SMD** | bass | 400 | 🟡 Medium | pick (900), walking (352) |

**統合後の予想:**
- Real/External: 5,719件 (3,559 + 2,160)
- Remaining Synthetic: 5,728件 (7,888 - 2,160)
- **合成データ依存度削減:** 100% → 73%

---

### 3. Dataset Downloader ✅

**ファイル:** `scripts/download_external_datasets.sh`  
**サイズ:** ~350行  
**ステータス:** 完成（実行可能）

**対応データセット:**
```bash
# All priority datasets
bash scripts/download_external_datasets.sh all

# Individual datasets
bash scripts/download_external_datasets.sh guitarset
bash scripts/download_external_datasets.sh urmp
bash scripts/download_external_datasets.sh maestro
bash scripts/download_external_datasets.sh smd
bash scripts/download_external_datasets.sh lakh
```

**機能:**
- ✅ GuitarSet: Git clone + 手動DL案内（Zenodo）
- ✅ URMP: wget + tar抽出（~10GB）
- ✅ MAESTRO: wget + unzip（~200MB MIDI only）
- ✅ SMD: Git clone + subset DL案内
- ✅ Lakh: wget matched subset（~30GB）
- ✅ Resume対応（既存ファイルチェック）
- ✅ Summary表示（ダウンロード状況統計）

---

### 4. Dataset Import Scripts ✅

#### GuitarSet Importer
**ファイル:** `scripts/import_guitarset.py`  
**機能:**
- JAMS annotation → MIDI変換
- 奏法分類（strum/arpeggio/fingerpicking/mixed）
- 出力: `data/external/guitarset/raw/*.mid`

```bash
# Dry run (確認のみ)
python scripts/import_guitarset.py --dry-run

# 実行
python scripts/import_guitarset.py \
  --guitarset-dir data/external/guitarset \
  --output-dir data/external/guitarset/raw
```

#### URMP Importer
**ファイル:** `scripts/import_urmp.py`  
**機能:**
- 弦楽器MIDI抽出（violin/viola/cello/double bass）
- 奏法推定（legato/staccato/spiccato/mixed）
  - Heuristics: note overlap, duration ratio, velocity variation
- 出力: `data/external/urmp/raw/strings/*.mid`

```bash
# Dry run
python scripts/import_urmp.py --dry-run

# 実行
python scripts/import_urmp.py \
  --urmp-dir data/external/urmp \
  --output-dir data/external/urmp/raw/strings
```

---

## ⏸️ 未実装項目

### Priority 1: Generator統合（run_manifest.py完成）

**タスク:**
1. ModularComposer統合
   - `composer.compose(section_data=section)` 実装
   - technique_params → Music21スコア生成
2. EmotionHumanizer統合
   - emotion → velocity/timing variation適用
3. LAMDA metadata抽出
   - 生成MIDI → lamda_integration.extract_lamda_metadata()
   - 品質スコア計算
4. Multi-instrument協調生成
   - sync_bass_drums() 完全実装
   - align_guitar_strings() 完全実装

**推定工数:** 3-4日

---

### Priority 2: 外部データセット実行（短期目標）

**タスク:**
1. GuitarSet ダウンロード & インポート
   ```bash
   bash scripts/download_external_datasets.sh guitarset
   python scripts/import_guitarset.py
   ```
2. URMP ダウンロード & インポート
   ```bash
   bash scripts/download_external_datasets.sh urmp
   python scripts/import_urmp.py
   ```
3. Stage1統合
   - `scripts/run_stage1_clean_multi.sh` のDATASETSテーブル更新
   ```bash
   GUITARSET guitar data/external/guitarset/raw output/guitarset/clean/guitar ...
   URMP strings data/external/urmp/raw/strings output/urmp/clean/strings ...
   ```
4. Stage2実行 & Gap再評価
   ```bash
   bash scripts/run_stage1_clean_multi.sh
   bash scripts/run_stage2_multi.sh
   python scripts/generate_distribution_counts.py --include-external
   ```

**推定工数:** 2-3日（ダウンロード時間含む）

---

### Priority 3: Manifest Runner小規模テスト

**タスク:**
1. テストManifest作成（100件）
   ```bash
   head -10 manifests/manifest_20251017.jsonl > manifests/test_manifest.jsonl
   # 各行のcountを10に変更 → 合計100件
   ```
2. 実行
   ```bash
   python scripts/run_manifest.py \
     --manifest manifests/test_manifest.jsonl \
     --pickle-out data/shards/test \
     --max-jobs 10
   ```
3. 出力検証
   - Shard pickle生成確認
   - メタデータ構造確認
   - Resume機能テスト

**推定工数:** 1日

---

### Priority 4: MAESTRO/SMD統合（中期目標）

**タスク:**
1. MAESTRO ダウンロード & Stage1統合
2. SMD Bass subset ダウンロード & Stage1統合
3. 完全統合後のGap再評価
4. Manifest更新（manifest_after_external.jsonl）

**推定工数:** 3-4日

---

## 🎯 次のステップ（優先度順）

### Week 1（今週）
1. ✅ run_manifest.py基本実装（完了）
2. ✅ Dataset Integration Guide作成（完了）
3. ✅ download_external_datasets.sh作成（完了）
4. ✅ import_guitarset.py/import_urmp.py作成（完了）
5. ⏸️ **GuitarSet/URMPダウンロード実行**
6. ⏸️ **外部データセットインポート実行**

### Week 2（来週）
7. ⏸️ Stage1/Stage2統合（GuitarSet/URMP）
8. ⏸️ Gap再評価（統合後）
9. ⏸️ run_manifest.py Generator統合開始

### Week 3（再来週）
10. ⏸️ Manifest Runner完全実装
11. ⏸️ 小規模テスト（100件生成）
12. ⏸️ MAESTRO/SMD統合

---

## 📊 進捗サマリー

**完了 (40%):**
- ✅ Manifest Runner基本実装（ShardWriter, technique_to_params）
- ✅ Dataset Integration戦略策定
- ✅ Download/Import scripts作成
- ✅ ドキュメント整備

**進行中 (30%):**
- 🚧 外部データセット実行準備
- 🚧 Generator統合設計

**未着手 (30%):**
- ⏸️ Generator完全統合
- ⏸️ Multi-instrument協調生成
- ⏸️ 小規模テスト & 検証
- ⏸️ 中期目標データセット統合

---

## 🔧 技術的課題

### 課題1: Generator API整合性
**問題:** `modular_composer.py`のAPI仕様未確認  
**対策:** 既存コード調査 + APIドキュメント確認

### 課題2: LAMDA metadata抽出精度
**問題:** 合成データのスコアリング精度不明  
**対策:** 小規模テストでReal Dataとのスコア比較

### 課題3: 外部データセット品質バラつき
**問題:** GuitarSet/URMP以外はアノテーション不足  
**対策:** Stage2フィルタリング閾値調整（Real+5%厳格適用）

### 課題4: Multi-instrument協調生成の複雑性
**問題:** タイミング/ハーモニー同期アルゴリズム未定  
**対策:** 単純版実装 → 段階的改善

---

## 💡 知見・改善提案

### 知見1: Suno依存からの脱却効果
- **戦略転換:** 100%合成 → 73%合成（外部データセット統合後）
- **メリット:** 実装コスト削減、品質向上、多様性確保
- **特に効果大:** strings_legato (URMP), guitar_strum (GuitarSet)

### 知見2: Manifest-driven設計の有効性
- **冪等性:** Resume対応で中断からの再開が安全
- **可視化:** 7,888件の不足が具体的なJSONL jobに変換
- **拡張性:** 新規奏法追加が容易（targets_hybrid.yaml更新のみ）

### 改善提案1: Dataset評価の自動化
- 現状: 手動でDataset品質判断
- 提案: Stage1/Stage2パイプライン統合前に品質プレビュー機能
  ```bash
  python scripts/preview_dataset.py \
    --input data/external/guitarset/raw \
    --sample-size 50
  ```

### 改善提案2: Technique分類の機械学習化
- 現状: Heuristics-based（import_urmp.pyなど）
- 提案: Stage2メトリクスを特徴量としたML分類器
  - Training: 既存3,559件のStage2結果
  - 精度向上: Heuristicsの60-70% → MLで80-90%

---

**最終更新:** 2025年10月17日  
**次回更新予定:** 外部データセット統合完了後
