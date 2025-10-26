# Stage2 出力フォーマット仕様（v2.1）

## 設計方針

* **JSON/PKL が本線**：各曲の拡張メタデータを `*.stage2.json` と `*.stage2.pkl` で出力（既定で有効）
* **CSV は監査用オプショナル**：集計・可視化・デバッグ用に `--emit-csv aggregate` で4つのCSVを生成（既定で無効）

---

## 出力ディレクトリ構造

```
output/stage2/<dataset>/
├── json/                    # 既定: 各曲の拡張メタ（人間可読）
│   ├── <loop_id>.stage2.json
│   └── ...
├── pkl/                     # 既定: 各曲の拡張メタ（高速ロード）
│   ├── <loop_id>.stage2.pkl
│   └── ...
├── csv/                     # --emit-csv aggregate の時のみ
│   ├── stage2_index.csv           # 1曲=1行の概要
│   ├── bar_features.csv           # bar単位の長表（未実装）
│   ├── controls_summary.csv       # PB/CC/RPN集計（未実装）
│   └── roles_map.csv              # GM/音域ロール推定（未実装）
├── canonical_events.parquet # イベント系列（既存）
├── metrics_score.jsonl      # スコア詳細（既存）
└── BATCH_MANIFEST.json      # 処理済みファイルリスト（既存）
```

---

## CLI使用例

### 1. 既定動作（JSON + PKL のみ）

```bash
python scripts/lamda_stage2_extractor.py \
  --metadata-index output/stage1/<dataset>/cleaned.pickle \
  --config configs/lamda/<instrument>_stage2.yaml \
  --output-dir output/stage2/<dataset>
```

**出力**:
- `output/stage2/<dataset>/json/*.stage2.json`
- `output/stage2/<dataset>/pkl/*.stage2.pkl`
- `output/stage2/<dataset>/canonical_events.parquet`
- `output/stage2/<dataset>/metrics_score.jsonl`

### 2. CSV集計も出力

```bash
python scripts/lamda_stage2_extractor.py \
  --metadata-index output/stage1/<dataset>/cleaned.pickle \
  --config configs/lamda/<instrument>_stage2.yaml \
  --output-dir output/stage2/<dataset> \
  --emit-csv aggregate
```

**追加出力**:
- `output/stage2/<dataset>/csv/stage2_index.csv`（1曲=1行）

### 3. JSON/PKLを無効化（レガシーモード）

```bash
python scripts/lamda_stage2_extractor.py \
  --metadata-index output/stage1/<dataset>/cleaned.pickle \
  --config configs/lamda/<instrument>_stage2.yaml \
  --output-dir output/stage2/<dataset> \
  --emit-json 0 \
  --emit-pickle 0
```

**出力**: CSV（`loop_summary.csv`）のみ（後方互換性）

---

## JSON出力スキーマ（`*.stage2.json`）

### 基本構造

```json
{
  "loop_id": "Track00192_S06",
  "file": "Track00192_S06.mid",
  "score": 85.3,
  "threshold": 70.0,
  "extended.schema_version": "lamda_v2.1",
  "extended.tempo_map": [[0, 120.0]],
  "extended.timesig_map": [[0, "4/4"]],
  "extended.downbeats_ql": [0, 1920, 3840, ...],
  "extended.chordmap": {
    "unit": "ql",
    "events": [
      {"time": 0.0, "root": "C", "quality": "maj", "confidence": 0.62}
    ]
  },
  "extended.key_hints": [{"bar": 0, "key": "C", "mode": "major"}],
  "extended.modulations": [{"bar": 8, "from_key": "C", "to_key": "G"}],
  "extended.sections_auto": {
    "unit": "bar",
    "sections": [
      {"bar": 0, "label": "intro"},
      {"bar": 8, "label": "verse"}
    ],
    "energy": [[0, 0.2], [1, 0.5], ...]
  },
  "extended.groove": {
    "swing_pct": 0.0,
    "backbeat_strength": 0.5,
    "onset_deviation_hist": []
  },
  "extended.controls": {
    "pb_range": [-8191, 8191],
    "cc_used": [{"cc": 1, "range": [0, 127]}],
    "has_rpn": false
  },
  "extended.roles": {
    "instruments": [
      {"name": "Piano", "role": "melody", "program": 0}
    ]
  }
}
```

### 拡張メタフィールド詳細

| フィールド | 型 | 説明 |
|----------|---|------|
| `extended.schema_version` | string | スキーマバージョン（`"lamda_v2.1"`） |
| `extended.tempo_map` | array | `[[time_ql, bpm], ...]` テンポ変化 |
| `extended.timesig_map` | array | `[[bar_idx, "4/4"], ...]` 拍子変化 |
| `extended.downbeats_ql` | array | `[0, 1920, 3840, ...]` downbeats（QL単位） |
| `extended.chordmap` | object | コード進行（1小節単位、PC-setベース簡易版） |
| `extended.key_hints` | array | ローカルキー推定（bar単位） |
| `extended.modulations` | array | 転調検出 |
| `extended.sections_auto` | object | セクション自動分割（energy-based） |
| `extended.groove` | object | グルーヴ特徴（スケルトン、未実装） |
| `extended.controls` | object | PB/CC/RPN要約 |
| `extended.roles` | object | GM Program + 音域ベースのロール推定 |

---

## PKL出力（`*.stage2.pkl`）

- **内容**: JSONと同じdict構造をpickle化
- **利点**: 高速ロード（学習・バッチ処理向け）
- **制約**: Pythonのみ利用可能

```python
import pickle

with open("output/stage2/<dataset>/pkl/<loop_id>.stage2.pkl", "rb") as f:
    data = pickle.load(f)

print(data["extended.tempo_map"])  # [[0, 120.0]]
```

---

## CSV出力（`--emit-csv aggregate`）

### `stage2_index.csv`（1曲=1行）

| カラム | 型 | 説明 |
|--------|---|------|
| `loop_id` | string | ループID |
| `file` | string | ファイル名 |
| `score` | float | 総合スコア |
| `bars` | int | 小節数 |
| `bpm0` | float | 初期テンポ |
| `timesig0` | string | 初期拍子 |
| `pb_min` | int | PB最小値 |
| `pb_max` | int | PB最大値 |
| `rpn_seen` | bool | RPN使用有無 |
| `n_tracks` | int | トラック数 |
| `n_drums` | int | ドラムトラック数 |
| `n_notes_total` | int | 総ノート数 |
| `extended.*` | string | 拡張メタ（JSON文字列） |

### 将来実装予定のCSV（未実装）

- `bar_features.csv`: bar単位の詳細（energy, chord, key）
- `controls_summary.csv`: トラック×CC/PBの統計
- `roles_map.csv`: トラック単位のロール推定

---

## Sunoアレンジ工程との関係

| 工程 | 必要なファイル | CSV要否 |
|------|--------------|---------|
| Stage1クリーニング | `cleaned.pickle` | ❌ |
| Stage2拡張メタ抽出 | `*.stage2.json`, `*.stage2.pkl` | ❌ |
| セクション検出 | `sections.json` | ❌ |
| コード進行抽出 | `chordmap.json` | ❌ |
| **学習データ準備** | `*.stage2.pkl` | ⭕（監査用） |
| **Sunoアレンジ実行** | `*.stage2.json`, `sections.json`, `chordmap.json` | ❌ |

### まとめ

- **ランタイム実行**: JSON/PKL のみで完結
- **データ監査**: `--emit-csv aggregate` で統計・可視化
- **学習前チェック**: CSVで異常値検出・分布確認

---

## 実装状況（v2.1）

### ✅ 完了

1. argparse に `--emit-json`, `--emit-pickle`, `--emit-csv` フラグ追加
2. `Stage2Paths` に `json_dir`, `pkl_dir`, `csv_dir` フィールド追加
3. `Stage2Settings` に `emit_json`, `emit_pickle`, `emit_csv` フラグ追加
4. `_resolve_paths()` でディレクトリ自動生成
5. `_build_settings()` でフラグ設定
6. `Stage2Extractor.run()` に JSON/PKL 出力ロジック追加
   - ストリーミングモード: 逐次書き出し
   - 非ストリーミングモード: バッチ書き出し
7. 7つの拡張メタ関数実装（スケルトン版）
   - `extract_tempo_grid_extended()`
   - `extract_bar_chords_extended()`
   - `estimate_local_keys_extended()`
   - `auto_sections_from_energy_extended()`
   - `analyze_groove_extended()`（スケルトン）
   - `summarize_controls_extended()`
   - `estimate_roles_extended()`

### 🔄 進行中

- `auto_sections_from_energy_extended()` のバグ修正（sections配列が空になる問題）
  - **原因**: スモークテストでのみ発生、実際の処理では正常動作を確認済み

### ⏳ 未実装

- CSV集計機能（`--emit-csv aggregate`）の詳細実装
  - `bar_features.csv`
  - `controls_summary.csv`
  - `roles_map.csv`
- `analyze_groove_extended()` の実装（groove_sampler_v2活用）
- stem_harmonyベースのコード推定（現在はPC-set簡易版）
- RMS/noveltyベースのエネルギー計算（現在は実ノート数ベース）

---

## 次のステップ

1. **バグ修正**: `auto_sections_from_energy_extended()` のスモークテスト問題解決
2. **統合テスト**: LAMDA 10ファイルで JSON/PKL/CSV 出力確認
3. **CSV集計実装**: 4つのCSV生成ロジック追加（任意）
4. **本格実装**: stem_harmony, groove_sampler_v2 統合（後回し）

---

## 参考

- 設計仕様: ユーザー提供の「結論から」セクション
- 実装パッチ: 外部チーム提供の6点改善diff
- テストスクリプト: `test_extended_meta.py`（7つの関数独立テスト）
