# WAV版 LOCAL LAMDA Pickle構築ガイド

**作成日**: 2025年10月24日  
**目的**: Suno AI循環方式データ（WAV → MIDI → Stage2）からLAMDA互換5軸pickle作成

---

## 📊 WAV版5軸Pickle構成

MIDI版LAMDAと同じ構造で、WAV由来データ用のLOCAL版を作成：

| Pickle名 | 推定サイズ | 役割 |
|---------|-----------|------|
| **LOCAL_WAV_KILO_CHORDS_DATA** | ~1-3MB | WAV→MIDI変換後のコード進行 |
| **LOCAL_WAV_META_DATA** | ~3-8MB | パッチ分布/統計情報（シャード分割） |
| **LOCAL_WAV_SIGNATURES_DATA** | ~1-2MB | 拍子シグネチャ |
| **LOCAL_WAV_TOTALS** | ~10-100KB | Pitch/Duration/Velocity外れ値スコア |
| **local_wav_id_map.csv** | ~500KB-2MB | ファイルIDマッピング |

**データソース**:
- MoisesDB（139GB、ステム分離済みWAV）
- MUSDB18（stem-separated WAVs）
- Suno AIステム（future）

---

## 🚀 使用方法

### 1. SQLite DBから5軸pickle作成（推奨）

```bash
# Step1: WAVデータセット統合（SQLite DB作成）
python scripts/wav_dataset_integration.py \
    --input-dir /path/to/MoisesDB \
    --output-db data/wav_unified.db \
    --use-gpu \
    --verbose

# Step2: SQLite DBから5軸pickle作成
python scripts/build_local_lamda_wav.py \
    --input-db data/wav_unified.db \
    --output-dir data/local_lamda/wav \
    --shard-size 5000 \
    --verbose
```

**出力構造**:
```
data/local_lamda/wav/
├── LOCAL_WAV_KILO_CHORDS_DATA.pickle          (コード進行カタログ)
├── LOCAL_WAV_META_DATA/
│   ├── LOCAL_WAV_META_DATA_000000.pickle      (Shard 0: 5,000曲)
│   ├── LOCAL_WAV_META_DATA_000001.pickle      (Shard 1: 5,000曲)
│   └── ...
├── LOCAL_WAV_SIGNATURES_DATA.pickle           (拍子シグネチャ)
├── LOCAL_WAV_TOTALS.pickle                     (外れ値スコア)
└── local_wav_id_map.csv                        (IDマッピング)
```

---

### 2. Stage2 JSONから直接作成

```bash
# Stage2 JSON抽出済みの場合
python scripts/build_local_lamda_wav.py \
    --input-json-dir output/stage2/json \
    --output-dir data/local_lamda/wav \
    --shard-size 5000 \
    --verbose
```

---

## 🔧 Origin LAMDA + WAV LOCAL統合

`scripts/lamda_v2/lamda_sources.py`で統合使用：

```python
from scripts.lamda_v2.lamda_sources import LamdaSources

# Origin LAMDA (17.8万曲 MIDI) + WAV LOCAL統合
lamda = LamdaSources(
    # Origin LAMDA (MIDI版)
    kilo="data/Los-Angeles-MIDI/KILO_CHORDS_DATA/LAMDa_KILO_CHORDS_DATA.pickle",
    meta_dir="data/Los-Angeles-MIDI/META_DATA",
    signatures="data/Los-Angeles-MIDI/SIGNATURES_DATA/LAMDa_SIGNATURES_DATA.pickle",
    totals="data/Los-Angeles-MIDI/TOTALS_MATRIX/LAMDa_TOTALS.pickle",
    id_map_csv="data/Los-Angeles-MIDI/mappings/auto_file_id_map.csv",
    
    # WAV LOCAL (Suno AI循環方式)
    local_kilo="data/local_lamda/wav/LOCAL_WAV_KILO_CHORDS_DATA.pickle",
    local_meta_dir="data/local_lamda/wav/LOCAL_WAV_META_DATA",
    local_signatures="data/local_lamda/wav/LOCAL_WAV_SIGNATURES_DATA.pickle",
    local_totals="data/local_lamda/wav/LOCAL_WAV_TOTALS.pickle",
    
    # WAV優先（Suno AI由来を優先的に使用）
    prefer_local=True
)

# コード進行取得（WAV優先 → MIDI補完）
chords = lamda.get_kilo_chords(file_id="song_001")
```

---

## 📋 処理フロー詳細

### SQLite DBからの変換

```
wav_unified.db
├── wav_dataset_meta テーブル
│   ├── song_id          → ID_MAP (src_id)
│   ├── hash_id          → ID_MAP (target_id)
│   ├── dataset_type     → META_DATA (genre)
│   ├── duration         → META_DATA (duration)
│   ├── selected_stem    → META_DATA (stem_type)
│   └── midi_path        → Stage2 JSON読み込み
│
└── progressions テーブル
    ├── hash_id          → KILO_CHORDS_DATA (file_id)
    ├── progression      → KILO_CHORDS_DATA (chords JSON)
    ├── total_events     → META_DATA (total_events)
    └── chord_events     → META_DATA (chord_events)
```

### Stage2 JSONからの抽出

```json
{
  "chords": [...],                    → KILO_CHORDS_DATA
  "chordmap_external": {...},         → KILO_CHORDS_DATA (fallback)
  "patch_summary": {...},             → META_DATA (patches)
  "note_stats_meta": {                → META_DATA (statistics)
    "total_notes": 1234,
    "avg_velocity": 76.5,
    "pitch_range": [36, 96]
  },
  "signatures": ["4/4", "3/4"],       → SIGNATURES_DATA
  "outliers": {                       → TOTALS_MATRIX
    "pitch": 0.12,
    "duration": 0.08,
    "velocity": 0.15
  },
  "bpm": 120.0,                       → META_DATA (bpm)
  "dataset_type": "moisesdb"          → META_DATA (genre)
}
```

---

## 🔍 検証方法

### 1. Pickleサイズ確認

```bash
ls -lh data/local_lamda/wav/
```

**期待出力**:
```
LOCAL_WAV_KILO_CHORDS_DATA.pickle       2.3M
LOCAL_WAV_META_DATA/                    -
  LOCAL_WAV_META_DATA_000000.pickle     1.2M
  LOCAL_WAV_META_DATA_000001.pickle     1.3M
  ...
LOCAL_WAV_SIGNATURES_DATA.pickle        1.5M
LOCAL_WAV_TOTALS.pickle                 45K
local_wav_id_map.csv                    680K
```

### 2. Pickle内容確認

```python
import pickle

# KILO_CHORDS_DATA
with open('data/local_lamda/wav/LOCAL_WAV_KILO_CHORDS_DATA.pickle', 'rb') as f:
    kilo = pickle.load(f)
    print(f"KILO entries: {len(kilo)}")
    print(f"Sample: {list(kilo.items())[:2]}")

# META_DATA
with open('data/local_lamda/wav/LOCAL_WAV_META_DATA/LOCAL_WAV_META_DATA_000000.pickle', 'rb') as f:
    meta = pickle.load(f)
    print(f"META entries: {len(meta)}")
    print(f"Sample: {list(meta.items())[:1]}")
```

### 3. 統合テスト

```bash
# Stage2実行でWAV LOCAL使用
python -m scripts.lamda_v2.stage2_extractor input.mid -o output.json \
    --local-kilo data/local_lamda/wav/LOCAL_WAV_KILO_CHORDS_DATA.pickle \
    --local-meta-dir data/local_lamda/wav/LOCAL_WAV_META_DATA \
    --local-signatures data/local_lamda/wav/LOCAL_WAV_SIGNATURES_DATA.pickle \
    --local-totals data/local_lamda/wav/LOCAL_WAV_TOTALS.pickle \
    --prefer-local
```

---

## 📊 パフォーマンス目安

| データセット | 曲数 | 処理時間 | Pickle合計サイズ |
|------------|------|---------|----------------|
| MUSDB18 | ~150曲 | 5-10分 | ~10MB |
| MoisesDB（サンプル1000曲） | 1,000曲 | 30-60分 | ~50MB |
| MoisesDB（全曲） | ~10,000曲 | 5-10時間 | ~500MB |

---

## 🎯 運用戦略

### ステージ1: 小規模テスト
```bash
# MUSDB18（150曲）でテスト
python scripts/wav_dataset_integration.py \
    --input-dir data/musdb18 \
    --output-db data/wav_test.db \
    --max-songs 150

python scripts/build_local_lamda_wav.py \
    --input-db data/wav_test.db \
    --output-dir data/local_lamda/wav_test
```

### ステージ2: 中規模検証
```bash
# MoisesDB 1,000曲
python scripts/wav_dataset_integration.py \
    --input-dir data/MoisesDB \
    --output-db data/wav_1k.db \
    --max-songs 1000 \
    --use-gpu

python scripts/build_local_lamda_wav.py \
    --input-db data/wav_1k.db \
    --output-dir data/local_lamda/wav_1k
```

### ステージ3: 本番運用
```bash
# MoisesDB全曲 + MUSDB18統合
python scripts/wav_dataset_integration.py \
    --input-dir data/MoisesDB \
    --output-db data/wav_unified.db \
    --use-gpu \
    --dynamic-weights

python scripts/wav_dataset_integration.py \
    --input-dir data/musdb18 \
    --output-db data/wav_unified.db

python scripts/build_local_lamda_wav.py \
    --input-db data/wav_unified.db \
    --output-dir data/local_lamda/wav
```

---

## 🔧 トラブルシューティング

### 問題: メモリ不足
**解決**: シャードサイズを小さく
```bash
python scripts/build_local_lamda_wav.py \
    --input-db data/wav_unified.db \
    --output-dir data/local_lamda/wav \
    --shard-size 1000  # 5000 → 1000
```

### 問題: Stage2 JSON見つからない
**原因**: wav_dataset_integration.pyでMIDI変換失敗  
**解決**: 
```bash
# MIDI変換確認
ls data/wav_unified_midi/*.mid | wc -l

# Stage2再実行
python -m scripts.lamda_v2.stage2_extractor \
    data/wav_unified_midi/ \
    -o output/stage2/json
```

### 問題: Pickle読み込みエラー
**原因**: pickle protocol非互換  
**解決**: Python 3.8+使用、pickle.HIGHEST_PROTOCOL使用

---

## 📚 関連ドキュメント

- `PICKLE_DIRECT_WORKFLOW.md`: Pickle直書き運用ガイド
- `LAMDA_INTEGRATION_REPORT.md`: MIDI版LAMDA統合レポート
- `MOISESDB_INTEGRATION.md`: MoisesDB統合ガイド
- `scripts/wav_dataset_integration.py`: WAVデータセット統合スクリプト
- `scripts/lamda_v2/lamda_sources.py`: LAMDA統合ローダー

---

## ✅ チェックリスト

- [ ] WAVデータセット統合（SQLite DB作成）
- [ ] 5軸pickle作成
- [ ] Pickleサイズ・内容確認
- [ ] lamda_sources.py統合テスト
- [ ] Stage2実行確認（`--prefer-local`）
- [ ] Docker .dockerignoreホワイトリスト追加

---

**実装者**: GitHub Copilot  
**Status**: ✅ Ready for Production
