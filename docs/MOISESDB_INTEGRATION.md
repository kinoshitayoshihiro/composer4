# MoisesDB Integration

**WAV版 LOCAL LAMDA統合システム**

MoisesDBの複数セグメントWAVファイルを統合し、LAMDA互換のSQLiteデータベースを構築します。

---

## 特徴

### ✅ 実装済み

1. **複数WAVセグメント統合**
   - 1曲 = N個のセグメントを自動結合
   - セグメント番号でソート（`segment_0000`, `segment_0001`, ...）
   - librosa + soundfile によるシームレス結合

2. **ハーモニック系ステム自動選択**
   - 優先度: `piano > keys > guitar > bass > strings > synth > brass > other`
   - 除外: `vocals`, `drums`, `percussion`
   - 複数ステムから最適な1つを自動選択

3. **LAMDA互換クエリインターフェース**
   - `query_by_hash()`: hash_id検索
   - `query_by_stem()`: ステムタイプで検索
   - `query_by_duration()`: 曲長範囲検索
   - `get_statistics()`: データベース統計

4. **並列処理対応**
   - `ProcessPoolExecutor` による高速化
   - チェックポイント/リジューム機能
   - tqdm プログレスバー

---

## インストール

```bash
# 依存関係（基本）
pip install librosa soundfile numpy tqdm

# WAV → MIDI変換（オプション）
pip install basic-pitch

# LAMDA Stage2（オプション）
# scripts/lamda_v2/stage2_extractor.py が必要
```

---

## 使用例

### 基本処理（単一プロセス）

```bash
python scripts/moisesdb_integration.py \
    --input-dir /path/to/MoisesDB \
    --output-db data/moisesdb_unified.db \
    --midi-output-dir data/moisesdb_midi \
    --max-songs 100 \
    --verbose
```

### 並列処理（推奨: 大規模データセット）

```bash
python scripts/moisesdb_integration_parallel.py \
    --input-dir /path/to/MoisesDB \
    --output-db data/moisesdb_unified.db \
    --workers 8 \
    --checkpoint-file data/moisesdb_checkpoint.json
```

### クエリモード

```bash
# hash_id検索
python scripts/moisesdb_integration.py \
    --output-db data/moisesdb_unified.db \
    --query-mode hash \
    --hash-id abc123...

# ステム検索
python scripts/moisesdb_integration.py \
    --output-db data/moisesdb_unified.db \
    --query-mode stem \
    --stem piano \
    --limit 10

# 曲長検索（60-180秒）
python scripts/moisesdb_integration.py \
    --output-db data/moisesdb_unified.db \
    --query-mode duration \
    --min-duration 60 \
    --max-duration 180

# 統計表示
python scripts/moisesdb_integration.py \
    --output-db data/moisesdb_unified.db \
    --query-mode stats
```

---

## ディレクトリ構造

### 入力（MoisesDB）

```
MoisesDB/
├── song_001/
│   ├── segment_0000_vocals.wav
│   ├── segment_0000_drums.wav
│   ├── segment_0000_guitar.wav
│   ├── segment_0000_piano.wav
│   ├── segment_0001_vocals.wav
│   ├── segment_0001_drums.wav
│   └── ...
├── song_002/
│   └── ...
└── song_XXX/
    └── ...
```

### 出力

```
data/
├── moisesdb_unified.db       # SQLite（LAMDA互換スキーマ）
├── moisesdb_unified.jsonl    # 処理結果メタデータ
├── moisesdb_checkpoint.json  # 並列処理チェックポイント
└── moisesdb_midi/
    ├── song_001.mid
    ├── song_001_guitar.wav   # 統合済みWAV
    ├── song_002.mid
    └── ...
```

---

## データベーススキーマ

### `progressions` テーブル（LAMDA互換）

| カラム         | 型      | 説明                     |
|----------------|---------|--------------------------|
| id             | INTEGER | 主キー                   |
| hash_id        | TEXT    | 曲の一意ID               |
| progression    | TEXT    | コード進行（JSON）       |
| total_events   | INTEGER | 総イベント数             |
| chord_events   | INTEGER | コードイベント数         |
| source_file    | TEXT    | ソース曲ID               |

### `moisesdb_meta` テーブル（MoisesDB固有）

| カラム           | 型      | 説明                          |
|------------------|---------|-------------------------------|
| song_id          | TEXT    | 主キー（曲ID）                |
| hash_id          | TEXT    | hash_id（progressions紐付け） |
| duration         | REAL    | 曲長（秒）                    |
| num_segments     | INTEGER | セグメント数                  |
| selected_stem    | TEXT    | 選択されたステム              |
| available_stems  | TEXT    | 利用可能ステム（JSON）        |
| midi_path        | TEXT    | MIDI変換パス                  |

---

## Python APIサンプル

```python
from pathlib import Path
from scripts.moisesdb_integration import MoisesDBIntegrator

# Integrator初期化
integrator = MoisesDBIntegrator(
    db_path=Path('data/moisesdb_unified.db'),
    midi_output_dir=Path('data/moisesdb_midi'),
    sr=22050
)

# データセット処理
results = integrator.process_dataset(
    input_dir=Path('/path/to/MoisesDB'),
    max_songs=100,
    verbose=True
)

# クエリ実行
# 1. hash検索
song_data = integrator.query_by_hash('abc123...')
print(song_data['progression'])

# 2. ステム検索
piano_songs = integrator.query_by_stem('piano', limit=10)
for song in piano_songs:
    print(f"{song['song_id']}: {song['duration']:.2f}s")

# 3. 統計
stats = integrator.get_statistics()
print(f"Total songs: {stats['total_songs']}")
print(f"Stem distribution: {stats['stem_counts']}")
```

---

## テスト

```bash
# 全テスト実行
python scripts/test_moisesdb_integration.py

# 個別テスト
python -m pytest scripts/test_moisesdb_integration.py::test_segment_merger
python -m pytest scripts/test_moisesdb_integration.py::test_harmonic_stem_selector
python -m pytest scripts/test_moisesdb_integration.py::test_database_integration
```

---

## パフォーマンス

### 処理速度（参考値）

| データセット | 曲数   | ワーカー数 | 処理時間    |
|--------------|--------|------------|-------------|
| 小規模       | 100    | 1          | 約10分      |
| 中規模       | 1,000  | 8          | 約30分      |
| 大規模       | 10,000 | 16         | 約2-3時間   |

※ WAV → MIDI変換を含む（basic-pitch使用）

### 最適化のヒント

1. **ワーカー数**: `--workers $(nproc)` でCPUコア数に合わせる
2. **チェックポイント**: 100曲ごとに自動保存（中断しても再開可能）
3. **リサンプリング**: `--sr 16000` で低レートに（速度優先）
4. **MIDI変換スキップ**: `suno_wav_to_midi.py` を無効化（インポートエラーで自動スキップ）

---

## トラブルシューティング

### Q1: `⚠️ suno_wav_to_midi not available`

**A**: MIDI変換機能が無効化されます（WAV統合のみ実行）。有効化する場合:

```bash
pip install basic-pitch
# scripts/suno_wav_to_midi.py が必要
```

### Q2: `⚠️ No harmonic stem found`

**A**: vocals/drumsのみの曲がスキップされます。ログで確認:

```bash
grep "No harmonic stem" moisesdb_integration.log
```

### Q3: メモリ不足

**A**: ワーカー数を減らす + リサンプリングレートを下げる:

```bash
python scripts/moisesdb_integration_parallel.py \
    --workers 4 \
    --sr 16000
```

### Q4: SQLite lock エラー

**A**: 並列書き込みの競合。各ワーカーで独立したコネクションを使用（実装済み）

---

## ロードマップ

### 実装済み ✅

- [x] セグメント統合ロジック
- [x] ハーモニック系ステム自動選択
- [x] LAMDA互換クエリインターフェース
- [x] 並列処理対応
- [x] チェックポイント/リジューム
- [x] 単体テスト

### 今後の拡張 🔜

- [ ] 品質フィルタ（MIDI変換品質スコアリング）
- [ ] GPU加速（CUDA対応WAV処理）
- [ ] リアルタイムプログレス通知（Webhook）
- [ ] マルチモーダル特徴量（Audio + MIDI）
- [ ] データセット分割（train/val/test）

---

## 関連ファイル

- `scripts/moisesdb_integration.py` - メイン実装
- `scripts/moisesdb_integration_parallel.py` - 並列処理版
- `scripts/test_moisesdb_integration.py` - テストスイート
- `lamda_unified_analyzer.py` - LAMDA統合アナライザー（参考）
- `scripts/suno_wav_to_midi.py` - WAV → MIDI変換
- `scripts/lamda_v2/stage2_extractor.py` - Stage2メタデータ抽出

---

## ライセンス

本プロジェクトのライセンスに従います。
