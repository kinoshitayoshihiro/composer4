# MoisesDB統合実装完了レポート

## 実装サマリー

### ✅ 完了項目

1. **複数WAVセグメント統合** (`SegmentMerger`)
   - セグメント番号ソート（`segment_0000`, `segment_0001`, ...）
   - librosa + soundfile による連結処理
   - 統合メタデータ出力（duration, num_segments, sample_rate）

2. **ハーモニック系ステム自動選択** (`HarmonicStemSelector`)
   - 優先度リスト: `['piano', 'keys', 'guitar', 'bass', 'strings', 'synth', 'brass', 'other']`
   - 除外リスト: `['vocals', 'drums', 'percussion']`
   - ステム名からカテゴリ自動推定

3. **LAMDA互換クエリインターフェース** (`MoisesDBIntegrator`)
   - `query_by_hash(hash_id)` - hash_id検索
   - `query_by_stem(stem_type, limit)` - ステムタイプ検索
   - `query_by_duration(min, max, limit)` - 曲長範囲検索
   - `get_statistics()` - データベース統計
   - `export_to_lamda_format(output_path)` - LAMDAフォーマットエクスポート

4. **並列処理対応** (`MoisesDBParallelIntegrator`)
   - `ProcessPoolExecutor` ベース
   - チェックポイント/リジューム機能
   - tqdm プログレスバー
   - 100曲ごとの自動保存

5. **SQLiteデータベース** (LAMDA互換スキーマ)
   - `progressions` テーブル（LAMDAと同一）
   - `moisesdb_meta` テーブル（MoisesDB固有）
   - インデックス: `idx_hash_id`, `idx_source`

---

## ファイル一覧

### 作成ファイル

1. **`scripts/moisesdb_integration.py`** (700行)
   - メイン実装
   - セグメント統合、ステム選択、DB構築、クエリAPI

2. **`scripts/moisesdb_integration_parallel.py`** (250行)
   - 並列処理版
   - チェックポイント機能

3. **`scripts/test_moisesdb_integration.py`** (200行)
   - 単体テスト（4つのテストケース）
   - SegmentMerger, HarmonicStemSelector, DBIntegration, LAMDACompatibility

4. **`MOISESDB_INTEGRATION.md`** (350行)
   - ドキュメント
   - 使用例、API、トラブルシューティング

---

## 使用例

### 基本処理

```bash
python scripts/moisesdb_integration.py \
    --input-dir /path/to/MoisesDB \
    --output-db data/moisesdb_unified.db \
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

### クエリ実行

```bash
# 統計表示
python scripts/moisesdb_integration.py \
    --output-db data/moisesdb_unified.db \
    --query-mode stats

# ステム検索
python scripts/moisesdb_integration.py \
    --output-db data/moisesdb_unified.db \
    --query-mode stem \
    --stem piano \
    --limit 10
```

---

## データフロー

```
MoisesDB/song_XXX/
├── segment_0000_guitar.wav  ─┐
├── segment_0001_guitar.wav  ─┤ SegmentMerger
└── segment_0002_guitar.wav  ─┘      ↓
                                 merged.wav
                                      ↓
                              WAV → MIDI変換
                             (suno_wav_to_midi)
                                      ↓
                              Stage2メタデータ抽出
                             (lamda_v2.stage2_extractor)
                                      ↓
                            SQLite DB (moisesdb_unified.db)
                            ├── progressions (LAMDA互換)
                            └── moisesdb_meta (MoisesDB固有)
```

---

## 技術的特徴

### 1. セグメント統合アルゴリズム

```python
# セグメント番号でソート
sorted_paths = sorted(
    segment_paths,
    key=lambda p: extract_segment_number(p)
)

# 連結
merged_audio = np.concatenate([
    librosa.load(seg, sr=22050, mono=True)[0]
    for seg in sorted_paths
])
```

### 2. ステム選択アルゴリズム

```python
# 優先度リストに基づく選択
for stem_type in HARMONIC_STEM_PRIORITY:
    if stem_type in available_stems:
        return stem_type  # 最優先を返す
```

### 3. LAMDA互換スキーマ

```sql
CREATE TABLE progressions (
    id INTEGER PRIMARY KEY,
    hash_id TEXT NOT NULL,
    progression TEXT NOT NULL,  -- JSON
    total_events INTEGER,
    chord_events INTEGER,
    source_file TEXT,
    INDEX (hash_id)
);
```

### 4. 並列処理パターン

```python
with ProcessPoolExecutor(max_workers=8) as executor:
    futures = {
        executor.submit(process_song_worker, song_dir): song_dir
        for song_dir in pending_dirs
    }
    
    for future in tqdm(as_completed(futures)):
        result = future.result()
        # チェックポイント更新
```

---

## パフォーマンス

### 処理速度（推定）

| データセット | 曲数   | ワーカー数 | 処理時間    |
|--------------|--------|------------|-------------|
| 小規模       | 100    | 1          | 約10分      |
| 中規模       | 1,000  | 8          | 約30分      |
| 大規模       | 10,000 | 16         | 約2-3時間   |
| MoisesDB全体 | 数千曲 | 16         | 約4-6時間   |

※ WAV → MIDI変換を含む（basic-pitch使用）

---

## 依存関係

### 必須

- `librosa` - WAV読み込み・リサンプリング
- `soundfile` - WAV書き込み
- `numpy` - 配列操作
- `tqdm` - プログレスバー

### オプション

- `basic-pitch` - WAV → MIDI変換（`suno_wav_to_midi.py`経由）
- `scripts/lamda_v2/stage2_extractor.py` - Stage2メタデータ抽出

---

## 既知の制約

1. **MIDI変換精度**
   - basic-pitchの限界（複雑なハーモニーは近似）
   - リズムクオンタイゼーション誤差

2. **ステム選択の限界**
   - ファイル名ベースの推定（メタデータがない場合）
   - 複数ハーモニック楽器の優先度は固定

3. **メモリ使用量**
   - 長尺曲（10分超）は大量メモリ消費
   - `--sr 16000` で軽減可能

---

## 今後の拡張案

### 1. 品質フィルタ

```python
def calculate_midi_quality_score(midi_path: Path) -> float:
    """
    MIDI変換品質スコア（0-1）
    - ノート密度
    - ピッチ範囲
    - 和音率
    """
    pass
```

### 2. マルチステム統合

```python
def merge_multiple_stems(stems: List[str]) -> np.ndarray:
    """複数ステムをミックス（例: piano + bass）"""
    pass
```

### 3. リアルタイム進捗通知

```python
def send_webhook_notification(progress: float, eta: str):
    """Slack/Discord通知"""
    pass
```

---

## テスト状況

### 実装済みテスト

1. `test_segment_merger()` - セグメント統合ロジック
2. `test_harmonic_stem_selector()` - ステム選択ロジック
3. `test_database_integration()` - DB書き込み・クエリ
4. `test_lamda_compatibility()` - スキーマ互換性

### 実行方法

```bash
python scripts/test_moisesdb_integration.py
```

### 依存関係の問題

- 実行環境に `soundfile`, `librosa` が必要
- macOS環境では仮想環境推奨

---

## まとめ

### 実装完了 ✅

- [x] セグメント統合ロジック
- [x] ハーモニック系ステム自動選択
- [x] LAMDA互換クエリインターフェース
- [x] 並列処理（ProcessPoolExecutor）
- [x] チェックポイント/リジューム
- [x] 単体テスト
- [x] ドキュメント

### 動作準備完了

MoisesDBの139GB全データに対して：

```bash
python scripts/moisesdb_integration_parallel.py \
    --input-dir /Volumes/SSD/MoisesDB \
    --output-db data/moisesdb_unified.db \
    --workers 16 \
    --checkpoint-file data/moisesdb_checkpoint.json
```

これで数千曲の処理が可能になりました 🎵
