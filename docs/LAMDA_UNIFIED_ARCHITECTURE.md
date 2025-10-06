# LAMDa Unified Architecture
# ===========================
# Los Angeles MIDI Dataset の統合アーキテクチャ設計

## 設計思想: 連邦制データアーキテクチャ

LAMDa データセットは**5つの独立した離れ小島ではなく、連邦のように助け合う統合システム**として設計されています。
CODE フォルダが「法律」として各コンポーネント間の統合ルールを定義しています。

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAMDa Unified System                          │
│                                                                   │
│  ┌────────────────┐   ┌────────────────┐   ┌─────────────────┐ │
│  │  CHORDS_DATA   │   │ KILO_CHORDS    │   │  SIGNATURES     │ │
│  │   (15GB)       │◄──┤    (602MB)     │◄──┤    (290MB)      │ │
│  │                │   │                │   │                 │ │
│  │ 詳細MIDI       │   │ 整数シーケンス │   │ 楽曲特徴量      │ │
│  │ イベント       │   │ (高速検索)     │   │ (類似度)        │ │
│  └───────┬────────┘   └────────────────┘   └─────────────────┘ │
│          │                                                       │
│          │              ┌────────────────┐                      │
│          └──────────────┤ CODE (Laws)    │                      │
│                         │  - TMIDIX.py   │                      │
│                         │  - MIDI.py     │                      │
│                         │  - Decoders    │                      │
│                         └────────────────┘                      │
│                                                                   │
│          ┌────────────────┐   ┌────────────────┐                │
│          │ TOTALS_MATRIX  │   │   META_DATA    │                │
│          │    (33MB)      │   │    (4.2GB)     │                │
│          │                │   │                │                │
│          │ 統計マトリックス│   │ 楽曲メタデータ │                │
│          │ (正規化)       │   │ (検索)         │                │
│          └────────────────┘   └────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

## データコンポーネント詳細

### 1. CHORDS_DATA (15GB, 162 files)
**役割**: 最も詳細な音楽情報を保持する「原典データ」

**フォーマット**: 
```python
[
    [hash_id, [
        [time_delta, dur1, patch1, pitch1, vel1, dur2, patch2, pitch2, vel2, ...],
        [time_delta, dur1, patch1, pitch1, vel1, ...],
        ...
    ]],
    ...
]
```

**構造解説**:
- `time_delta`: 前のイベントからの時間差 (×16でミリ秒)
- 4要素グループ: `[duration, patch, pitch, velocity]`
- 複数の4要素グループ = 和音 (polyphony)

**使用例**:
```python
# イベント例: [113, 41, 0, 51, 55, 44, 0, 39, 49]
# → time_delta=113, 
#    note1=(dur=41, patch=0, pitch=51, vel=55),
#    note2=(dur=44, patch=0, pitch=39, vel=49)
# = 2音の和音
```

**用途**:
- コード進行抽出 (music21でコード名推定)
- 詳細な音楽分析 (リズムパターン, ベロシティ変化)
- MIDI再構築

### 2. KILO_CHORDS_DATA (602MB)
**役割**: 「高速検索インデックス」 - 整数エンコードされたコードシーケンス

**フォーマット**:
```python
[
    [hash_id, [68, 68, 68, 68, 63, 66, 68, 73, 71, ...]],
    ...
]
```

**用途**:
- **高速コード進行マッチング**: 整数配列の比較は文字列より高速
- **パターン検索**: `[68, 68, 63]` のような進行パターンを瞬時に検索
- **類似度計算**: レーベンシュタイン距離, 動的計画法アルゴリズムに適合

**利点**:
- CHORDS_DATAを毎回パースする必要がない
- メモリ効率的 (整数配列)
- 既存の文字列マッチングアルゴリズムが適用可能

### 3. SIGNATURES_DATA (290MB)
**役割**: 「楽曲指紋」 - ピッチ/コード出現頻度の統計

**フォーマット**:
```python
[
    [hash_id, [
        [pitch1, count1],
        [pitch2, count2],
        ...
    ]],
    ...
]
```

**用途**:
- **キー推定**: 最頻出ピッチから調性を推定
- **類似楽曲検索**: コサイン類似度, ユークリッド距離で比較
- **ジャンル分類**: 特徴ベクトルとして機械学習に使用

**例**:
```python
# Signatures例
[[60, 150], [64, 120], [67, 110]]  # C, E, Gが多い → Cメジャーキー
```

### 4. TOTALS_MATRIX (33MB)
**役割**: 「統計的正規化層」 - データセット全体の統計

**フォーマット**: 多次元配列 (event × pitch × count の集計)

**用途**:
- **TF-IDF正規化**: 頻出ピッチの重み付け調整
- **異常検知**: データセット平均からの逸脱度計算
- **特徴量スケーリング**: 機械学習モデルの入力正規化

### 5. META_DATA (4.2GB, 5 files)
**役割**: 「コンテキスト情報」 - 作曲家, テンポ, 拍子など

**内容** (推定):
- 作曲家名
- 曲名
- テンポ (BPM)
- 拍子 (4/4, 3/4, etc.)
- ジャンル

**用途**:
- **フィルタリング**: "バッハの曲のみ"
- **メタデータ検索**: テンポ120-140のジャズ曲
- **コンテキスト推薦**: 同じ作曲家の類似曲

### 6. CODE (Integration Layer)
**役割**: 「連邦の法律」 - 各コンポーネントを統合するライブラリ

**主要ファイル**:

#### TMIDIX.py (4799 lines)
- 完全なMIDI処理エンジン
- CHORDS_DATA ↔ MIDI 相互変換
- チャンネル管理 (16チャンネル, ch.9=ドラム)
- パッチ追跡

#### MIDI.py
- 基本MIDI I/O操作

#### los_angeles_midi_dataset_chords_data_decoder.py
- 公式デコーダー実装
- 4要素グループ抽出アルゴリズム:
```python
for i in range(0, len(ss[1:]), 4):
    s = ss[1:][i:i+4]
    dur, patch, pitch, vel = s[0], s[1], s[2], s[3]
```

## 統合活用パターン

### パターン1: コード進行推薦システム
```python
# 1. ユーザー入力: "C → Am → F → G"
user_progression = ["C", "Am", "F", "G"]

# 2. KILO_CHORDS_DATA で高速検索
similar_sequences = find_similar_kilo_sequences(user_progression)

# 3. SIGNATURES_DATA で特徴マッチング
for seq in similar_sequences:
    signature = get_signature(seq.hash_id)
    similarity_score = cosine_similarity(user_signature, signature)

# 4. META_DATA でフィルタリング
filtered = [s for s in similar_sequences if s.genre == "Jazz"]

# 5. CHORDS_DATA で詳細取得
for result in filtered:
    detailed_progression = extract_from_chords_data(result.hash_id)
```

### パターン2: キーベースの楽曲検索
```python
# 1. SIGNATURES_DATA からキー推定
for song in signatures:
    top_pitches = song.top_pitches[:7]  # トップ7音
    key = estimate_key(top_pitches)  # C major, A minor, etc.

# 2. TOTALS_MATRIX で正規化
normalized_signature = normalize_with_totals(song.signature)

# 3. 類似キーの楽曲をKILO_CHORDS_DATAから取得
similar_songs = kilo_chords[kilo_chords.key == key]

# 4. CHORDS_DATAで詳細分析
```

### パターン3: スタイル転送
```python
# 1. META_DATAから作曲家スタイル抽出
bach_songs = meta_data[meta_data.composer == "Bach"]

# 2. SIGNATURES_DATAでスタイル特徴量計算
bach_style = aggregate_signatures(bach_songs)

# 3. KILO_CHORDS_DATAでコード進行パターン抽出
bach_patterns = extract_patterns(bach_songs, kilo_chords)

# 4. ユーザー曲をスタイル変換
user_song_transformed = apply_style(user_song, bach_patterns, bach_style)
```

## 我々のリポジトリへの統合

### 現状の課題
```python
# lamda_analyzer.py (旧バージョン)
# ❌ CHORDS_DATAのみ使用
# ❌ music21で毎回パース (遅い)
# ❌ 他のデータソースを活用していない
```

### 統合アーキテクチャ (lamda_unified_analyzer.py)
```python
# ✅ 全データソース統合
# ✅ KILO_CHORDS で高速検索
# ✅ SIGNATURES で類似度マッチング
# ✅ TOTALS で正規化
# ✅ CODE/TMIDIX.py との互換性
```

### データベーススキーマ
```sql
-- progressions: CHORDS_DATAから抽出
CREATE TABLE progressions (
    id INTEGER PRIMARY KEY,
    hash_id TEXT NOT NULL,
    progression TEXT,  -- JSON: [{chord, root, notes, time_delta}, ...]
    total_events INTEGER,
    chord_events INTEGER,
    source_file TEXT
);

-- kilo_sequences: KILO_CHORDS_DATAから
CREATE TABLE kilo_sequences (
    hash_id TEXT PRIMARY KEY,
    sequence TEXT,  -- "[68, 68, 63, 66, ...]"
    sequence_length INTEGER
);

-- signatures: SIGNATURES_DATAから
CREATE TABLE signatures (
    hash_id TEXT PRIMARY KEY,
    pitch_distribution TEXT,  -- "[[60, 150], [64, 120], ...]"
    top_pitches TEXT  -- "[[60, 150], [64, 120], [67, 110]]"
);

-- 全テーブルがhash_idで紐付け可能 (連邦制)
```

## 実行計画

### Vertex AI実行
```bash
# 新しい統合スクリプトを使用
python scripts/build_lamda_unified_db.py

# または Notebookガイド
# docs/vertex_ai_lamda_unified_guide.py の Cell 1-7 を実行
```

### ローカル開発
```python
from lamda_unified_analyzer import LAMDaUnifiedAnalyzer

analyzer = LAMDaUnifiedAnalyzer(Path('data/Los-Angeles-MIDI'))

# データベース構築
analyzer.build_unified_database(Path('lamda_unified.db'))

# 活用例
kilo_data = analyzer.load_kilo_chords()
signatures = analyzer.load_signatures()
totals = analyzer.load_totals_matrix()
```

## まとめ

LAMDaは**単なるMIDIデータセットではなく、統合的な音楽情報システム**です:

1. **CHORDS_DATA**: 原典データ (詳細)
2. **KILO_CHORDS_DATA**: 検索インデックス (高速)
3. **SIGNATURES_DATA**: 特徴量 (類似度)
4. **TOTALS_MATRIX**: 統計層 (正規化)
5. **META_DATA**: コンテキスト (検索)
6. **CODE**: 統合ライブラリ (法律)

この**連邦制アーキテクチャ**を活用することで、我々のリポジトリは:
- より高速な検索
- より精度の高い推薦
- より柔軟な分析

が可能になります 🎵
