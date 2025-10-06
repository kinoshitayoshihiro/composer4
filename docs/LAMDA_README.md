# LAMDa Integration Guide
# Los Angeles MIDI Dataset 統合システム完全ガイド

[![LAMDa](https://img.shields.io/badge/LAMDa-Unified-blue)](https://github.com/asigalov61/Los-Angeles-MIDI-Dataset)
[![Architecture](https://img.shields.io/badge/Architecture-Federated-green)](docs/LAMDA_UNIFIED_ARCHITECTURE.md)

## 📖 目次

1. [概要](#概要)
2. [クイックスタート](#クイックスタート)
3. [アーキテクチャ](#アーキテクチャ)
4. [ローカルテスト](#ローカルテスト)
5. [Vertex AI 実行](#vertex-ai-実行)
6. [データベース活用](#データベース活用)
7. [トラブルシューティング](#トラブルシューティング)

---

## 📚 概要

このプロジェクトは [Los Angeles MIDI Dataset (LAMDa)](https://github.com/asigalov61/Los-Angeles-MIDI-Dataset) を**統合的に活用**するシステムです。

### 🎯 特徴

- **連邦制アーキテクチャ**: 5つのデータソースが協調動作
- **高速検索**: KILO_CHORDSによる整数ベース検索
- **類似度マッチング**: SIGNATURESによる特徴量計算
- **統計的正規化**: TOTALS_MATRIXによるデータ正規化
- **完全統合**: 全コンポーネントが hash_id で紐付け

### 📊 データソース

```
CHORDS_DATA (15GB)        → 詳細MIDIイベント (原典データ)
├── 162 pickleファイル
└── 形式: [time_delta, dur, patch, pitch, vel, ...]

KILO_CHORDS_DATA (602MB)  → 整数シーケンス (高速検索)
├── 整数エンコード済みコード進行
└── 形式: [68, 68, 63, 66, ...]

SIGNATURES_DATA (290MB)   → 楽曲特徴量 (類似度)
├── ピッチ/コード出現頻度
└── 形式: [[pitch, count], ...]

TOTALS_MATRIX (33MB)      → 統計マトリックス (正規化)
└── データセット全体の統計情報

META_DATA (4.2GB)         → メタデータ (検索)
└── 作曲家, テンポ, 拍子など

CODE/ (15MB)              → 統合ライブラリ (法律)
├── TMIDIX.py (4799 lines)
└── 各コンポーネントの統合仕様
```

---

## 🚀 クイックスタート

### 前提条件

```bash
# Python 3.10+
python --version

# 依存関係インストール
pip install music21 numpy tqdm
```

### 3ステップで開始

#### Step 1: テストサンプル作成 (ローカル)

```bash
# 100サンプルの小規模テストデータを作成
python scripts/create_test_sample.py
```

**実行時間**: 10-30秒  
**出力**: `data/Los-Angeles-MIDI/TEST_SAMPLE/`

#### Step 2: ローカルテスト実行

```bash
# 小規模データでテストビルド
python scripts/test_local_build.py
```

**実行時間**: 2-5分  
**出力**: `data/test_lamda.db` (テストデータベース)

#### Step 3: Vertex AI でフルビルド

```bash
# 1. Vertex AI Colab Enterprise を開く
# https://console.cloud.google.com/vertex-ai/colab

# 2. Notebook作成して以下を実行
# docs/vertex_ai_lamda_unified_guide.py の Cell 1-7
```

**実行時間**: 90-120分  
**コスト**: ¥30-50  
**出力**: `gs://otobon/lamda/lamda_unified.db`

---

## 🏗️ アーキテクチャ

### 連邦制データモデル

LAMDaは**独立した離れ小島ではなく、連邦のように助け合う**統合システムです:

```
┌─────────────────────────────────────────────────────────┐
│                  LAMDa Federal System                    │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────┐ │
│  │ CHORDS_DATA  │───┤ KILO_CHORDS  │───┤ SIGNATURES  │ │
│  │   (原典)     │   │  (検索索引)  │   │  (特徴量)   │ │
│  └──────┬───────┘   └──────────────┘   └─────────────┘ │
│         │                                                │
│         │       ┌──────────────────────────────┐        │
│         └───────┤ CODE (Integration Layer)     │        │
│                 │  • TMIDIX.py (統合仕様)      │        │
│                 │  • MIDI.py (基本操作)        │        │
│                 └──────────────────────────────┘        │
│                                                           │
│  ┌──────────────┐   ┌──────────────┐                    │
│  │ TOTALS_MATRIX│   │  META_DATA   │                    │
│  │  (統計層)    │   │ (コンテキスト)│                    │
│  └──────────────┘   └──────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

### データベーススキーマ

```sql
-- 全テーブルが hash_id で連邦制リンク

CREATE TABLE progressions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hash_id TEXT NOT NULL,           -- 連邦ID
    progression TEXT NOT NULL,        -- JSON形式のコード進行
    total_events INTEGER,
    chord_events INTEGER,
    source_file TEXT,
    INDEX idx_hash_id (hash_id)
);

CREATE TABLE kilo_sequences (
    hash_id TEXT PRIMARY KEY,         -- 連邦ID
    sequence TEXT NOT NULL,           -- 整数配列
    sequence_length INTEGER
);

CREATE TABLE signatures (
    hash_id TEXT PRIMARY KEY,         -- 連邦ID
    pitch_distribution TEXT NOT NULL, -- [[pitch, count], ...]
    top_pitches TEXT                  -- トップ10ピッチ
);
```

### 統合活用パターン

#### パターン1: コード進行推薦

```python
# 1. ユーザー入力
user_input = ["C", "Am", "F", "G"]

# 2. KILO_CHORDSで高速検索 (整数配列比較)
similar_sequences = search_kilo_chords(user_input)

# 3. SIGNATURESで類似度スコアリング
for seq in similar_sequences:
    signature = get_signature(seq.hash_id)
    score = cosine_similarity(user_sig, signature)

# 4. CHORDS_DATAで詳細取得
best_match = get_detailed_progression(best_seq.hash_id)
```

#### パターン2: キーベース検索

```python
# 1. SIGNATURESからキー推定
key = estimate_key_from_signature(song.hash_id)

# 2. TOTALS_MATRIXで正規化
normalized = normalize_signature(song, totals_matrix)

# 3. 同じキーの楽曲をKILO_CHORDSから高速検索
similar_key_songs = search_by_key(key, kilo_data)
```

詳細は **[Architecture Guide](docs/LAMDA_UNIFIED_ARCHITECTURE.md)** 参照。

---

## 🧪 ローカルテスト

### なぜローカルテストが重要か?

- ✅ **高速反復**: 2-5分で検証完了
- ✅ **コスト削減**: Vertex AI実行前に問題発見
- ✅ **コード検証**: アルゴリズムの動作確認
- ✅ **パフォーマンス推定**: フルビルド時間を予測

### ステップバイステップ

#### 1. テストサンプル作成

```bash
python scripts/create_test_sample.py
```

**何が起こるか?**
- CHORDS_DATAから100サンプル抽出
- 対応するKILO_CHORDS, SIGNATURESを抽出
- TOTALS_MATRIXをコピー
- `TEST_SAMPLE/` ディレクトリに保存

**出力例:**
```
📦 Creating test sample with 100 entries...
   Source: data/Los-Angeles-MIDI
   Output: data/Los-Angeles-MIDI/TEST_SAMPLE

1️⃣ Processing CHORDS_DATA...
   ✅ Created: TEST_SAMPLE/CHORDS_DATA/sample_100.pickle
      Samples: 100
      Hash IDs: 100

2️⃣ Processing KILO_CHORDS_DATA...
   ✅ Created: TEST_SAMPLE/KILO_CHORDS_DATA/sample_100.pickle
      Samples: 97

3️⃣ Processing SIGNATURES_DATA...
   ✅ Created: TEST_SAMPLE/SIGNATURES_DATA/sample_100.pickle
      Samples: 98

✅ Test sample created successfully!
```

#### 2. ローカルビルド実行

```bash
python scripts/test_local_build.py
```

**何が起こるか?**
- LAMDaUnifiedAnalyzerを初期化
- TEST_SAMPLE データを処理
- SQLiteデータベースを構築
- 自動検証と統計表示
- パフォーマンス推定

**出力例:**
```
🧪 LAMDa Local Test - Small Sample (100 entries)

📊 Initializing LAMDaUnifiedAnalyzer...
🔨 Building test database...
   (This should take 2-5 minutes for 100 samples)

📁 Processing CHORDS_DATA...
  Analyzing sample_100.pickle...
  100%|██████████| 100/100 [01:23<00:00,  1.20it/s]

📁 Processing KILO_CHORDS_DATA...
  Loading sequences: 100%|██████████| 97/97 [00:01<00:00]

📁 Processing SIGNATURES_DATA...
  Loading signatures: 100%|██████████| 98/98 [00:00<00:00]

✅ Database built successfully in 142.3 seconds!
   Database size: 245.7 KB

🔍 Validating database...
   Tables found: progressions, kilo_sequences, signatures
   • progressions: 87 records
   • kilo_sequences: 97 records
   • signatures: 98 records
   • Linked records (progressions ↔ kilo): 81
   • Linked records (progressions ↔ signatures): 83

📄 Sample progression:
   Hash ID: f1b24a0b1f5255f95e0b69c2cc3949f4
   Total events: 3427
   Chord events: 142
   First 3 chords:
     • G dominant seventh chord (root: G)
     • C major triad (root: C)
     • D dominant seventh chord (root: D)

✅ Database validation passed!

📈 Performance Estimation:
   Test build time: 142.3s for 100 samples
   Time per sample: 1.423s

🔮 Full build estimation (assuming ~180,000 samples):
   Estimated time: 4267.4 minutes (71.1 hours)
   Estimated cost: ¥1635 (at ¥23/hour)

⚠️  Note: Vertex AI e2-standard-4 will be faster due to better CPU/RAM
```

#### 3. クエリテスト

ローカルビルド成功後、自動的にクエリ例がテストされます:

```python
# 例1: コード進行検索
SELECT hash_id, progression 
FROM progressions 
WHERE progression LIKE '%"chord": "C major triad"%'

# 例2: イベント数フィルタ
SELECT COUNT(*) 
FROM progressions 
WHERE total_events > 1000

# 例3: KILO sequence長さ分布
SELECT sequence_length, COUNT(*) 
FROM kilo_sequences
GROUP BY sequence_length
```

---

## ☁️ Vertex AI 実行

### 準備確認

**✅ チェックリスト** ([完全版](docs/LAMDA_EXECUTION_CHECKLIST.md)):

- [ ] GCS データアップロード完了 (`gs://otobon/lamda/`)
- [ ] Vertex AI インスタンス作成 (`shimogami88-Default`)
- [ ] ローカルテスト成功
- [ ] コード確認完了
- [ ] 予算承認 (¥30-50)

### 実行方法

#### オプション A: Pythonスクリプト

```bash
# Vertex AI Colab Enterprise のターミナルで実行
python scripts/build_lamda_unified_db.py
```

#### オプション B: Notebookガイド (推奨)

1. **Vertex AI Colab Enterpriseを開く**
   ```
   https://console.cloud.google.com/vertex-ai/colab
   ```

2. **新しいNotebookを作成**

3. **ガイドをコピー**
   ```python
   # docs/vertex_ai_lamda_unified_guide.py の内容をコピー
   ```

4. **Cell 1-7を順番に実行**

   **Cell 1**: 環境確認 + GCS認証
   ```python
   # Python version, working directory
   # GCS access validation
   ```

   **Cell 2**: リポジトリ + 依存関係
   ```python
   # git clone composer4
   # pip install music21 numpy tqdm
   ```

   **Cell 3**: CHORDS_DATA ダウンロード
   ```python
   # Download 575MB (compressed)
   # Extract to 15GB
   # Time: 5-10 minutes
   ```

   **Cell 4**: KILO, SIGNATURES, TOTALS
   ```python
   # Download 602MB + 290MB + 33MB
   # Time: 2-3 minutes
   ```

   **Cell 5**: データベース構築 ⏱️ **メイン処理**
   ```python
   # Process all data sources
   # Time: 60-90 minutes
   # Progress bars with tqdm
   ```

   **Cell 6**: GCSにアップロード
   ```python
   # Upload lamda_unified.db
   # Time: 1-2 minutes
   ```

   **Cell 7**: サマリー表示
   ```python
   # Statistics
   # Cost report
   # Next steps
   ```

### モニタリング

実行中は以下を確認:

- **進捗バー**: tqdmによるリアルタイム表示
- **ログ出力**: 各ステップの完了メッセージ
- **エラー**: 赤字で表示される例外

### 完了確認

```bash
# GCSに出力されたか確認
gsutil ls -lh gs://otobon/lamda/lamda_unified.db

# ダウンロードしてローカル確認
gsutil cp gs://otobon/lamda/lamda_unified.db ./
sqlite3 lamda_unified.db "SELECT COUNT(*) FROM progressions;"
```

---

## 💡 データベース活用

### Pythonからの利用

```python
import sqlite3
from lamda_unified_analyzer import LAMDaUnifiedAnalyzer

# データベース接続
conn = sqlite3.connect('lamda_unified.db')
cursor = conn.cursor()

# 例1: コード進行検索
cursor.execute("""
    SELECT hash_id, progression 
    FROM progressions 
    WHERE progression LIKE ?
    LIMIT 10
""", ('%C major%',))

results = cursor.fetchall()

# 例2: hash_idから全情報取得
hash_id = results[0][0]

# CHORDS_DATAから詳細
cursor.execute("SELECT * FROM progressions WHERE hash_id = ?", (hash_id,))
progression_detail = cursor.fetchone()

# KILO_CHORDSから整数シーケンス
cursor.execute("SELECT sequence FROM kilo_sequences WHERE hash_id = ?", 
               (hash_id,))
kilo_seq = cursor.fetchone()

# SIGNATURESから特徴量
cursor.execute("SELECT pitch_distribution FROM signatures WHERE hash_id = ?",
               (hash_id,))
signature = cursor.fetchone()

conn.close()
```

### コマンドラインからの利用

```bash
# 統計情報
sqlite3 lamda_unified.db "
    SELECT 
        COUNT(*) as total,
        AVG(total_events) as avg_events,
        AVG(chord_events) as avg_chords
    FROM progressions
"

# トップ10頻出コード
sqlite3 lamda_unified.db "
    SELECT progression, COUNT(*) as count
    FROM progressions
    GROUP BY progression
    ORDER BY count DESC
    LIMIT 10
"
```

### Recommendation System例

```python
def recommend_similar_progressions(user_progression, top_k=10):
    """類似コード進行を推薦"""
    
    conn = sqlite3.connect('lamda_unified.db')
    cursor = conn.cursor()
    
    # 1. ユーザー進行の特徴量計算
    user_sig = extract_signature(user_progression)
    
    # 2. 全SIGNATURESと比較
    cursor.execute("SELECT hash_id, pitch_distribution FROM signatures")
    
    similarities = []
    for hash_id, sig_str in cursor.fetchall():
        sig = parse_signature(sig_str)
        score = cosine_similarity(user_sig, sig)
        similarities.append((hash_id, score))
    
    # 3. トップK取得
    top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    # 4. 詳細情報取得
    recommendations = []
    for hash_id, score in top_similar:
        cursor.execute("""
            SELECT p.progression, k.sequence 
            FROM progressions p
            JOIN kilo_sequences k ON p.hash_id = k.hash_id
            WHERE p.hash_id = ?
        """, (hash_id,))
        
        result = cursor.fetchone()
        if result:
            recommendations.append({
                'hash_id': hash_id,
                'score': score,
                'progression': result[0],
                'kilo_sequence': result[1]
            })
    
    conn.close()
    return recommendations
```

---

## 🔧 トラブルシューティング

### 問題: GCS Permission Denied

```bash
# 解決策: 認証
gcloud auth application-default login
```

### 問題: Memory Error (CHORDS_DATA処理中)

```python
# 解決策: バッチサイズ削減
# lamda_unified_analyzer.py の build_unified_database() を修正

# 変更前
for pickle_path in chords_files:
    progressions = self.analyze_chords_file(pickle_path)
    # 全て処理

# 変更後
for pickle_path in chords_files:
    progressions = self.analyze_chords_file(pickle_path)
    
    # 10ファイルごとにコミット
    if len(processed_files) % 10 == 0:
        conn.commit()
```

### 問題: Database Locked

```bash
# 解決策: 全接続を閉じる
rm lamda_unified.db
# 再実行
python scripts/test_local_build.py
```

### 問題: ローカルテストが遅い

**原因**: ディスクI/O制約

**解決策**:
1. SSD使用を確認
2. サンプルサイズ削減 (100 → 50)
3. Vertex AIで実行 (より高速なストレージ)

### 問題: Vertex AI実行が途中で停止

**確認事項**:
1. インスタンスタイプ: e2-standard-4以上
2. ディスク容量: 20GB以上
3. ネットワーク接続
4. GCS権限

**ログ確認**:
```python
# Cell実行中のエラーログを確認
import traceback
traceback.print_exc()
```

---

## 📚 ドキュメント一覧

| ドキュメント | 内容 | 対象 |
|-------------|------|------|
| **[このファイル](docs/LAMDA_README.md)** | 総合ガイド | 全ユーザー |
| [Architecture Guide](docs/LAMDA_UNIFIED_ARCHITECTURE.md) | 詳細設計 | 開発者 |
| [Execution Checklist](docs/LAMDA_EXECUTION_CHECKLIST.md) | 実行手順 | 実行担当者 |
| [Vertex AI Guide](docs/vertex_ai_lamda_unified_guide.py) | Notebook実行 | Vertex AIユーザー |
| [Main README](../README.md) | プロジェクト全体 | 全ユーザー |

---

## 🎯 Next Steps

1. **✅ ローカルテスト完了**
   ```bash
   python scripts/create_test_sample.py
   python scripts/test_local_build.py
   ```

2. **☁️ Vertex AI実行**
   - Checklist確認: [LAMDA_EXECUTION_CHECKLIST.md](LAMDA_EXECUTION_CHECKLIST.md)
   - Notebook実行: `vertex_ai_lamda_unified_guide.py`

3. **💡 統合活用**
   - Recommendation system構築
   - Search interface開発
   - Style transfer実装

---

## 🙏 Credits

- **LAMDa Dataset**: [Los Angeles MIDI Dataset](https://github.com/asigalov61/Los-Angeles-MIDI-Dataset) by asigalov61
- **TMIDIX Library**: LAMDa CODE folder
- **Architecture Design**: 連邦制データモデル ("離れ小島ではなく、連邦")

---

## 📄 License

This project follows the MIT License. See [LICENSE](../LICENSE) for details.

LAMDa Dataset has its own license. Please refer to the [original repository](https://github.com/asigalov61/Los-Angeles-MIDI-Dataset).

---

**Questions or Issues?** Open an issue on GitHub or check the [troubleshooting section](#トラブルシューティング) above.

**Ready to start?** Go to [Quick Start](#クイックスタート)! 🚀
