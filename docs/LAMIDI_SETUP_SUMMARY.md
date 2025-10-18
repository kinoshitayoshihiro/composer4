# Los-Angeles-MIDI クリーニングセットアップ完了

## 作成・編集したファイル

### 1. メインスクリプト
- **`scripts/run_lamidi_full.sh`** (新規作成)
  - Los-Angeles-MIDI専用クリーニングラッパー
  - POP909と同じ構造で共通ランナーを呼び出し

### 2. データセット設定
- **`scripts/run_dataset_full.sh`** (編集)
  - LAMIDIデータセット設定を追加:
    ```
    LAMIDI|piano|data/Los-Angeles-MIDI/MIDIs|data/cleaned/lamidi|data/quarantine/lamidi|data/lamidi_metadata|5000|8|lamidi-v1
    ```

### 3. 進捗モニター
- **`scripts/monitor_lamidi.sh`** (新規作成)
  - リアルタイム進捗表示
  - クリーニング済みファイル数
  - 隔離ファイル数
  - Pickleインデックス状態
  - ログ表示

### 4. データセット情報確認
- **`scripts/check_lamidi_dataset.sh`** (新規作成)
  - MIDIファイル数を自動カウント
  - 実行前の事前確認用

### 5. ドキュメント
- **`docs/LAMIDI_CLEANING_GUIDE.md`** (新規作成)
  - 詳細な使い方ガイド
  - トラブルシューティング
  - Stage 2への移行方法

## クリーニングシステムの理解

### Stage 1: MIDIクリーニング (`scripts/clean_midi.py`)

#### 処理フロー
1. **MIDIファイル読み込み**
   - シンボリックリンク対応
   - 決定論的ファイル列挙

2. **共通クリーニング** (`cleaners/common.py`)
   - テンポ正規化
   - トラック数チェック
   - ノート数チェック
   - 空トラック削除

3. **楽器別クリーニング**
   - Piano: `cleaners/piano.py`
   - Guitar: `cleaners/guitar.py`
   - Bass: `cleaners/bass.py`
   - Strings: `cleaners/strings.py`
   - Drums: `cleaners/drums.py`

4. **品質判定**
   - 成功 → `data/cleaned/<dataset>/` へ保存
   - 失敗 → `data/quarantine/<dataset>/` へ隔離

5. **メタデータ生成**
   - LAMDAメタデータ抽出
   - Pickleシャードへ直接書き込み
   - `.meta.json`は生成しない (Pickle-Direct v2運用)

#### 主要オプション
- `--in`: 入力ディレクトリ
- `--out`: クリーニング済み出力先
- `--quarantine`: 隔離ファイル出力先
- `--instrument`: 楽器タイプ
- `--pickle-out`: Pickleメタデータ出力先
- `--shard-size`: シャードサイズ (推奨: 5000)
- `--resume`: 中断から再開
- `--emit-meta-json off`: JSONメタデータを生成しない
- `--jobs`: 並列度
- `--seed`: 乱数シード

### Stage 2: メタデータ抽出 (`scripts/lamda_stage2_extractor.py`)

Stage 1完了後に実行:
- Pickleインデックスから特徴量抽出
- スコアリングとフィルタリング
- JSONL形式で出力

## 次のステップ

### 1. データセット情報確認
```bash
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3
./scripts/check_lamidi_dataset.sh
```

### 2. EXPECTED_TOTAL更新
出力されたファイル数を `scripts/monitor_lamidi.sh` に設定

### 3. ドライラン
```bash
./scripts/run_lamidi_full.sh --dry-run
```

### 4. 実行
```bash
# メインターミナル
./scripts/run_lamidi_full.sh

# 別ターミナル（進捗監視）
./scripts/monitor_lamidi.sh
```

### 5. Stage 2実行
```bash
python scripts/lamda_stage2_extractor.py \
  --metadata-index data/lamidi_metadata/piano_metadata_v2.pickle \
  --output data/lamidi_stage2_scored.jsonl
```

## 技術的特徴

### Pickle-Direct v2運用
- `.meta.json`を生成せず、Pickleシャードに直接書き込み
- ディスクI/O削減
- 処理速度向上

### レジューム機能
- 既存シャードから自動再開
- SSD停止対策
- `--resume`フラグで有効化（デフォルトON）

### 並列処理
- ProcessPoolExecutorによる並列化
- デフォルト8並列
- CPU使用率に応じて調整可能

### 決定論的処理
- ファイルハッシュによる再現性確保
- 固定シードによる乱数制御
- プロベナンス情報記録

## 既存データセットの状態

✅ **POP909**: クリーニング完了、Stage2完了
✅ **SLAKH**: クリーニング完了、Stage2完了
✅ **loops**: クリーニング完了、Stage2完了
🔄 **Los-Angeles-MIDI**: これから実行

## 参考資料

### スクリプト関連
- `scripts/run_dataset_full.sh`: 共通ランナー
- `scripts/clean_midi.py`: メインクリーニングスクリプト
- `cleaners/`: 楽器別クリーナーモジュール

### モニタリング
- `scripts/monitor_pop909.sh`: POP909モニター（参考）
- `scripts/status_pop909.sh`: ステータス確認（参考）

### ログ
- `logs/clean_LAMIDI_*.log`: クリーニングログ
- `logs/clean_POP909_*.log`: POP909ログ（参考）
