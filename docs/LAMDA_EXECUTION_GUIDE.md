# LAMDA (Los-Angeles-MIDI) 楽器別クリーニング - 実行手順

## 🎯 概要

**LAMDA (Los-Angeles-MIDI Dataset)** を楽器別に分離してクリーニングします。

### データセット特性
- **総ファイル数**: 約40万件 (404,714個)
- **タイプ**: マルチトラック楽曲データセット
- **含まれる楽器**: piano, strings, guitar, bass, drums等
- **用途**: コード進行パターン、グルーヴ抽出、ヒューマナイズパラメータの学習

### 処理対象楽器
- ✅ **LAMDA_PIANO** - ピアノパート
- ✅ **LAMDA_STRINGS** - ストリングス（violin, viola, cello等）
- ✅ **LAMDA_GUITAR** - ギター（ストラム/アルペジオ）
- ✅ **LAMDA_BASS** - ベース（ルート + グルーヴ）
- ⚠️ **LAMDA_DRUMS** - ドラム（大容量のため明示指定時のみ）

---

## 📋 実行手順

### ステップ1: データセット確認

```bash
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3
./scripts/check_lamidi_dataset.sh
```

**確認項目:**
- MIDIファイル総数
- ディレクトリ構造
- アクセス権限

---

### ステップ2: ドライラン（推奨）

実際の実行コマンドを確認します：

```bash
# 全楽器（piano/strings/guitar/bass）のコマンド確認
./scripts/run_lamda_full.sh --dry-run

# 特定楽器のみ確認
./scripts/run_lamda_full.sh --dry-run piano
./scripts/run_lamda_full.sh --dry-run guitar bass
```

**確認内容:**
- 入力ディレクトリ
- 出力ディレクトリ
- Pickleメタデータ保存先
- 並列度
- シャードサイズ

---

### ステップ3: クリーニング実行

#### 🔹 推奨: 4楽器を段階的に実行

大規模データセットのため、楽器別に順次実行を推奨：

```bash
# 1. Piano（最優先）
./scripts/run_lamda_full.sh piano

# 2. Strings
./scripts/run_lamda_full.sh strings

# 3. Guitar
./scripts/run_lamda_full.sh guitar

# 4. Bass
./scripts/run_lamda_full.sh bass
```

#### 🔸 オプション: 一括実行

```bash
# Piano/Strings/Guitar/Bassを一括実行（drumsを除く）
./scripts/run_lamda_full.sh

# 全楽器（drumsを含む）
./scripts/run_lamda_full.sh piano strings guitar bass drums
```

---

### ステップ4: 進捗モニタリング

**別ターミナルで実行:**

```bash
./scripts/monitor_lamda.sh
```

**表示内容:**
```
📊 LAMDA Multi-Instrument Cleaning Monitor
================================================
Time: 2025-10-18 12:30:45

🎹 PIANO:
   Total: 15234  |  ✅ Cleaned: 12456 (81.8%)  |  🗑️  Quarantined: 2778
   📦 Pickle Index: ✅ READY

🎹 STRINGS:
   Total: 8432  |  ✅ Cleaned: 7123 (84.5%)  |  🗑️  Quarantined: 1309
   📦 Pickle Shards: 3 (Index pending)

🎹 GUITAR:
   ⏳ Not started or no files processed

🎹 BASS:
   ⏳ Not started or no files processed

🎹 DRUMS:
   ⏳ Not started or no files processed
```

---

## 📂 出力構造

### クリーニング済みファイル
```
data/
├── cleaned/
│   ├── lamda_piano/      # Piano クリーニング済み
│   ├── lamda_strings/    # Strings クリーニング済み
│   ├── lamda_guitar/     # Guitar クリーニング済み
│   ├── lamda_bass/       # Bass クリーニング済み
│   └── lamda_drums/      # Drums クリーニング済み
```

### 隔離ファイル
```
data/
└── quarantine/
    ├── lamda_piano/      # Piano 品質不適合
    ├── lamda_strings/    # Strings 品質不適合
    ├── lamda_guitar/     # Guitar 品質不適合
    ├── lamda_bass/       # Bass 品質不適合
    └── lamda_drums/      # Drums 品質不適合
```

### Pickleメタデータ
```
data/
├── lamda_piano_metadata/
│   ├── piano_shard_0000.pickle
│   ├── piano_shard_0001.pickle
│   └── piano_metadata_v2.pickle  # インデックス
├── lamda_strings_metadata/
│   └── ...
└── ...
```

### ログ
```
logs/
├── clean_LAMDA_PIANO_piano_20251018_123045.log
├── clean_LAMDA_STRINGS_strings_20251018_134512.log
└── ...
```

---

## 🎵 Stage 1 処理内容（楽器別）

### Piano (`cleaners/piano.py`)
- ✅ Piano楽器チェック（GM Program: 0-7）
- ✅ 音域チェック（21-108推奨）
- ✅ ポリフォニーチェック
- ✅ ダイナミクスチェック（velocity分布）
- ✅ ボイシング品質（和音構造）

### Strings (`cleaners/strings.py`)
- ✅ Strings楽器チェック（GM: 40-51）
- ✅ Pad vs Ostinato判定
- ✅ 持続音処理
- ✅ 音域チェック
- ✅ レガート/スタッカート判定

### Guitar (`cleaners/guitar.py`)
- ✅ Guitar楽器チェック（GM: 24-31）
- ✅ ストラム vs アルペジオ判定
- ✅ 同時性インデックス（simultaneity_index）
- ✅ コード構造チェック（3音以上）
- ⚠️ アルペジオ偏重の抑制（品質ゲート）

### Bass (`cleaners/bass.py`)
- ✅ Bass楽器チェック（GM: 32-39）
- ✅ on-beat比率チェック
- ✅ キック同期率（オプション）
- ✅ スケール適合性
- ✅ 跳躍量チェック

### Drums (`cleaners/drums.py`)
- ✅ Drum楽器チェック（Ch.10 または GM: 128）
- ✅ キック/スネア/ハイハット識別
- ✅ on-beat比率（backbeat等）
- ✅ ゴーストノート率
- ✅ notes_per_bar範囲チェック

---

## ⚙️ 設定

### 並列度調整

`scripts/run_dataset_full.sh` の該当行を編集：

```bash
# デフォルト: 8並列
LAMDA_PIANO|piano|...|...|...|...|4000|8|lamda-piano-v1

# 4並列に変更
LAMDA_PIANO|piano|...|...|...|...|4000|4|lamda-piano-v1
```

### シャードサイズ調整

```bash
# デフォルト: 4000ファイル/シャード
LAMDA_PIANO|piano|...|...|...|...|4000|8|lamda-piano-v1

# 5000に変更（メモリ余裕がある場合）
LAMDA_PIANO|piano|...|...|...|...|5000|8|lamda-piano-v1
```

---

## 🔧 トラブルシューティング

### SSD停止・中断からの再開

`--resume` オプションは既にデフォルトで有効です：

```bash
# 自動的に既存シャードから再開
./scripts/run_lamda_full.sh piano
```

### エラーログ確認

```bash
# 最新ログを表示
ls -lt logs/clean_LAMDA_*.log | head -1

# ログの最後50行を確認
tail -50 logs/clean_LAMDA_PIANO_piano_*.log
```

### 進捗が進まない場合

1. **ファイルアクセス確認**
   ```bash
   ls -la data/Los-Angeles-MIDI/MIDIs/ | head
   ```

2. **プロセス確認**
   ```bash
   ps aux | grep clean_midi.py
   ```

3. **ディスク容量確認**
   ```bash
   df -h /Volumes/SSD-SCTU3A
   ```

---

## 📊 Stage 2: メタデータ抽出

Stage 1完了後、各楽器のPickleインデックスから特徴量を抽出：

```bash
# Piano
python scripts/lamda_stage2_extractor.py \
  --metadata-index data/lamda_piano_metadata/piano_metadata_v2.pickle \
  --output data/lamda_piano_stage2_scored.jsonl

# Strings
python scripts/lamda_stage2_extractor.py \
  --metadata-index data/lamda_strings_metadata/strings_metadata_v2.pickle \
  --output data/lamda_strings_stage2_scored.jsonl

# Guitar
python scripts/lamda_stage2_extractor.py \
  --metadata-index data/lamda_guitar_metadata/guitar_metadata_v2.pickle \
  --output data/lamda_guitar_stage2_scored.jsonl

# Bass
python scripts/lamda_stage2_extractor.py \
  --metadata-index data/lamda_bass_metadata/bass_metadata_v2.pickle \
  --output data/lamda_bass_stage2_scored.jsonl
```

---

## 📚 参考: 設計方針

### なぜ楽器別に分離？

1. **品質ゲートの最適化**: 楽器ごとに異なる品質基準
2. **BPM層化サンプリング**: 楽器別の特性に合わせた抽出
3. **パターン抽出**: 後段のrhythm_library.yaml等への統合
4. **偏り是正**: Guitar のアルペジオ偏重等を統計的に補正

### 公式LAMDAスクリプトを使わない理由

- ❌ Colab + tegridy-tools 依存
- ❌ "ファイル丸ごと" 志向（パターン抽出に不向き）
- ❌ 品質ゲートが弱い
- ✅ 自作スクリプト: pattern-first / quality-first 設計

### 出力形式: "設計図" として保存

- **Drums**: 拍相対グリッド + velocity_bin
- **Chords**: ローマ数字列 + キー正規化
- **Bass/Piano/Guitar**: 度数表現 + 拍内配置
- **Humanize**: 統計パラメータ（μ/σ）

---

## ✅ チェックリスト

実行前:
- [ ] MIDIファイルが `/data/Los-Angeles-MIDI/MIDIs` に存在
- [ ] `scripts/cleaners/` に楽器別クリーナーが配置済み
- [ ] ドライランで設定を確認済み

実行中:
- [ ] `monitor_lamda.sh` で進捗を監視
- [ ] ログファイルでエラーがないか確認

実行後:
- [ ] 各楽器の Pickle Index が生成されている
- [ ] クリーニング成功率が適切（目安: 60-85%）
- [ ] Stage 2 の準備が整っている

---

## 🎯 次のステップ

1. ✅ **Stage 1完了**: 楽器別クリーニング + Pickleインデックス生成
2. ⏭️ **Stage 2実行**: 特徴量抽出 + スコアリング
3. ⏭️ **パターン抽出**: BPM層化サンプリング + 品質ゲート
4. ⏭️ **統合**: rhythm_library.yaml / chordmap.yaml への追加
5. ⏭️ **検証**: Suno完成曲との組み合わせテスト

詳細は `docs/LAMIDI_CLEANING_GUIDE.md` をご参照ください。
