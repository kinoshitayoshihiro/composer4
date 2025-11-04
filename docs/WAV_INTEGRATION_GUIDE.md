# WAV Acoustic Features Integration Guide

E-GMDやその他の音声データを **MIDI特徴と統合** してハイブリッド学習を実現する手順。

---

## 📂 ファイル構成

```
scripts/
├── clean_wav_stage1.py       # WAV Stage1クリーナ（重複除去・標準化・品質タグ付け）
├── run_wav_stage1.sh          # Stage1実行スクリプト
├── merge_wav_features.py      # Stage2 MIDI特徴 + WAV音響特徴マージ
└── rhythm_stage2_extractor.py # Stage2 MIDIリズム特徴抽出（既存）

configs/
└── wav_stage1.yaml            # WAV Stage1設定

output/
├── wav_stage1/                # WAV Stage1出力
│   ├── index/
│   │   ├── wav_index.pkl      # 音響メタデータ（MD5, dur_s, rms, peak, onset_rate_hz, clip_ratio）
│   │   └── wav_index.csv
│   ├── cleaned/               # クリーニング済みWAV（任意: --write-audio）
│   └── wav_cleaning_summary.json
└── rhythm_ai/
    └── egmd_stage2/
        ├── rhythm_features.parquet              # MIDI特徴のみ
        └── rhythm_features_with_wav.parquet     # MIDI + WAV統合
```

---

## 🚀 実行フロー

### 1. WAV Stage1: クリーニング＆インデックス化

```bash
# E-GMD audio rootを指定
bash scripts/run_wav_stage1.sh /path/to/e-gmd/audio_root output/wav_stage1

# または直接実行（詳細オプション）
python scripts/clean_wav_stage1.py \
  --input /path/to/e-gmd/audio_root \
  --out-dir output/wav_stage1 \
  --sr 44100 \
  --peak 0.98 \
  --trim-db -50 \
  --min-dur 1.5 \
  --max-dur 120 \
  --write-audio  # WAV書き出し（任意）
```

**出力**:
- `output/wav_stage1/index/wav_index.pkl`: 音響メタデータ
- `output/wav_stage1/wav_cleaning_summary.json`: 統計サマリー

**Stage1処理内容**:
- ✅ MD5重複除去（生バイト完全一致）
- ✅ 44.1kHz リサンプル + モノラル化
- ✅ ピーク正規化 (0.98)
- ✅ 前後サイレント自動トリム (-50dB閾値)
- ✅ 品質タグ付け: `too_short`, `too_long`, `clipping`, `too_quiet`
- ✅ 音響統計: `dur_s`, `rms`, `peak`, `onset_rate_hz`, `clip_ratio`

---

### 2. MIDI Stage2実行（既存）

```bash
# E-GMD MIDI Stage2（既に完了）
bash scripts/run_rhythm_stage2_egmd.sh

# 出力: output/rhythm_ai/egmd_stage2/rhythm_features.parquet
```

---

### 3. WAV特徴マージ

```bash
python scripts/merge_wav_features.py \
  --rhythm-parquet output/rhythm_ai/egmd_stage2/rhythm_features.parquet \
  --wav-index output/wav_stage1/index/wav_index.pkl \
  --output output/rhythm_ai/egmd_stage2/rhythm_features_with_wav.parquet
```

**追加カラム**:
- `wav_path`: マッチしたWAVファイルパス
- `wav_duration_s`: WAV長 (秒)
- `wav_onset_rate_hz`: オンセット密度 (Hz) - **マイクロタイミング代理**
- `wav_clip_ratio`: クリッピング率 (0-1) - **ダイナミクス異常検出**
- `wav_rms`: RMS音量 - **演奏強度**
- `wav_peak`: ピーク音量 - **ダイナミックレンジ**

**マッチングロジック**:
- E-GMD命名規則対応:
  - MIDI: `drummer1/session1/1_funk-groove_120_beat_4-4.midi`
  - WAV:  `drummer1/session1/audio_mic/1_funk-groove_120_beat_4-4.wav`
- ファイル名ステム一致でペアリング
- 未マッチは `0.0` / `''` で埋める

---

## 📊 学習への統合

### merge_rhythm_datasets.sh 修正例

```bash
# WAV統合版Parquetを使用
df_egmd = pd.read_parquet('output/rhythm_ai/egmd_stage2/rhythm_features_with_wav.parquet')

# 特徴量リスト拡張
feature_cols = [
    # MIDI特徴（既存）
    'tempo_bpm', 'swing_pct', 'backbeat_strength',
    'kick_downbeat_rate', 'snare_backbeat_rate', 'hat_density',
    'onset_deviation_mean', 'onset_deviation_std',
    'density_mean', 'density_std', 'density_min', 'density_max',
    'kick_onset_count', 'snare_onset_count', 'hat_onset_count',
    
    # WAV音響特徴（NEW）
    'wav_onset_rate_hz',   # マイクロタイミング代理
    'wav_clip_ratio',       # ダイナミクス異常
    'wav_rms',              # 演奏強度
    'wav_peak',             # ダイナミックレンジ
]
```

---

## 🎯 期待効果

### MIDI特徴のみ (現状)
- リズムパターン（Kick/Snare/Hat配置）
- Swing量・Backbeat強度
- オンセット偏差（MIDI内部タイミング）

### MIDI + WAV統合 (提案)
- **マイクロタイミング**: `wav_onset_rate_hz` で実演奏のリアルタイム揺らぎ捕捉
- **ダイナミクス**: `wav_rms` / `wav_peak` で生演奏の強弱変化を学習
- **品質フィルタ**: `wav_clip_ratio` で過剰クリッピング除外

**学習モデル向上**:
- XGBoost / LogReg の **数値特徴** として直接使用可能
- Swing予測精度向上（MIDIだけでは捉えにくい"揺らぎ"を補完）
- ダイナミクス生成の自然さ向上

---

## ⚠️ 注意事項

### 依存ライブラリ
- `soundfile`: WAV読み込み
- `numpy`: 音響統計計算
- `scipy`: 高品質リサンプル（任意）

インストール:
```bash
pip3 install --break-system-packages soundfile numpy scipy
```

### MD5重複除去
- **完全一致のみ除外** (バイトレベル)
- **"ほぼ同一"** (例: 異なるマイク録音) は **残す**
  → 学習時に `sample_weight` で重み付け制御推奨

### E-GMD特有の問題
- 同一演奏の複数マイク録音が存在
- `wav_index.pkl` で `md5_raw` が同一 → 自動除外済み

---

## 📝 次のステップ

1. **E-GMD audio rootパス確認**:
   ```bash
   find /Volumes/SSD-SCTU3A -type d -name "e-gmd*" 2>/dev/null
   ```

2. **WAV Stage1実行**:
   ```bash
   bash scripts/run_wav_stage1.sh /path/to/e-gmd/audio output/wav_stage1
   ```

3. **WAV特徴マージ**:
   ```bash
   python scripts/merge_wav_features.py \
     --rhythm-parquet output/rhythm_ai/egmd_stage2/rhythm_features.parquet \
     --wav-index output/wav_stage1/index/wav_index.pkl \
     --output output/rhythm_ai/egmd_stage2/rhythm_features_with_wav.parquet
   ```

4. **統合学習**:
   - `merge_rhythm_datasets.sh` 修正
   - WAV特徴カラム追加
   - XGBoost/LogReg再学習

---

## 🔧 トラブルシューティング

### Q: WAVファイルが見つからない
```bash
# E-GMDは `.midi` のみ？
find /path/to/e-gmd -name "*.wav" -o -name "*.flac" | head -5
```

### Q: マッチング率が低い
- E-GMD命名規則確認
- `merge_wav_features.py` の `match_midi_to_wav()` ロジック調整
- デバッグ: `--output` で中間CSVを確認

### Q: メモリ不足
- `clean_wav_stage1.py`: `--write-audio` 無効化でメモリ節約
- `merge_wav_features.py`: バッチ処理に分割

---

**関連ファイル**:
- `scripts/clean_egmd_simple.py` (E-GMD MIDI Stage1)
- `scripts/run_rhythm_stage2_egmd.sh` (E-GMD MIDI Stage2)
- `scripts/merge_rhythm_datasets.sh` (3データセット統合+ML学習)
