# 🎹 Piano Production Guide

**最終更新**: 2025-10-30  
**対象**: Salamander Grand Piano V3 (SFZ), SampleTank 4 Piano, Miroslav Philharmonik 2

---

## 📋 目次

1. [事前準備](#事前準備)
2. [単曲受け入れテスト](#単曲受け入れテスト)
3. [9曲バッチ処理](#9曲バッチ処理)
4. [Audio KPI集計](#audio-kpi集計)
5. [トラブルシューティング](#トラブルシューティング)
6. [運用TIPS](#運用tips)

---

## 事前準備

### 1. VST/SFZプラグインインストール

#### Salamander SFZ（推奨・無料）

```bash
# 1. sfizzダウンロード（無料）
# https://sfz.tools/sfizz/

# 2. Salamander Grand Piano V3配置
# パスに日本語/空白を避ける
# 例: /Users/username/SFZ/SalamanderGrandPianoV3_48khz24bit/

# 3. VST3パス確認
find /Library/Audio/Plug-Ins/VST3 -name "*sfizz*"
find ~/Library/Audio/Plug-Ins/VST3 -name "*sfizz*"

# 期待パス: /Library/Audio/Plug-Ins/VST3/sfizz.vst3
```

#### SampleTank 4

```bash
# 1. ST4インストール
# 2. Settings → Rebuild instrument database
# 3. ピアノ音色ロード確認

# VST3パス確認
find /Library/Audio/Plug-Ins/VST3 -name "*SampleTank*"
```

#### Miroslav Philharmonik 2

```bash
# VST3パス確認
find /Library/Audio/Plug-Ins/VST3 -name "*Miroslav*" -o -name "*Philharmonik*"
```

### 2. 環境変数設定（オプション）

```bash
# KS先行時間（デフォルト: 80ms）
export VIOPTT_KS_ADVANCE_MS=80

# CCスムージング（デフォルト: 0.125拍 = 1/8拍）
export VIOPTT_CC_SLEW_BEATS=0.125

# サンプルレート（自動検出あり、明示する場合）
export ENGINE_SR=44100  # または 48000
```

---

## 単曲受け入れテスト

### Step 1: 制御MIDI生成テスト

```bash
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3

# Salamander Piano
python3 scripts/vioptt_render_stub.py \
  --hints song_packages/test_project/test_song/articulation_hints.json \
  --mapping configs/vioptt_mapping.yaml \
  --instrument piano_sfz_salamander \
  --output /tmp/test_piano_controls.mid \
  --tempo-bpm 120
```

**期待結果**:
```
✅ Control MIDI saved: /tmp/test_piano_controls.mid
   Total hints: 32
   Keyswitches: 1
   CC mappings: 4
   KS advance: 80ms (76 ticks)
   CC slew: 62ms (60 ticks)  # BPM120で約62ms（0.125拍）
```

**確認ポイント**:
- CC11（Expression）: 40-127
- CC64（Sustain Pedal）: 0 or 127（threshold=0.55、hysteresis=0.05）
- CC1（Dynamics）: 0-127
- CC91（Reverb Send）: 10-50

### Step 2: WAVレンダリングテスト（VST到着後）

```bash
# Salamander SFZ（44.1kHz）
VIOPTT_KS_ADVANCE_MS=80 VIOPTT_CC_SLEW_BEATS=0.125 \
bash scripts/run_vioptt_pipeline.sh \
  --song-dir song_packages/test_project/test_song \
  --instrument piano_sfz_salamander \
  --vst-path "/Library/Audio/Plug-Ins/VST3/sfizz.vst3" \
  --tempo-bpm 120
```

**期待結果**:
```
[00:00:00] 🎻 VioPTT WAV Pipeline
[00:00:00] ============================================================
[00:00:00] Song directory: song_packages/test_project/test_song
[00:00:00] Instrument:     piano_sfz_salamander
[00:00:00] VST path:       /Library/Audio/Plug-Ins/VST3/sfizz.vst3
[00:00:00] Tempo:          120 BPM
[00:00:00] 
[00:00:00] 📊 Step 1/4: articulation_hints.json already exists (skipping)
[00:00:00] 🎹 Step 2/4: Generating control MIDI...
✅ Control MIDI saved: piano_sfz_salamander_controls.mid
[00:00:01] 🔗 Step 3/4: Merging MIDI tracks...
✅ MIDI merged: piano_sfz_salamander_merged.mid
[00:00:01] 🎧 Step 4/4: Rendering WAV with DAWDreamer...
✅ WAV rendering completed!
```

**確認ポイント**:
- `piano_sfz_salamander_controls.mid` 生成（制御MIDI）
- `piano_sfz_salamander_merged.mid` 生成（統合MIDI）
- `piano_sfz_salamander_rendered.wav` 生成（WAV出力）
- 無音区間なし、クリッピングなし

### Step 3: Audio KPI検証

```bash
# ピアノ専用プロファイル使用
python3 scripts/validate_audio_quality.py \
  --wav song_packages/test_project/test_song/piano_sfz_salamander_rendered.wav \
  --midi song_packages/test_project/test_song/piano_sfz_salamander_merged.mid \
  --gate configs/audio_gate_prod.yaml \
  --profile piano_kpi \
  --out-json song_packages/test_project/test_song/audio_kpi_piano.json
```

**期待KPI（ピアノプロファイル）**:
- `render_rtf` ≤ 0.50（2×速以上）
- `clip_ratio` ≤ 0.001（0.1%以下）
- `integrated_lufs`: -22 〜 -16 LUFS
- `crest_factor_db`: 10 〜 20 dB
- `latency_ms_onset` ≤ 120ms
- `missing_onset_rate` ≤ 0.03（3%以下）

**判定**:
```json
{
  "overall_status": "PASS",
  "render_rtf": 0.45,
  "clip_ratio": 0.0008,
  "integrated_lufs": -18.2,
  "crest_factor_db": 14.3,
  "latency_ms_onset": 85,
  "missing_onset_rate": 0.01
}
```

---

## 9曲バッチ処理

### Step 1: バッチWAV生成

```bash
# 並列度: jobs=1→2→4 と段階的に（VST競合確認）
bash scripts/run_batch_vioptt_generation.sh \
  --root song_packages/test_project \
  --instrument piano_sfz_salamander \
  --vst-path "/Library/Audio/Plug-Ins/VST3/sfizz.vst3" \
  --jobs 2
```

**期待結果**:
```
🎻 VioPTT Batch WAV Generation
============================================================
Root:       song_packages/test_project
Instrument: piano_sfz_salamander
VST:        /Library/Audio/Plug-Ins/VST3/sfizz.vst3
Jobs:       2

📂 Found 9 song directories

[00:00:00] Processing test_song...
[00:00:15] ✅ test_song completed
[00:00:15] Processing test_song2...
[00:00:30] ✅ test_song2 completed
...
[00:02:15] ✅ All 9 songs completed!
```

### Step 2: Audio KPI一括検証

```bash
# 全9曲のAudio KPI検証
for song_dir in song_packages/test_project/*/; do
    song_name=$(basename "$song_dir")
    
    python3 scripts/validate_audio_quality.py \
      --wav "$song_dir/piano_sfz_salamander_rendered.wav" \
      --midi "$song_dir/piano_sfz_salamander_merged.mid" \
      --gate configs/audio_gate_prod.yaml \
      --profile piano_kpi \
      --out-json "$song_dir/audio_kpi_piano.json"
done
```

---

## Audio KPI集計

### 集計実行

```bash
python3 scripts/aggregate_audio_kpi.py \
  --root song_packages/test_project \
  --out-csv output/audio_kpi_summary.csv \
  --out-md output/audio_kpi_summary.md \
  --gate configs/audio_gate_prod.yaml \
  --profile piano_kpi
```

**期待出力**:

**output/audio_kpi_summary.csv**:
```csv
song,instrument,file,render_rtf,clip_ratio,integrated_lufs,crest_factor_db,overall_status
test_song,piano_sfz_salamander,test_song/audio_kpi_piano.json,0.45,0.0008,-18.2,14.3,PASS
test_song2,piano_sfz_salamander,test_song2/audio_kpi_piano.json,0.48,0.0009,-17.9,13.8,PASS
...
```

**output/audio_kpi_summary.md**:
```markdown
# Audio KPI Summary Report

**Generated**: 2025-10-30 01:00:00
**Total Songs**: 9

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| PASS | 9 | 100.0% |
| WARNING | 0 | 0.0% |
| FAIL | 0 | 0.0% |

## KPI Statistics

| KPI | Mean | Std | Min | Max | SLO | Status |
|-----|------|-----|-----|-----|-----|--------|
| render_rtf | 0.46 | 0.02 | 0.43 | 0.49 | ≤ 0.50 | ✅ PASS |
| clip_ratio | 0.0008 | 0.0001 | 0.0007 | 0.0010 | ≤ 0.001 | ✅ PASS |
| integrated_lufs | -18.1 | 0.3 | -18.5 | -17.8 | -22 - -16 | ✅ PASS |
| crest_factor_db | 14.2 | 0.5 | 13.5 | 14.8 | 10 - 20 | ✅ PASS |
```

---

## トラブルシューティング

### 音が出ない

**原因と対策**:

1. **CC11（Expression）が 0**:
   - `vioptt_mapping.yaml` で `min: 80` に変更

2. **VST/SFZパス権限**:
   - macOS: システム設定 → セキュリティとプライバシー → フルディスクアクセス → Terminal/VSCode 追加

3. **サンプル実体がない**:
   - Miroslav/SampleTank: ライブラリインストール確認
   - Salamander: SFZパス確認（`/path/to/SalamanderGrandPianoV3.sfz`）

4. **サンプルレート不一致**:
   - 48kHz版SFZを使う場合: `ENGINE_SR=48000` 設定
   - または自動検出（VST_PATHに "48khz" 含む）

### ペダルのバタつき

**症状**: CC64（Sustain Pedal）が頻繁にON/OFFする

**対策**:
```yaml
# vioptt_mapping.yaml
sustain_pedal:
  cc: 64
  source: "legato_ratio"
  threshold: 0.55
  hysteresis: 0.10  # ← 0.05 → 0.10 に上げる
```

### 明るさが刺さる

**症状**: SampleTank 4 Pianoの明るさが耳に刺さる

**対策**:
```yaml
# vioptt_mapping.yaml - piano_sampletank4
brightness:
  cc: 74
  source: "accent_score"
  min: 50
  max: 100  # ← 110 → 100 に下げる
  clip: "soft"
```

### WAV出力が無音

**確認ポイント**:

1. **制御MIDIのみで確認**:
   ```bash
   bash scripts/run_vioptt_pipeline.sh \
     --song-dir song_packages/test_project/test_song \
     --instrument piano_sfz_salamander \
     --no-merge \
     --tempo-bpm 120
   
   # piano_sfz_salamander_controls.mid をDAWで開く
   # CC11/64/1/91 が出ているか確認
   ```

2. **Duration延長**:
   - デフォルト: drums.mid duration + 2.0s
   - 手動延長: `--duration 128.0`（例）

3. **DAWDreamerエラー確認**:
   - ログファイル確認: `logs/rhythm_stage2_*.log`
   - Python エラー: `ImportError`, `RuntimeError` 等

---

## 運用TIPS

### CCスムージング調整

**既定: 1/8拍（0.125）**

- 伸ばしたい（滑らかに）: `VIOPTT_CC_SLEW_BEATS=0.25`（1/4拍）
- 短くしたい（キレ重視）: `VIOPTT_CC_SLEW_BEATS=0.0625`（1/16拍）

**例**:
```bash
# バラード（滑らか）
VIOPTT_CC_SLEW_BEATS=0.25 bash scripts/run_vioptt_pipeline.sh ...

# アップテンポ（キレ）
VIOPTT_CC_SLEW_BEATS=0.0625 bash scripts/run_vioptt_pipeline.sh ...
```

### サンプルレート選択

**44.1kHz（標準）**:
- 一般的なCD品質
- Salamander 44.1kHz版、SampleTank 4、Miroslav

**48kHz（高品質）**:
- 映像制作標準
- Salamander 48kHz版

**自動検出**:
- VST_PATHに "48khz" → ENGINE_SR=48000
- それ以外 → ENGINE_SR=44100

**手動設定**:
```bash
ENGINE_SR=48000 bash scripts/run_vioptt_pipeline.sh ...
```

### Audio KPIしきい値調整

**ピアノプロファイル（piano_kpi）微調整例**:

```yaml
# configs/audio_gate_prod.yaml
piano_kpi:
  # LUFS範囲を広げる（-24 〜 -14 LUFS）
  integrated_lufs:
    min: -24.0  # ← -22.0 から変更
    max: -14.0  # ← -16.0 から変更
  
  # クレストファクター範囲を広げる（8-22dB）
  crest_factor_db:
    min: 8.0    # ← 10.0 から変更
    max: 22.0   # ← 20.0 から変更
```

### リリース判定基準

**Go/No-Go チェックリスト**:

- [ ] Song生成KPI: Pass ≥ 95% / Warning 0-5% / Safe-Kit ≤ 10%
- [ ] Audio KPI（ピアノ）: 全9曲 PASS
- [ ] render_rtf ≤ 0.5（全曲2×速以上）
- [ ] clip_ratio ≤ 0.1%（全曲クリッピングなし）
- [ ] integrated_lufs: -22〜-16 LUFS（全曲）
- [ ] crest_factor_db: 10-20 dB（全曲）
- [ ] latency_ms_onset ≤ 120ms（全曲）
- [ ] missing_onset_rate ≤ 3%（全曲）

**1曲でもFAILがある場合**:
1. しきい値を微調整（上記参照）
2. マッピングを緩める（hysteresis拡大、min/max調整）
3. 楽曲固有の問題か全体的な問題か判断

---

## クイックリファレンス

### 単曲テスト（制御MIDIのみ）

```bash
python3 scripts/vioptt_render_stub.py \
  --hints song_packages/test_project/test_song/articulation_hints.json \
  --mapping configs/vioptt_mapping.yaml \
  --instrument piano_sfz_salamander \
  --output /tmp/test_piano.mid \
  --tempo-bpm 120
```

### 単曲WAVレンダリング

```bash
bash scripts/run_vioptt_pipeline.sh \
  --song-dir song_packages/test_project/test_song \
  --instrument piano_sfz_salamander \
  --vst-path "/Library/Audio/Plug-Ins/VST3/sfizz.vst3" \
  --tempo-bpm 120
```

### 9曲バッチ

```bash
bash scripts/run_batch_vioptt_generation.sh \
  --root song_packages/test_project \
  --instrument piano_sfz_salamander \
  --vst-path "/Library/Audio/Plug-Ins/VST3/sfizz.vst3" \
  --jobs 2
```

### Audio KPI集計

```bash
python3 scripts/aggregate_audio_kpi.py \
  --root song_packages/test_project \
  --out-csv output/audio_kpi_summary.csv \
  --out-md output/audio_kpi_summary.md \
  --gate configs/audio_gate_prod.yaml \
  --profile piano_kpi
```

---

**End of Guide**
