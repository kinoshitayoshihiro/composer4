# Phase D: 原曲追従強化（F0抽出・Piano転写・音色カーブ）

Phase C（CREPE/OaF受け口）を拡張し、**原曲の挙動を数値化→MIDI反映**する3つの最小スクリプトを実装しました。

---

## 概要

### 目的
- **F0抽出**: Bass/Lead/Guitarの音高カーブ（ピッチベンド、スライド、ビブラート）
- **Piano転写**: Onsets-and-Framesでボイシング/ペダル情報を取得
- **音色カーブ**: Synth/Padの明るさ/粗さ/振幅エンベロープを抽出

### NO-OP安全設計
- CREPE未導入→librosa YINフォールバック
- OaF未導入→librosaオンセット+pyinフォールバック
- DDSPモデル不要（librosa近似指標で即効）

---

## 1. F0抽出（ops/crepe_extract.py）

### 機能
- **CREPE優先**: 高精度F0抽出（16kHzリサンプル、Viterbi smoothing）
- **librosaフォールバック**: YIN/pyin（CREPE未導入時）
- **出力指標**（bar集計）:
  - `f0_median_hz`: 中央F0（Hz）
  - `f0_median_midi`: 中央F0（MIDIノート）
  - `f0_voiced_ratio`: 有声区間割合（0-1）
  - `vibrato_rate_hz`: ビブラート周波数（3-9Hz）
  - `slide_activity`: ピッチ変動量（90パーセンタイル）

### 実行例
```bash
# Bass F0抽出
python3 ops/crepe_extract.py \
  --audio data/.../song_001/stemswav_001/stem_wav_001_(Bass).wav \
  --bars song_packages/suno_project/song_001/bars.parquet \
  --out song_packages/suno_project/song_001/bass_f0.parquet \
  --hop-ms 10 --smooth-ms 120 --min-hz 27.5 --max-hz 880

# Lead/Guitar F0抽出（同様）
python3 ops/crepe_extract.py \
  --audio data/.../song_001/stemswav_001/stem_wav_001_(Lead).wav \
  --bars song_packages/suno_project/song_001/bars.parquet \
  --out song_packages/suno_project/song_001/lead_f0.parquet
```

### 出力確認
```bash
python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("song_packages/suno_project/song_001/bass_f0.parquet")
print(df.head(10))
# 期待: bar_index, f0_median_hz, f0_median_midi, f0_voiced_ratio, vibrato_rate_hz, slide_activity
PY
```

### E2E統合
```bash
bash scripts/e2e_suno_arrangement.sh \
  song_packages/suno_project/song_001 \
  --enable-f0-extract  # Bass/Lead/Guitar自動検出・抽出
```

---

## 2. Piano転写（ops/oaf_transcribe.py）

### 機能
- **Onsets-and-Frames優先**: `piano_transcription_inference`（高精度ペダル検出）
- **librosaフォールバック**: オンセット検出+pyin（OaF未導入時）
- **出力**:
  - `notes`: `[{start_sec, end_sec, midi, velocity, confidence, bar, beat, start_beats, end_beats}, ...]`
  - `pedal`: `[{start_sec, end_sec}, ...]`（サステインペダル区間）

### 実行例
```bash
python3 ops/oaf_transcribe.py \
  --audio data/.../song_001/stemswav_001/stem_wav_001_(Piano).wav \
  --bars song_packages/suno_project/song_001/bars.parquet \
  --tempo-bpm 74.677 --ppq 480 \
  --out song_packages/suno_project/song_001/piano_oaf.json
```

### 出力確認
```bash
python3 - <<'PY'
import json
obj = json.load(open("song_packages/suno_project/song_001/piano_oaf.json"))
print(f"Backend: {obj['meta']['backend']}")
print(f"Notes: {len(obj['notes'])}")
print(f"Pedal segments: {len(obj['pedal'])}")
print("\nFirst 3 notes:")
for n in obj['notes'][:3]:
    print(f"  Bar {n['bar']}, Beat {n['beat']:.2f}: MIDI {n['midi']}, Vel {n['velocity']}")
PY
```

### E2E統合
```bash
bash scripts/e2e_suno_arrangement.sh \
  song_packages/suno_project/song_001 \
  --enable-oaf  # Piano自動検出・転写（既存Phase C実装済み）
```

---

## 3. 音色カーブ（ops/ddsp_timbre_curves.py）

### 機能
- **DDSP不要**: librosaスペクトル特徴で近似
- **出力指標**（bar集計）:
  - `brightness`: スペクトル重心（正規化0-1）
  - `roughness`: スペクトルフラットネス（正規化0-1）
  - `am_env`: 振幅エンベロープ/RMS（正規化0-1）
  - `noise_high_ratio`: 高域（>6kHz）エネルギー比率
  - `vibrato_rate_hz`: 明るさ揺れから推定（3-9Hz）

### 実行例
```bash
# Synth/Pad音色カーブ抽出
python3 ops/ddsp_timbre_curves.py \
  --audio data/.../song_001/stemswav_001/stem_wav_001_(SynthPad).wav \
  --bars song_packages/suno_project/song_001/bars.parquet \
  --out song_packages/suno_project/song_001/synthpad_timbre.parquet \
  --hop-ms 20 --smooth-ms 200 --hi-cut-hz 6000
```

### 出力確認
```bash
python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("song_packages/suno_project/song_001/synthpad_timbre.parquet")
print(df.head(10))
# 期待: bar_index, brightness, roughness, am_env, noise_high_ratio, vibrato_rate_hz
PY
```

### E2E統合
```bash
bash scripts/e2e_suno_arrangement.sh \
  song_packages/suno_project/song_001 \
  --enable-timbre-curves  # Synth/Pad/Keys自動検出・抽出
```

---

## 既存ジェネレーターへの配線（最小NO-OP）

### scripts/instrument_midi_to_plan_real.py拡張案

**Bass**:
```python
# --bass-f0 bass_f0.parquet オプション追加
if args.bass_f0 and Path(args.bass_f0).exists():
    f0_df = pd.read_parquet(args.bass_f0)
    for bar in plan["bars"]:
        b = bar["bar_index"]
        row = f0_df[f0_df["bar_index"] == b]
        if row.empty:
            continue
        # レジスター補正（median_midiから適正オクターブ選択）
        median_midi = row.iloc[0]["f0_median_midi"]
        if np.isfinite(median_midi):
            bar["register_hint"] = int(median_midi // 12)  # オクターブ
        # スライド活性度（slide_activity > 0.5でポジション変動許可）
        if row.iloc[0]["slide_activity"] > 0.5:
            bar["allow_position_shift"] = True
        # ビブラート（vibrato_rate_hz > 4Hzで付与）
        if row.iloc[0]["vibrato_rate_hz"] > 4.0:
            bar["add_vibrato"] = True
```

**Piano**:
```python
# --oaf-piano piano_oaf.json オプション追加
if args.oaf_piano and Path(args.oaf_piano).exists():
    oaf = json.load(open(args.oaf_piano))
    notes = oaf.get("notes", [])
    pedal = oaf.get("pedal", [])
    
    # ボイシング補助（notesからvoicing密度推定）
    for bar in plan["bars"]:
        b = bar["bar_index"]
        bar_notes = [n for n in notes if n["bar"] == b]
        if bar_notes:
            # 同時発音数からvoicing複雑度推定
            unique_starts = set(n["start_sec"] for n in bar_notes)
            avg_polyphony = len(bar_notes) / max(1, len(unique_starts))
            bar["voicing_complexity"] = min(4, int(avg_polyphony))  # 1-4声
    
    # ペダル情報をsustain延長に反映
    for seg in pedal:
        # ペダル区間内のノートをsustain延長（end_beatsを次ノートまで伸ばす）
        bar_in_pedal = int((seg["start_sec"] * tempo_bpm / 60.0) // 4)
        if 0 <= bar_in_pedal < len(plan["bars"]):
            plan["bars"][bar_in_pedal]["apply_pedal"] = True
```

**Synth/Pad**:
```python
# --timbral-curves timbre.parquet オプション追加
if args.timbral_curves and Path(args.timbral_curves).exists():
    timbre_df = pd.read_parquet(args.timbral_curves)
    for bar in plan["bars"]:
        b = bar["bar_index"]
        row = timbre_df[timbre_df["bar_index"] == b]
        if row.empty:
            continue
        # CC11 Expression: am_env（0-1 → 0-127）
        bar["cc11_expression"] = int(row.iloc[0]["am_env"] * 127)
        # CC74 Brightness: brightness（0-1 → 0-127）
        bar["cc74_brightness"] = int(row.iloc[0]["brightness"] * 127)
        # CC1 Modulation: vibrato_rate_hz（4-9Hz → 64-127）
        vib = row.iloc[0]["vibrato_rate_hz"]
        bar["cc1_modulation"] = int(np.clip((vib - 4) / 5 * 63 + 64, 0, 127))
        # roughness（ノイズ系）: 高ければフィルター開く
        if row.iloc[0]["roughness"] > 0.7:
            bar["filter_open"] = True
```

---

## E2E統合実行例

### 全フェーズON（Phase A-D統合）
```bash
bash scripts/e2e_suno_arrangement.sh \
  song_packages/suno_project/song_001 \
  --drums-mode magenta \
  --enable-crepe \
  --enable-oaf \
  --enable-f0-extract \
  --enable-timbre-curves \
  --kpi
```

**実行フロー**:
1. Step 1.5: CREPE vocal F0抽出（Phase C）
2. Step 1.6: OaF Piano転写（Phase C）
3. **Step 1.7: F0抽出**（Phase D、Bass/Lead/Guitar）
4. **Step 1.8: 音色カーブ**（Phase D、Synth/Pad/Keys）
5. Step 2: Magenta GrooVAE humanize
6. Step 3-5: Bass/Guitar/Piano/Pads plan生成（F0/OaF/Timbre反映）
7. Step 6: MIDI書き出し
8. Step 7: KPI Gate検証

### 段階的テスト（Phase Dのみ）
```bash
# F0抽出のみ
bash scripts/e2e_suno_arrangement.sh \
  song_packages/suno_project/song_001 \
  --enable-f0-extract \
  --dry-run

# 音色カーブのみ
bash scripts/e2e_suno_arrangement.sh \
  song_packages/suno_project/song_001 \
  --enable-timbre-curves \
  --dry-run
```

---

## 依存インストール（オプショナル）

### CREPE（F0精度向上）
```bash
pip install crepe
# 効果: librosa YINより高精度、ビブラート検出改善
```

### Onsets-and-Frames（Piano転写精度向上）
```bash
pip install piano-transcription-inference
# 効果: ペダル情報取得、ボイシング複雑度改善
```

### librosa（必須、フォールバック用）
```bash
pip install librosa soundfile
# 既にインストール済みの場合はスキップ可
```

---

## トラブルシューティング

### Q1: CREPE/OaF未導入でも動く？
✅ **YES**: librosaフォールバックで自動動作（NO-OPではなく代替実行）

### Q2: 既存MIDI生成に影響ある？
✅ **NO**: `--enable-f0-extract`/`--enable-timbre-curves`未指定時は完全スキップ（従来通り）

### Q3: F0抽出が遅い
```bash
# hop-msを増やす（精度↓速度↑）
python3 ops/crepe_extract.py ... --hop-ms 20  # デフォルト10ms

# CREPEモデルを軽量化
# ops/crepe_extract.py Line 83: model='full' → model='tiny'
```

### Q4: Piano転写でノートが少なすぎる
```bash
# オンセット感度を上げる
# ops/oaf_transcribe.py librosa.onset.onset_detect に delta=0.01 追加
```

---

## 次のステップ

### Phase E: MIDI反映ロジック実装
- `scripts/instrument_midi_to_plan_real.py` に `--bass-f0`, `--oaf-piano`, `--timbral-curves` オプション追加
- Bass: レジスター/スライド/ビブラート反映
- Piano: ボイシング複雑度/ペダル延長反映
- Synth/Pad: CC11/CC74/CC1マッピング

### Phase F: ABテスト
- F0抽出ON vs OFF（Bass自然さ比較）
- OaF ON vs OFF（Pianoボイシング比較）
- Timbre ON vs OFF（Synth表現力比較）

### Phase G: KPI拡張
- F0追従精度（median_midi vs 生成MIDI）
- ボイシング多様性（unique pitches/bar）
- 音色ダイナミクス（CC変動範囲）

---

## 参考資料

- [CREPE論文](https://arxiv.org/abs/1802.06182)
- [Onsets-and-Frames論文](https://arxiv.org/abs/1710.11153)
- [piano_transcription_inference](https://github.com/bytedance/piano_transcription)
- [librosa Feature Extraction](https://librosa.org/doc/latest/feature.html)

---

**Phase D完了の証跡**:
- ✅ `ops/crepe_extract.py`（F0抽出、CREPEまたはlibrosaフォールバック）
- ✅ `ops/oaf_transcribe.py`（Piano転写、OaFまたはlibrosaフォールバック）
- ✅ `ops/ddsp_timbre_curves.py`（音色カーブ、librosa近似）
- ✅ E2E統合（`--enable-f0-extract`, `--enable-timbre-curves`フラグ追加）
- 📋 MIDI反映ロジック（Phase E待機中、配線ヒント提示済み）
