# Phase E実装完了レポート

## ✅ 実装完了（2025年11月5日）

### 目的
Phase Dで抽出した原曲特徴（F0、Piano OaF、音色カーブ）をMIDI生成に反映

---

## 実装内容

### 1. instrument_midi_to_plan_real.py修正

#### 1.1 オプション追加（Line 1186-1205）
```python
# Phase E ガイド入力（原曲追従強化）
ap.add_argument("--bass-f0", type=str, default=None,
    help="bass_f0.parquet (Phase D: Bass F0抽出、レジスター/スライド/ビブラート反映用)")
ap.add_argument("--oaf-piano", type=str, default=None,
    help="piano_oaf.json (Phase D: Piano OaF転写、ボイシング/ペダル反映用)")
ap.add_argument("--timbral-curves", type=str, default=None,
    help="synthpad_timbre.parquet (Phase D: 音色カーブ、CC11/CC74/CC1反映用)")
```

#### 1.2 ガイド入力読み込み（Line 1315-1365）
- **bass_f0.parquet読み込み**:
  - bar集計データ: f0_median_midi, slide_activity, vibrato_rate_hz
  - DEBUG出力: Bars数、中央MIDI平均
- **piano_oaf.json読み込み**:
  - notes配列: bar/beat付きノート情報
  - pedal配列: サステインペダル区間
  - DEBUG出力: ノート数、ペダルセグメント数
- **timbre_curves.parquet読み込み**:
  - bar集計データ: brightness, roughness, am_env, vibrato_rate_hz
  - DEBUG出力: Bars数、Brightness平均

#### 1.3 bars拡張処理（Line 1368-1420）

**Bass F0反映**（args.role == "bass"時のみ）:
```python
if bass_f0_df is not None and args.role == "bass":
    for b in bars:
        if median_midi is finite:
            bars["register_hint"] = median_midi // 12  # C0=0, C1=1, ...
        if slide_activity > 0.5:
            bars["allow_position_shift"] = True
        if vibrato_rate_hz > 4.0:
            bars["add_vibrato"] = True
```

**Piano OaF反映**（args.role == "piano"時のみ）:
```python
if oaf_piano_data is not None and args.role == "piano":
    # ボイシング複雑度
    for b in bars:
        bar_notes = [n for n in notes if n["bar"] == b]
        unique_starts = set(n["start_sec"] for n in bar_notes)
        avg_polyphony = len(bar_notes) / max(1, len(unique_starts))
        bars["voicing_complexity"] = min(4, int(avg_polyphony))
    
    # ペダル延長
    for seg in pedal:
        bar_in_pedal = int((seg["start_sec"] * bpm / 60.0) // 4)
        bars[bar_in_pedal]["apply_pedal"] = True
```

**Synth/Pad Timbre反映**（args.role in ["strings", "piano"]時）:
```python
if timbre_curves_df is not None and args.role in ["strings", "piano"]:
    for b in bars:
        bars["cc11_expression"] = int(am_env * 127)
        bars["cc74_brightness"] = int(brightness * 127)
        bars["cc1_modulation"] = int(np.clip((vib - 4) / 5 * 63 + 64, 0, 127))
        if roughness > 0.7:
            bars["filter_open"] = True
```

#### 1.4 Bass生成ロジック修正（Line 1562-1623）

**register_hint反映**:
```python
if register_hint is not None:
    current_octaves = [p // 12 for p in pitches]
    avg_octave = sum(current_octaves) / len(current_octaves)
    octave_shift = register_hint - avg_octave
    if abs(octave_shift) >= 0.5:
        shift_semitones = int(round(octave_shift)) * 12
        pitches = [p + shift_semitones for p in pitches]
```

**allow_position_shift反映**（スライド表現）:
```python
if allow_shift and gi > 0:
    prev_p = pitches[gi - 1]
    if abs(p - prev_p) <= 2:  # 近接音
        p += random.choice([-1, 1])  # 半音ずらす
```

**add_vibrato反映**（ベロシティ揺らぎ）:
```python
if add_vib:
    vel += random.choice([-3, 0, 3])  # ビブラート風揺らぎ
```

#### 1.5 Piano生成ロジック修正（Line 1657-1690）

**voicing_complexity反映**（ボイシング拡張）:
```python
expanded_vo = vo[:]
if voicing_complexity >= 3 and len(vo) >= 2:
    expanded_vo.append(vo[1] + 12)  # オクターブ上追加
if voicing_complexity >= 4 and len(vo) >= 3:
    expanded_vo.append(vo[2] + 12)  # さらに3度上追加
```

**apply_pedal反映**（duration延長）:
```python
if apply_pedal:
    lens = [min(l * 1.5, seg.end_b - t) for t, l in zip(times, lens)]
```

#### 1.6 Plan出力にTimbre CC情報追加（Line 1862-1882）
```python
# Phase E: Timbre CC情報をbarsメタに追加（plan_to_midiで利用）
if timbre_curves_df is not None and args.role in ["strings", "piano"]:
    bars_cc = []
    for row in bars:
        bar_cc = {
            "bar_index": b,
            "cc11": int(row["cc11_expression"]),
            "cc74": int(row["cc74_brightness"]),
            "cc1": int(row["cc1_modulation"]),
            "filter_open": bool(row.get("filter_open", False))
        }
        bars_cc.append(bar_cc)
    plan["meta"]["timbre_cc"] = bars_cc
```

---

### 2. E2E統合（scripts/e2e_suno_arrangement.sh）

#### 2.1 Bass F0オプション配線（Line 650-656）
```bash
# Phase E: Bass F0オプション追加
BASS_F0_OPT=""
if [[ -f "$SONG_DIR/bass_f0.parquet" ]]; then
    BASS_F0_OPT="--bass-f0 $SONG_DIR/bass_f0.parquet"
    echo "      [Phase E] Bass F0 detected: $SONG_DIR/bass_f0.parquet"
fi
# instrument_midi_to_plan_real.py実行時に$BASS_F0_OPT追加
```

#### 2.2 Piano OaFオプション配線（Line 722-728）
```bash
# Phase E: Piano OaFオプション追加
PIANO_OAF_OPT=""
if [[ -f "$SONG_DIR/piano_oaf.json" ]]; then
    PIANO_OAF_OPT="--oaf-piano $SONG_DIR/piano_oaf.json"
    echo "      [Phase E] Piano OaF detected: $SONG_DIR/piano_oaf.json"
fi
# instrument_midi_to_plan_real.py実行時に$PIANO_OAF_OPT追加
```

#### 2.3 Timbre Curvesオプション配線（Line 760-770）
```bash
# Phase E: Timbre Curvesオプション追加（Synth/Pad代表としてstringsに適用）
TIMBRE_CURVES_OPT=""
# Synth/Pad用のtimbre curves検索（複数パターン対応）
for STEM_ROLE in "synthpad" "synth" "pad" "keys"; do
    if [[ -f "$SONG_DIR/${STEM_ROLE}_timbre.parquet" ]]; then
        TIMBRE_CURVES_OPT="--timbral-curves $SONG_DIR/${STEM_ROLE}_timbre.parquet"
        echo "      [Phase E] Timbre curves detected: $SONG_DIR/${STEM_ROLE}_timbre.parquet"
        break
    fi
done
# instrument_midi_to_plan_real.py実行時に$TIMBRE_CURVES_OPT追加
```

---

## 期待効果

### Bass（--bass-f0）
- ✅ **レジスター補正**: 原曲F0中央値からオクターブ推定、音域一致
- ✅ **スライド表現**: slide_activity > 0.5で前音から半音ずらす（フレット移動風）
- ✅ **ビブラート**: vibrato_rate_hz > 4.0でベロシティ揺らぎ追加

### Piano（--oaf-piano）
- ✅ **ボイシング複雑度**: 和音密度に応じて追加音挿入（1-4段階）
- ✅ **ペダル延長**: サステインペダル区間でduration 1.5倍

### Synth/Pad（--timbral-curves）
- ✅ **CC11 Expression**: 振幅エンベロープ反映（0-127）
- ✅ **CC74 Brightness**: スペクトル重心反映（0-127）
- ✅ **CC1 Modulation**: ビブラート周波数反映（0-127）
- ✅ **filter_open**: roughness > 0.7で高域強調フラグ

---

## NO-OP安全設計

### 完全オプショナル
- Phase Dガイド未存在時: エラーなし、通常生成続行
- 依存未インストール時: WARNINGのみ、処理継続
- ロール不一致時: 対象ロール以外は無視

### DEBUG出力
- `--debug`フラグでガイド読み込み状況詳細表示
- bars拡張属性の適用状況カウント表示

---

## 次のアクション

### Phase F: ABテスト準備完了
- F1: Bass F0 ON/OFF比較（レジスター一致率測定）
- F2: Piano OaF ON/OFF比較（ボイシング複雑度測定）
- F3: Timbre ON/OFF比較（CC変動範囲測定）

### Phase G: KPI拡張実装待機
- G1: F0追従精度チェック追加
- G2: ボイシング多様性チェック追加
- G3: 音色ダイナミクスチェック追加

---

## Phase E実装完了

**実装ファイル**:
1. ✅ `scripts/instrument_midi_to_plan_real.py`（3オプション追加、bars拡張、ロジック反映）
2. ✅ `scripts/e2e_suno_arrangement.sh`（Bass/Piano/Strings配線完了）

**実装状況**: 全機能完了、E2E統合完了

**次フェーズ**: Phase F（ABテスト）実施準備完了
