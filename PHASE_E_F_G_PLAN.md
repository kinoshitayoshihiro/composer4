# Phase E-G実装計画（微修正完了後）

## ✅ 完了済み（Phase A-D + 微修正1-7）

### Phase A: Magenta統合
- ✅ Magenta GrooVAE統合完了（humanize効果確認済み）
- ✅ 専用venv分離（requirements-magenta.txt作成）
- ✅ CI検証スクリプト（verify_magenta_humanize.py IOI std追加）

### Phase B: GMD学習準備
- ✅ データセット準備スクリプト完成
- 📋 軽学習実施待機中（オプショナル）

### Phase C: CREPE/OaF受け口
- ✅ 前処理スクリプト準備完了
- ✅ E2E統合準備完了

### Phase D: 原曲追従強化
- ✅ ops/crepe_extract.py作成（F0抽出）
- ✅ ops/oaf_transcribe.py作成（Piano転写）
- ✅ ops/ddsp_timbre_curves.py作成（音色カーブ）
- ✅ E2E統合（--enable-f0-extract, --enable-timbre-curves）
- ✅ docs/PHASE_D_ORIGINAL_TRACKING.md作成

### 微修正1-7
1. ✅ E2E堅牢化: set -Eeuo pipefail（既存確認）
2. ✅ 変数引用: パス変数ダブルクォート（既存確認）
3. ✅ Magenta venv固定: requirements-magenta.txt作成
4. ✅ CI追加チェック: PPQ確認、Drumsチャンネル維持確認
5. ✅ verify_magenta_humanize.py: IOI std追加（小数点4桁精度向上）
6. ✅ KPI Auto-Repair: Magenta時穏当化（backbeat_strength*0.9、metadata.json自動読取）
7. ✅ adapt_drums_to_plan.py: tempo_bpm必須明示（コメント追加）

---

## 📋 Phase E: MIDI反映ロジック実装（次のフェーズ）

### 目的
Phase Dで抽出した原曲特徴をMIDI生成に反映

### 実装内容

#### 1. Bass F0反映（ops/crepe_extract.py出力活用）
**ファイル**: `scripts/instrument_midi_to_plan_real.py`

**追加オプション**:
```bash
--bass-f0 <path>  # bass_f0.parquet from crepe_extract.py
```

**反映ロジック**:
```python
# instrument_midi_to_plan_real.py修正（疑似コード）
ap.add_argument("--bass-f0", type=Path, help="Bass F0 parquet from crepe_extract.py")

if args.bass_f0 and Path(args.bass_f0).exists():
    f0_df = pd.read_parquet(args.bass_f0)
    for bar in plan["bars"]:
        b = bar["bar_index"]
        row = f0_df[f0_df["bar_index"] == b]
        if row.empty: continue
        
        # レジスター補正（中央F0からオクターブ推定）
        median_midi = row.iloc[0]["f0_median_midi"]
        if np.isfinite(median_midi):
            bar["register_hint"] = int(median_midi // 12)  # C0=0, C1=1, ...
        
        # ポジション補正（スライド活性度）
        if row.iloc[0]["slide_activity"] > 0.5:
            bar["allow_position_shift"] = True  # フレット移動許可
        
        # ビブラート追加（振動周波数）
        if row.iloc[0]["vibrato_rate_hz"] > 4.0:
            bar["add_vibrato"] = True
```

**期待効果**:
- Bass音域が原曲に近づく（register_hint）
- スライド奏法の再現性向上（allow_position_shift）
- ビブラート表現の自動化（add_vibrato）

---

#### 2. Piano OaF反映（ops/oaf_transcribe.py出力活用）
**ファイル**: `scripts/instrument_midi_to_plan_real.py`

**追加オプション**:
```bash
--oaf-piano <path>  # piano_oaf.json from oaf_transcribe.py
```

**反映ロジック**:
```python
ap.add_argument("--oaf-piano", type=Path, help="Piano OaF JSON from oaf_transcribe.py")

if args.oaf_piano and Path(args.oaf_piano).exists():
    oaf = json.load(open(args.oaf_piano))
    notes = oaf.get("notes", [])
    pedal = oaf.get("pedal", [])
    
    # ボイシング複雑度（和音密度）
    for bar in plan["bars"]:
        b = bar["bar_index"]
        bar_notes = [n for n in notes if n["bar"] == b]
        if bar_notes:
            unique_starts = set(n["start_sec"] for n in bar_notes)
            avg_polyphony = len(bar_notes) / max(1, len(unique_starts))
            bar["voicing_complexity"] = min(4, int(avg_polyphony))  # 1-4段階
    
    # ペダル延長（サステインペダル区間）
    for seg in pedal:
        bar_in_pedal = int((seg["start_sec"] * tempo_bpm / 60.0) // 4)
        if 0 <= bar_in_pedal < len(plan["bars"]):
            plan["bars"][bar_in_pedal]["apply_pedal"] = True
```

**期待効果**:
- Pianoボイシングが原曲に近づく（voicing_complexity）
- サステインペダルの適切な再現（apply_pedal）

---

#### 3. Synth/Pad Timbre反映（ops/ddsp_timbre_curves.py出力活用）
**ファイル**: `scripts/instrument_midi_to_plan_real.py`

**追加オプション**:
```bash
--timbral-curves <path>  # synthpad_timbre.parquet from ddsp_timbre_curves.py
```

**反映ロジック**:
```python
ap.add_argument("--timbral-curves", type=Path, help="Timbre curves parquet from ddsp_timbre_curves.py")

if args.timbral_curves and Path(args.timbral_curves).exists():
    timbre_df = pd.read_parquet(args.timbral_curves)
    for bar in plan["bars"]:
        b = bar["bar_index"]
        row = timbre_df[timbre_df["bar_index"] == b]
        if row.empty: continue
        
        # CC11 Expression（振幅エンベロープ）
        bar["cc11_expression"] = int(row.iloc[0]["am_env"] * 127)
        
        # CC74 Brightness（スペクトル重心）
        bar["cc74_brightness"] = int(row.iloc[0]["brightness"] * 127)
        
        # CC1 Modulation（ビブラート）
        vib = row.iloc[0]["vibrato_rate_hz"]
        bar["cc1_modulation"] = int(np.clip((vib - 4) / 5 * 63 + 64, 0, 127))
        
        # フィルター開度（ラフネス）
        if row.iloc[0]["roughness"] > 0.7:
            bar["filter_open"] = True  # 高域強調
```

**期待効果**:
- Synth音色のダイナミクス向上（CC11/CC74）
- ビブラート表現の自動化（CC1）
- フィルター開度の動的制御（filter_open）

---

### 実装手順

1. **instrument_midi_to_plan_real.py修正**（3オプション追加）
   - --bass-f0: F0反映ロジック追加
   - --oaf-piano: OaF反映ロジック追加
   - --timbral-curves: Timbre反映ロジック追加

2. **E2E統合**（scripts/e2e_suno_arrangement.sh）
   - Step 3.2-3.4修正（各オプション配線）
   ```bash
   # Step 3.2: Bass MIDI生成
   if [[ -f "$SONG_DIR/bass_f0.parquet" ]]; then
       BASS_F0_OPT="--bass-f0 $SONG_DIR/bass_f0.parquet"
   fi
   "$PYTHON_BIN" scripts/instrument_midi_to_plan_real.py \
       --role bass --plan "$SONG_DIR/full_arrangement.json" \
       $BASS_F0_OPT \
       --out "$SONG_DIR/bass.mid"
   ```

3. **検証**
   - 単体テスト（各オプション単独ON/OFF）
   - E2E統合テスト（全オプションON）
   - KPI Gate確認（品質劣化ないこと）

---

## 📋 Phase F: ABテスト

### 目的
Phase D/E機能の効果測定

### テスト項目

#### F1: F0抽出ON vs OFF（Bass自然さ）
- **評価指標**:
  - 原曲F0との距離（Hz、MIDIノート）
  - レジスター一致率
  - スライド再現率

#### F2: OaF ON vs OFF（Pianoボイシング）
- **評価指標**:
  - ボイシング複雑度の一致率
  - ペダル適用率の一致率
  - 和音密度の近似度

#### F3: Timbre ON vs OFF（Synth表現力）
- **評価指標**:
  - CC変動範囲（0-127のうち使用範囲）
  - Brightness近似度（スペクトル重心）
  - Expression変動率

---

## 📋 Phase G: KPI拡張

### 新規KPI追加

#### G1: F0追従精度（Bass/Lead）
- **定義**: `median_midi（原曲） vs 生成MIDI中央値`
- **合格基準**: ±2半音以内
- **実装**: `scripts/kpi_gate_enhanced.py`追加チェック

#### G2: ボイシング多様性（Piano）
- **定義**: `unique pitches/bar`
- **合格基準**: 原曲の70%以上
- **実装**: `scripts/kpi_gate_enhanced.py`追加チェック

#### G3: 音色ダイナミクス（Synth/Pad）
- **定義**: `CC11/CC74/CC1変動範囲`
- **合格基準**: 変動範囲≧30（0-127のうち30以上使用）
- **実装**: `scripts/kpi_gate_enhanced.py`追加チェック

---

## 📋 Phase H: 本番運用

### 目標
- 5万曲バッチ処理
- KPI Gate Pass率90%以上
- 継続的改善体制確立

### モニタリング項目
- KPI Gate Pass率（全体、Phase D/E寄与分）
- F0追従精度（Phase E寄与分）
- ボイシング多様性（Phase E寄与分）
- 音色ダイナミクス（Phase E寄与分）

---

## 次のアクション

**Phase E開始準備完了**

微修正1-7完了により、Phase E実装の準備が整いました。

次のステップ:
1. `scripts/instrument_midi_to_plan_real.py`修正（3オプション追加）
2. E2E統合（Step 3.2-3.4修正）
3. 単体テスト（各オプション単独検証）
4. E2E統合テスト（全オプションON検証）

**実装開始指示を待機中**
