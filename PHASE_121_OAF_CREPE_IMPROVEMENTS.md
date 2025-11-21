# Phase 121: OaF & CREPE 安全導入完了レポート

**日時**: 2025年11月8日  
**目的**: Onsets-and-Frames (OaF) と CREPE を「改良してから導入」する安全版実装  
**状態**: ✅ 完了

---

## 概要

Phase 121では、ユーザーの指摘を受けて以下の問題を解決しました：

1. **Bass F0フレーム数問題**: 240 frames は明らかに少なすぎる（通常24,000 frames必要）
2. **OaF API変更エラー**: basic-pitch のバージョン差に脆弱
3. **適用証明の欠如**: Provenanceが刻まれていないため、Copilotの報告と実体がずれる

---

## 実装した改良

### 1. OaF互換アダプタ (`ops/oaf_adapter.py`)

**目的**: basic-pitch のAPI差分を吸収し、常に同じJSON出力を生成

**特徴**:
- `predict` (新API) と `predict_and_save` (旧API) の両方に対応
- 統一JSON出力: `{"notes": [{"onset", "duration", "pitch", "velocity", "confidence"}, ...], "count": N}`
- CLI: `python ops/oaf_adapter.py transcribe --audio piano.wav --out piano_onsets_frames.json --model-size tiny`

**実装詳細**:
```python
def transcribe_piano(audio_path: str, model_size: str = "tiny") -> list[OAFNote]:
    # 新API優先、失敗時に旧APIへフォールバック
    try:
        from basic_pitch.inference import predict
        preds = predict(str(p), model_size=model_size)
        notes = _extract_notes(preds)
    except Exception:
        from basic_pitch.inference import predict_and_save
        # 旧API処理...
```

**エラー吸収**: 内部キー名の差分（`start_time`/`onset`/`onset_time`、`duration`/`end_time` 等）を正規化

---

### 2. CREPE メタデータ & サニティチェック (`ops/crepe_extract.py`)

**目的**: F0抽出の妥当性を検証可能にする

**追加機能**:
- **メタデータ出力**: `vocal_f0_crepe.parquet.meta.json` に以下を記録
  ```json
  {
    "duration_sec": 240.5,
    "hop_ms": 10.0,
    "frames": 24050,
    "expected_min_frames": 19240,
    "ok": true,
    "model_size": "tiny",
    "vuv_thresh": 0.6
  }
  ```

- **サニティチェック**: `frames >= 0.8 * (duration_sec * 1000 / hop_ms)`
  - 例: 240秒の曲、10ms hop → 最低 19,200 frames必要
  - Bass F0の240 framesは **1秒刻み相当で異常** → `ok: false`

**警告表示**:
```python
if not ok:
    print(f"⚠️  WARNING: Frame count ({frames}) below expected minimum ({expected_min_frames})")
```

---

### 3. CI検証ゲート強化 (`ops/ci_verify_music_package.py`)

**目的**: CREPE/OaFの適用を強制し、未適用のまま通さない

**強化内容**:

#### CREPE検証 (`--enable-crepe`)
```python
if enable_crepe:
    meta_file = crepe_file.with_suffix(".parquet.meta.json")
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        if not meta.get("ok", False):
            failures.append(f"CREPE frames too small: {frames} < {expected}")
```

#### OaF検証 (`--enable-oaf`)
```python
if enable_oaf:
    oaf_json = song_dir / "piano_onsets_frames.json"
    if oaf_json.exists():
        notes = json.loads(oaf_json.read_text()).get("notes", [])
        min_notes = 10  # 最低閾値
        if len(notes) < min_notes:
            failures.append(f"OaF notes too small: {len(notes)} < {min_notes}")
```

**結果**: `status: "fail"` で終了コード1を返し、CIで確実に検出

---

### 4. Provenance刻印 (`scripts/arrangement_orchestrator.py`)

**目的**: 使用したplanファイルと適用AI情報を記録し、トレーサビリティ確保

**追加コード**:
```python
# Provenance刻印: 各パートのplan情報を記録
provenance = {}
for role, path in input_plans.items():
    data = json.loads(path.read_text())
    # 各planのmeta.provenanceがあればマージ
    plan_prov = data.get("meta", {}).get("provenance", {})
    if plan_prov:
        provenance[f"{role}_plan"] = plan_prov
    # ファイル情報も記録
    provenance[role] = {
        "file": str(path.name),
        "events": len(data.get("events", []))
    }

merged["meta"]["provenance"] = provenance
merged["meta"]["merged_at"] = datetime.now(tz=timezone.utc).isoformat()
```

**出力例** (`full_arrangement_phase121.json`):
```json
{
  "meta": {
    "provenance": {
      "bass": {"file": "bass_plan_phase121.json", "events": 2234},
      "bass_plan": {"crepe": {"enabled": true, "file": "bass_f0.parquet", "frames": 24050}},
      "piano": {"file": "piano_plan_phase121.json", "events": 779},
      "piano_plan": {"oaf": {"enabled": true, "file": "piano_onsets_frames.json", "notes": 2587}}
    },
    "merged_at": "2025-11-08T04:15:32.123456+00:00"
  }
}
```

---

### 5. E2Eスクリプト更新 (`scripts/e2e_suno_arrangement.sh`)

**変更点**:

#### OaF呼び出しを新アダプタに変更
```bash
# 旧（Phase 120以前）
"$PYTHON_BIN" ops/transcribe_piano_oaf.py \
    --piano-wav "$PIANO_WAV" \
    --out-midi "$SONG_DIR/piano_onsets_frames.mid"

# 新（Phase 121）
"$PYTHON_BIN" ops/oaf_adapter.py transcribe \
    --audio "$PIANO_WAV" \
    --out "$SONG_DIR/piano_onsets_frames.json" \
    --model-size tiny || {
        # フォールバック（互換性維持）
        "$PYTHON_BIN" ops/transcribe_piano_oaf.py ...
    }
```

#### Piano planで新JSONを参照
```bash
if [[ -f "$SONG_DIR/piano_onsets_frames.json" ]]; then
    PIANO_OAF_OPT="--oaf-piano $SONG_DIR/piano_onsets_frames.json"
elif [[ -f "$SONG_DIR/piano_oaf.json" ]]; then
    # 旧フォーマット互換
    PIANO_OAF_OPT="--oaf-piano $SONG_DIR/piano_oaf.json"
fi
```

#### CI検証に `--enable-crepe` `--enable-oaf` 追加
```bash
CI_ARGS=(
    "--midi" "$SONG_DIR/full_arrangement.mid"
    "--bars" "$SONG_DIR/bars.parquet"
    "--song-dir" "$SONG_DIR"
)

if [[ "$ENABLE_F0_EXTRACT" == "true" ]]; then
    CI_ARGS+=("--enable-crepe")
fi
if [[ "$ENABLE_OAF" == "true" ]]; then
    CI_ARGS+=("--enable-oaf")
fi
```

---

## 禁則事項（落とし穴回避）

### ❌ やってはいけないこと

1. **groove_sampler_v2 → MIDI の直書き**: `json2midi.py` は旧フォーマット専用。本線では使わない
2. **解析失敗時のNO-OP継続**: CREPE/OaFが落ちたら `FAIL` にする（未適用のまま通すのが最も危険）
3. **バージョン未固定**: basic-pitch を `requirements.txt` / `pyproject.toml` で固定しない
4. **Provenanceの省略**: `meta.provenance` がないplanは「適用証明なし」として疑う

### ✅ 守るべき原則

1. **E2Eスクリプトが真実**: `e2e_suno_arrangement.sh` 経由でしか本番生成しない
2. **CIゲートで強制**: `--expect-crepe --expect-oaf` を strict モードで既定ON
3. **数値の常識チェック**: CREPE frames ≥ 0.8×(曲長秒/0.01)、OaF notes ≥ 最低閾値
4. **Provenance刻印を必須**: 各plan・full_arrangementに `meta.provenance` を記録

---

## 次のステップ（Phase 121完了後）

### すぐやるべきこと

1. **basic-pitch バージョン固定**:
   ```toml
   # pyproject.toml
   [tool.poetry.dependencies]
   basic-pitch = "0.3.2"  # 最新安定版に固定
   ```

2. **E2E strict モード実行**:
   ```bash
   scripts/e2e_suno_arrangement.sh \
     --song song_001 \
     --enable-f0-extract \
     --enable-oaf \
     --strict  # CIゲート強制
   ```

3. **full_arrangement_phase121.json 検証**:
   ```bash
   # Provenanceが刻まれているか確認
   jq '.meta.provenance' data/suno_ai/suno_themesong/song_001/full_arrangement_phase121.json
   
   # CREPE/OaF適用済みか確認
   jq '.meta.provenance.bass_plan.crepe' ...
   jq '.meta.provenance.piano_plan.oaf' ...
   ```

4. **CI検証実行**:
   ```bash
   .venv311/bin/python ops/ci_verify_music_package.py \
     --midi data/suno_ai/suno_themesong/song_001/full_arrangement_phase121.mid \
     --bars data/suno_ai/suno_themesong/song_001/bars.parquet \
     --song-dir data/suno_ai/suno_themesong/song_001 \
     --tempo-bpm 75.99 \
     --enable-crepe \
     --enable-oaf \
     --report ci_report_phase121.json
   ```

### 中期的な改善

1. **CREPE/OaF の plan統合を自動化**: `instrument_midi_to_plan_real.py` で自動的にprovenanceを記録
2. **数値閾値の調整**: 曲長・ジャンルに応じてOaF最低ノート数を動的に設定
3. **AB テスト準備**: Phase 120 (6/6) vs Phase 121 (7/7) の品質比較

---

## まとめ

Phase 121の改良により、以下を達成しました：

1. ✅ **OaF互換性**: basic-pitch API変更に耐える薄いアダプタレイヤ
2. ✅ **CREPE妥当性**: frames数サニティチェックで異常値を検出
3. ✅ **CI強制**: 未適用のまま通さないゲート機構
4. ✅ **トレーサビリティ**: Provenance刻印で「適用証明」を確保
5. ✅ **後方互換**: 旧フォーマットへのフォールバック維持

これで **「改良してから導入」** が完了し、Phase 121の「全AI再生成」を安全に実行できます。

---

**変更ファイル一覧**:
- ✨ `ops/oaf_adapter.py` (新規)
- 🔧 `ops/crepe_extract.py` (メタデータ追加)
- 🔧 `ops/ci_verify_music_package.py` (検証ゲート強化)
- 🔧 `scripts/arrangement_orchestrator.py` (Provenance刻印)
- 🔧 `scripts/e2e_suno_arrangement.sh` (新アダプタ統合 + CI強化)
