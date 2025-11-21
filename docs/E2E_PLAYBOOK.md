# E2E Playbook - Suno Theme Song Pipeline

**目的**: Suno テーマ曲パイプラインの"本線（Golden Path）"を1枚で明確化し、Copilot由来のバイパスや新設スクリプト混入を防ぐ。

---

## Golden Path（E2E本線）

### 1. 素材→パッケージ

```bash
scripts/make_song_package_from_sources.sh <song_dir> --stems-dir <stems_dir>
```

**生成物**:
- `bars.parquet` (21 cols)
- `sections.json`
- `chordmap.json`
- `lyric_anchors.json`
- `stem_features.parquet`
- ほか

### 2. ミックス・バリアント

```bash
python3 scripts/generate_suno_song_package_v1_1.py \
  --song-id ${SONG_ID} \
  --analysis-dir data/suno_ai/suno_themesong/${SONG_ID}/analysis \
  --variant soft|standard|bright \
  --out data/suno_ai/suno_themesong/${SONG_ID}/song_package_<variant>.yaml
```

**3バリアント**:
- `soft`: 柔らかい表現
- `standard`: 標準（デフォルト）
- `bright`: 明るく強調

### 3. アレンジ生成（全AI）

```bash
./scripts/e2e_suno_arrangement.sh <song_dir>
```

**必須AI**:
- ✅ **Magenta** (Drums)
- ✅ **CREPE** (Vocal F0 → Bass guide)
- ✅ **OaF** (Onsets-and-Frames: Piano転写)
- ✅ **RhythmAI** (Drum density guide)
- ✅ **EmotionAI** (感情プロファイル)
- ✅ **HarmonyAI** (和声進行、usage_history.db学習)

### 4. 統合 & 書き出し

```bash
python3 scripts/midi_writer.py \
  --bass <bass_plan.json> \
  --guitar <guitar_plan.json> \
  --piano <piano_plan.json> \
  --strings <strings_plan.json> \
  --drums <drums_plan.json> \
  --tempo-bpm <BPM> \
  --out <song_dir>/full_arrangement.mid
```

**重要**: `midi_writer.py`が本線。`json2midi.py`等の直書きツールは**禁止**。

### 5. CI検証（厳格）

```bash
python3 scripts/ci_verify_music_package.py \
  --song-dir <song_dir> \
  --strict
```

**検証項目**:
- Duration整合性
- PPQ一貫性
- Over-end event検出
- Provenance完全性
- AI適用確認（CREPE/OaF/Magenta/EmotionAI/HarmonyAI）

---

## 禁止事項（Fail扱い）

### ❌ 1. 直書きMIDI禁止

`groove_sampler_v2 → MIDI`のような直書きユーティリティで最終MIDIを作らない。

**理由**: Provenance欠落、再現性喪失。

### ❌ 2. Provenance欠落

`*_plan.json` / `full_arrangement.json`にAI使用痕跡が無い（例: `meta.provenance`）。

**必須フィールド**:
```json
{
  "meta": {
    "provenance": {
      "bass_f0": {"enabled": true, "file": "bass_f0.parquet", "bars": 240},
      "oaf": {"enabled": true, "file": "piano_onsets_frames.json", "notes": 3047},
      "emotion_ai": {"enabled": true, "profile": "auto"},
      "harmony_ai": {"enabled": true, "usage_db": "usage_history.db"},
      "magenta": {"enabled": true, "model": "groove", "temperature": 1.0}
    }
  }
}
```

### ❌ 3. 解析のみ・未適用

CREPE/OaFの出力ファイルはあるが、planの`context_sources`に反映されていない。

**例**:
```json
// ❌ BAD
"context_sources": {
  "bass_f0": false,  // ← bass_f0.parquetがあるのにfalse
  "oaf_piano": false // ← piano_onsets_frames.jsonがあるのにfalse
}

// ✅ GOOD
"context_sources": {
  "bass_f0": true,   // ← bass_f0.parquet適用
  "oaf_piano": true  // ← piano_onsets_frames.json適用
}
```

---

## 必須チェックリスト（Commit前に）

- [ ] **全5パート生成完了** (Bass, Guitar, Piano, Strings, Drums)
- [ ] **Bass F0適用確認** (`context_sources.bass_f0 = true`)
- [ ] **Piano OaF適用確認** (`context_sources.oaf_piano = true`)
- [ ] **Drums Magenta生成確認** (`meta.provenance.magenta.enabled = true`)
- [ ] **EmotionAI/HarmonyAI適用** (全パートで`provenance`確認)
- [ ] **Full Arrangement統合** (`midi_writer.py`使用)
- [ ] **CI検証PASS** (`ci_verify_music_package.py --strict`)
- [ ] **MIDI再生確認** (Logic Pro / GarageBandで聴取)
- [ ] **Provenance完全性** (全AI適用記録確認)

---

## トラブルシュート

### 1. CREPE framesが小さすぎる（例: 240）

**症状**: `bass_f0.meta.json`の`frames`が240程度しかない。

**原因**: 
- hop長設定ミス（hop_ms=1000等）
- 処理失敗（NumPy互換性、入力WAVパス誤り）

**対策**:
```bash
# 正しいCREPE抽出（hop_ms=10.0推奨）
python3 ops/crepe_extract.py \
  --audio <bass_stem.wav> \
  --out <bass_f0.parquet> \
  --hop-ms 10.0 \
  --model tiny
```

**期待値**: 480秒の曲で約48,000フレーム（10ms刻み）。

### 2. OaF API変更

**症状**: `basic-pitch`のバージョンアップでAPI破損。

**対策**: `ops/oaf_adapter.py`（互換レイヤ）で旧/新APIを吸収。

**バージョン固定**:
```txt
# requirements.txt
basic-pitch==0.4.0  # ← 固定推奨
```

### 3. Magenta未適用疑い

**症状**: `drums_plan.json`の`meta.provenance`に`magenta`が無い。

**確認**:
```bash
python3 -c "
import json
with open('drums_plan.json') as f:
    data = json.load(f)
    prov = data.get('meta', {}).get('provenance', {})
    print('Magenta:', prov.get('magenta', 'MISSING'))
"
```

**対策**: `e2e_suno_arrangement.sh`内でMagenta呼び出し確認。

---

## よく使うコマンド（雛形）

※ `venv`は任意。`--help`で各スクリプトの正規オプションを再確認すること。

### ① Package生成

```bash
SONG_ID="song_001"
STEMS_DIR="stemswav_001"

bash scripts/make_song_package_from_sources.sh \
  data/suno_ai/suno_themesong/${SONG_ID} \
  --stems-dir "data/suno_ai/suno_themesong/${SONG_ID}/${STEMS_DIR}"
```

### ② Song Package（3 variants）

```bash
# Soft variant
python3 scripts/generate_suno_song_package_v1_1.py \
  --song-id ${SONG_ID} \
  --analysis-dir data/suno_ai/suno_themesong/${SONG_ID}/analysis \
  --variant soft \
  --out data/suno_ai/suno_themesong/${SONG_ID}/song_package_soft.yaml

# Standard variant
python3 scripts/generate_suno_song_package_v1_1.py \
  --song-id ${SONG_ID} \
  --analysis-dir data/suno_ai/suno_themesong/${SONG_ID}/analysis \
  --variant standard \
  --out data/suno_ai/suno_themesong/${SONG_ID}/song_package_v1_1.yaml

# Bright variant
python3 scripts/generate_suno_song_package_v1_1.py \
  --song-id ${SONG_ID} \
  --analysis-dir data/suno_ai/suno_themesong/${SONG_ID}/analysis \
  --variant bright \
  --out data/suno_ai/suno_themesong/${SONG_ID}/song_package_bright.yaml
```

### ③ E2E（strict）

```bash
./scripts/e2e_suno_arrangement.sh data/suno_ai/suno_themesong/${SONG_ID} \
  && python3 scripts/ci_verify_music_package.py \
       --song-dir data/suno_ai/suno_themesong/${SONG_ID} \
       --strict
```

### ④ 個別パート生成（例: Bass with F0）

```bash
python3 scripts/instrument_midi_to_plan_real.py \
  --role bass \
  --song-package data/suno_ai/suno_themesong/${SONG_ID}/song_package.yaml \
  --bars data/suno_ai/suno_themesong/${SONG_ID}/bars.parquet \
  --chordmap data/suno_ai/suno_themesong/${SONG_ID}/chordmap.json \
  --sections data/suno_ai/suno_themesong/${SONG_ID}/sections.json \
  --stems-features data/suno_ai/suno_themesong/${SONG_ID}/stem_features.parquet \
  --bass-f0 data/suno_ai/suno_themesong/${SONG_ID}/bass_f0.parquet \
  --voice-leading --multi-chords --anchors-strict --follow-drum-density \
  --enable-emotion-ai --enable-harmony-ai --emotion-profile auto \
  --out data/suno_ai/suno_themesong/${SONG_ID}/bass_plan_phase121.json
```

### ⑤ Piano with OaF

```bash
python3 scripts/instrument_midi_to_plan_real.py \
  --role piano \
  --song-package data/suno_ai/suno_themesong/${SONG_ID}/song_package.yaml \
  --bars data/suno_ai/suno_themesong/${SONG_ID}/bars.parquet \
  --chordmap data/suno_ai/suno_themesong/${SONG_ID}/chordmap.json \
  --sections data/suno_ai/suno_themesong/${SONG_ID}/sections.json \
  --stems-features data/suno_ai/suno_themesong/${SONG_ID}/stem_features.parquet \
  --oaf-piano data/suno_ai/suno_themesong/${SONG_ID}/piano_onsets_frames.json \
  --voice-leading --multi-chords --anchors-strict --follow-drum-density \
  --enable-emotion-ai --enable-harmony-ai --emotion-profile auto \
  --out data/suno_ai/suno_themesong/${SONG_ID}/piano_plan_phase121.json
```

### ⑥ Full Arrangement統合

```bash
python3 scripts/arrangement_orchestrator.py \
  --bass data/suno_ai/suno_themesong/${SONG_ID}/bass_plan_phase121.json \
  --guitar data/suno_ai/suno_themesong/${SONG_ID}/guitar_plan_phase121.json \
  --piano data/suno_ai/suno_themesong/${SONG_ID}/piano_plan_phase121.json \
  --strings data/suno_ai/suno_themesong/${SONG_ID}/strings_plan_phase121.json \
  --drums data/suno_ai/suno_themesong/${SONG_ID}/drums_plan_phase121.json \
  --tempo-bpm 75 \
  --out data/suno_ai/suno_themesong/${SONG_ID}/full_arrangement_phase121.json
```

### ⑦ MIDI変換（本線）

```bash
python3 scripts/midi_writer.py \
  --plan data/suno_ai/suno_themesong/${SONG_ID}/full_arrangement_phase121.json \
  --out data/suno_ai/suno_themesong/${SONG_ID}/full_arrangement_phase121.mid
```

---

## 注意事項

### スクリプト引数名の実装差

各スクリプトの引数名（`--stem-features` / `--stems-features`、`--oaf-json` / `--oaf-piano`など）は実装差があります。

**必ず`--help`で確認**:
```bash
python3 scripts/instrument_midi_to_plan_real.py --help
python3 scripts/midi_writer.py --help
```

### Pinned Context推奨（Copilot/Continue）

`docs/E2E_PLAYBOOK.md`をCopilot/Continueの**Pinned Context**に固定し、毎回参照させると誤配線を防げます。

**設定例（Continue）**:
```json
// .continue/config.json
{
  "pinnedContext": [
    "docs/E2E_PLAYBOOK.md"
  ]
}
```

---

## Phase 121 完了基準

- [x] **OaF互換アダプタ作成** (`ops/oaf_adapter.py`)
- [x] **CREPE meta + サニティチェック** (`ops/crepe_extract.py`)
- [x] **CI検証ゲート強化** (`ci_verify_music_package.py`)
- [x] **Provenance刻印実装** (全AI適用記録)
- [x] **Bass F0抽出** (47,997フレーム、479.96秒)
- [x] **Piano OaF転写** (3,047 notes)
- [x] **全5パート生成** (Bass 1591 / Guitar 1263 / Piano 779 / Strings 608 / Drums 7247)
- [x] **Full Arrangement統合** (11,488 events)
- [ ] **MIDI変換** (`midi_writer.py`使用)
- [ ] **CI検証PASS** (`--strict`モード)
- [ ] **最終レポート作成**

---

## 参考

- **Phase 121改良完了レポート**: `KPI_GATE_PRODUCTION_COMPLETE.md`
- **CI検証スクリプト**: `scripts/ci_verify_music_package.py`
- **E2Eスクリプト**: `scripts/e2e_suno_arrangement.sh`
- **MIDI書き出し**: `scripts/midi_writer.py`（本線）、~~`scripts/json2midi.py`~~（禁止）

---

**最終更新**: 2025年11月8日  
**バージョン**: Phase 121  
**ステータス**: Production Ready
