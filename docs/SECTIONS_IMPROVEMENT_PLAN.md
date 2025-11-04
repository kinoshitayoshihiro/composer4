# Sections.json 改善プラン - スクリプトファイル活用調査

## 📋 実施内容
3つのフォルダ (`utilities/`, `scripts/`, `generator/`) を精査し、sections.json の精度向上とstage2で活用できるスクリプトを分類しました。

---

## 🎯 A) Sections.json 改善に**直接活用できる**ファイル

### 【高優先度】テンポ/QL換算の厳密化

#### 1. `utilities/tempo_from_mix.py` (257行)
**役割**: ミックスWAVからテンポカーブ/拍/downbeats推定  
**入力**: WAV + 複数ステム(任意)  
**出力**: `{"beats": [t0,t1,...], "downbeats": [d0,d1,...], "tempo_bpm": float}`

**活用方法**:
```python
# sections_from_audio.py に統合
from utilities.tempo_from_mix import extract_tempo_curve

y, sr = load_mix_from_stems(...)
tempo_map, beats, downbeats = extract_tempo_curve(y, sr)

# sections.json に追加
sections_json["tempo_map"] = [[bar, bpm], ...]  # rubato/rit 対応
sections_json["timesig"] = {"num": 4, "denom": 4}
```

**効果**:
- QL換算が厳密化（rubato/ritardando対応）
- Phase 28/32（マーカー/Export）の精度向上
- chordmap統一時のスナップ誤差削減

---

#### 2. `utilities/tempo_utils.py` + `tempo_curve.py` + `time_utils.py`
**役割**: テンポカーブの補間、秒↔QL相互変換  
**活用**: `tempo_from_mix.py` の結果を QL 単位に正確変換

```python
from utilities.tempo_utils import seconds_to_ql, ql_to_seconds
from utilities.tempo_curve import interpolate_tempo_at_bar

# 各バーの厳密なQL位置計算
for bar in sections:
    bar_ql = seconds_to_ql(bar_time_sec, tempo_map)
```

---

### 【中優先度】ハーモニー/キー推定

#### 3. `utilities/harmonic_utils.py` (170行)
**役割**: キー推定、ハーモニックノード選択  
**現状**: ギター用ハーモニクス関数が中心（`choose_harmonic`, `apply_harmonic_notation`）

**改良案**:
```python
# 新規関数を追加（バークロマから調推定）
def estimate_key_per_section(C_bars, section_boundaries):
    """セクション毎のキー候補を返す"""
    key_hints = []
    for i, (bar_start, bar_end) in enumerate(section_boundaries):
        C_section = C_bars[:, bar_start:bar_end].mean(axis=1)
        key = detect_key_from_chroma(C_section)  # Krumhansl-Schmuckler
        key_hints.append([bar_start, key])
    return key_hints

# sections.json に追加
sections_json["key_hint"] = [[0, "D"], [16, "G"], ...]
```

**効果**:
- Stage1 の local key prior が効く
- Phase 26/31（Hybrid Harmony/Voice-Leading）の精度向上
- 転調マーカー出力（Phase 32）が自動化

---

### 【低優先度】バリデーション/エラー検出

#### 4. `utilities/section_validator.py` (189行)
**役割**: sections.json の整合性チェック  
**検証項目**:
- `bar` が単調増加
- 1セクション最小長 ≥ 8 bars
- `energy` ∈ [0, 1]
- ラベルの妥当性（intro/verse/chorus/bridge/outro）

**活用**:
```python
from utilities.section_validator import validate_sections

# sections_from_audio.py の最後に追加
validate_sections(sections_json_path)
```

**効果**: 壊れたJSONの早期検出、"出して終わり"を防止

---

### 【任意】音声解析の補助

#### 5. `utilities/audio_analysis.py` (簡易)
**役割**: ピッチ抽出（YIN）、振幅エンベロープ、velocity変換  
**用途**: sections.json の検証/可視化（必須ではない）

---

## 🎵 B) Stage2 で**活用できる**ファイル

### 【即戦力】薄いアダプタで直結可能

#### 6. `utilities/accent_mapper.py` (164行)
**役割**: 歌詞/子音クラス/アンカーに基づくアクセント付与  
**入力**: `lyric_anchors.json`, heatmap  
**出力**: Velocity/Duration微調整

**使い所**: **Phase 23 Prosody**（既存）と完全一致  
**注意**: 既存Phase 23と排他的に使用（フラグで切替）

```python
# ops/stage2_phase23.py
if args.use_accent_mapper:
    from utilities.accent_mapper import AccentMapper
    mapper = AccentMapper(heatmap, global_settings)
    part = mapper.apply_accents(part, anchors)
```

---

#### 7. `utilities/articulation_mapper.py` (簡易)
**役割**: スタッカート/テヌート/フラム等の発音記号→ノート形状  
**使い所**: **Phase 17 Articulation** と一致  
**注意**: 既存Phase 17とどちらか片方のみ適用

---

#### 8. `utilities/breath_mask.py` (109行)
**役割**: ブレス/無声区間の"隙間"をマスク化  
**入力**: vocal WAV or `lyric_anchors.json`  
**出力**: boolean mask (低エネルギーフレーム)

**使い所**: **Phase 29 Vocal-Aware Ducking** のマスク生成器  
```python
from utilities.breath_mask import infer_breath_mask

mask = infer_breath_mask(vocal_wav, hop_ms=10.0)
# Velocity/Duration を mask が True の箇所で削減
```

**効果**: 語感の明瞭化、ボーカル被り低減

---

#### 9. `generator/bass_utils.py` (292行)
**役割**: ルート/5度/ウォーキングベース生成補助  
**関数**:
- `mirror_pitches()`: ボーカルメロディのミラーリング
- `get_approach_note()`: クロマティック/ダイアトニックアプローチ
- ルート/5度選択

**使い所**: **Stage1の仮ノート生成**（Phase 11前）  
```python
from generator.bass_utils import mirror_pitches, get_approach_note

# chordmap があれば最低限のベース下地を生成
bass_notes = generate_root_fifth_pattern(chordmap)
```

**効果**: ゼロノート救済（"保険の生成"）

---

### 【中優先度】タイミング/グルーヴ

#### 10. `utilities/timing_utils.py` (197行)
**役割**: push-pull, swing, テンポカーブ補間  
**使い所**: **Phase 28 Export Postprocess** の量子化

```python
from utilities.timing_utils import _combine_timing, interp_tempo

# マイクロタイミング適用
for note in part:
    timing_blend = _combine_timing(
        note.offset, beat_len_ql, 
        swing_ratio=0.6, push_pull_curve=[...]
    )
    note.offset += timing_blend.offset_ql
    note.velocity *= timing_blend.vel_scale
```

**効果**: 量子化が"音楽的"に、人力修正減少

---

#### 11. `utilities/timing_corrector.py`
**役割**: オフセット補正、量子化ミス修正  
**使い所**: Phase 28（Export Postprocess）と Phase 19/29（Groove微調整）

---

#### 12. `utilities/groove_sampler_v2.py` (3792行)
**役割**: n-gramドラムグルーヴ生成（Blake2bハッシュベース）  
**使い所**: **Phase 27 Style Adaptation** のグルーヴ候補事前抽出

```python
from utilities.groove_sampler_v2 import GrooveSampler

sampler = GrooveSampler(midi_loops_dir)
patterns = sampler.sample_groove(seed_pattern, bars=4)
```

**効果**: style名だけでなく中身（パターン集合）が豊かに

---

### 【大物】採用は要ハブ

#### 13. `utilities/arrangement_builder.py`
**役割**: 旧Modular Composerの中核（ロール横断配置・バランス）  
**条件**: 互換ハブ（compat layer）を1枚噛ませて標準化

**入力**: `chordmap_unified.json` + `sections.json` + `lyric_anchors.json` + `rhythm_library`  
**使い所**: Stage1で各ロールの"初期パート"を一括生成

**リスク**: Stage2（Phase 13-32）と機能重複  
**推奨**: 適用範囲を"初期ノート生成まで"に限定（Velocity/タイミングはStage2に任せる）

---

#### 14. `scripts/auto_pipeline_stage1.py` (251行)
**役割**: Stage1の完全自動パイプライン  
**フロー**:
1. `sections_from_audio.py` → sections.json
2. `stem_harmony_7th_v2.py` → chordmap.json
3. `chordmap_unify.py` → chordmap_unified.json
4. `anchors_from_vocal.py` → lyric_anchors.json
5. JSON統合 → modular_composer形式

**現状**: ラッパー的な役割  
**推奨**: `ops/generate_stage1_jsons.py` を呼ぶエイリアスとして残す（互換性維持）

---

### 【保留/研究素材】

#### 15. `utilities/bass_timbre_dataset.py`
音色/学習データ用。MIDI生成には直接不要（将来のサンプラー/音色選定用）

#### 16. `utilities/duv_apply.py`
Dynamics/Velocity/Duration変換（Phase 18/25/29と重複の可能性）  
採用時は既存Phaseを無効化して片方のみ適用

---

## 📐 C) sections_from_audio.py の改良方針

### 現状の問題点（ChatGPT診断）
```json
{
  "unit": "bar",
  "sections": [
    {"bar": 0, "label": "intro"},
    {"bar": 90, "label": "verse"},
    {"bar": 138, "label": "outro"}
  ],
  "energy": [[0, 0.3], [1, 0.35], ..., [150, 0.8]]
}
```

⚠️ **問題**:
- セクションが3つだけ（intro/verse/outro）
- コーラス/プリコーラス/ブリッジ等が無い
- 転調ヒント/tempo_map/timesig が無い

✅ **良い点**:
- スキーマ整合（unit:"bar" + sections + energy）
- エネルギー系列は全バー分入っている

---

### 改良ロードマップ

#### **Phase 1: テンポ/QL換算の厳密化** ⭐ 最優先
```python
# sections_from_audio.py に追加
from utilities.tempo_from_mix import extract_tempo_curve
from utilities.tempo_utils import seconds_to_ql

y, sr = _load_mix_from_stems(...)
tempo_map, beats, downbeats = extract_tempo_curve(y, sr)

# sections.json に追加
sections_json["tempo_map"] = [[bar, bpm], ...]
sections_json["timesig"] = {"num": 4, "denom": 4}
```

**効果**: QL換算厳密化、Phase 28/32精度向上

---

#### **Phase 2: セクション細分化** ⭐ 高優先
```python
# 現在の detect_sections() を改良
def detect_sections_v2(y, sr, *, min_bars=8, target_sections=7):
    """目標: intro/verse/pre_chorus/chorus/bridge/outro の5-7区間"""
    
    # 1) エネルギーカーブから chorus 候補抽出（ピーク群）
    energy_peaks = find_energy_peaks(rms_bar, prominence=0.3)
    
    # 2) クロマ類似度から反復パターン検出（chorus = 繰り返し）
    repetition_map = detect_repetitive_sections(C_bars)
    
    # 3) pre_chorus = chorus直前8-16小節、bridge = 谷
    sections = []
    for peak in energy_peaks:
        if peak - 8 >= 0:
            sections.append((peak - 8, "pre_chorus"))
        sections.append((peak, "chorus"))
    
    # 4) 最小長8小節を強制、短すぎる区間はマージ
    sections = merge_short_sections(sections, min_bars=8)
    
    return sections
```

**効果**: セクション数が3→7に増加、音楽的に自然な境界

---

#### **Phase 3: キーヒント付与** ⭐ 中優先
```python
from utilities.harmonic_utils import estimate_key_per_section

# セクション毎にキー推定（Krumhansl-Schmuckler法）
key_hints = estimate_key_per_section(C_bars, section_boundaries)

# sections.json に追加
sections_json["key_hint"] = [[0, "D"], [32, "G"], [64, "A"], ...]
```

**効果**: 転調対応、Phase 26/31（Harmony/Voice-Leading）精度向上

---

#### **Phase 4: バリデーション** ⭐ 低優先
```python
from utilities.section_validator import validate_sections

# 出力前に健全性チェック
validate_sections(sections_json)
```

**効果**: 壊れたJSONの早期検出

---

## 🎯 D) 実装の優先順位

### 【IMMEDIATE - 今週中】
1. ✅ `tempo_from_mix.py` を `sections_from_audio.py` に統合
2. ✅ `tempo_map` と `timesig` を sections.json に追加
3. ✅ セクション細分化ロジック実装（5-7区間目標）

### 【SHORT-TERM - 2週間以内】
4. ✅ `harmonic_utils.py` にキー推定関数追加
5. ✅ `key_hint` を sections.json に追加
6. ✅ `section_validator.py` でバリデーション

### 【MID-TERM - 1ヶ月以内】
7. ⏳ Stage2 で `accent_mapper.py` / `breath_mask.py` / `bass_utils.py` 統合
8. ⏳ `timing_utils.py` を Phase 28（Export）に統合
9. ⏳ `groove_sampler_v2.py` を Phase 27（Style Adaptation）に統合

### 【LONG-TERM - 研究/検証】
10. ⏳ `arrangement_builder.py` の互換ハブ作成
11. ⏳ `duv_apply.py` の Phase 18/25 統合検討

---

## 📊 E) 期待される効果

### Sections.json の品質向上
| 項目 | 現在 | 改良後 | 効果 |
|------|------|--------|------|
| セクション数 | 3 (intro/verse/outro) | 5-7 (intro/verse/pre_chorus/chorus/bridge/outro) | セクション境界トリガ有効化 |
| テンポ情報 | 無し | `tempo_map` 付き | QL換算厳密化、rubato対応 |
| キー情報 | 無し | `key_hint` 付き | 転調対応、ハーモニー精度向上 |
| 最小長保証 | 無し | 8小節強制 | 細切れ防止 |
| バリデーション | 無し | 自動チェック | 壊れたJSON検出 |

### Stage2 パイプライン強化
| Phase | 統合ファイル | 効果 |
|-------|-------------|------|
| Phase 11 (初期生成) | `bass_utils.py` | ゼロノート救済 |
| Phase 17 (Articulation) | `articulation_mapper.py` | 発音記号→ノート形状 |
| Phase 23 (Prosody) | `accent_mapper.py` | アクセント付与 |
| Phase 27 (Style) | `groove_sampler_v2.py` | グルーヴ語彙拡充 |
| Phase 28 (Export) | `timing_utils.py` | 音楽的量子化 |
| Phase 29 (Ducking) | `breath_mask.py` | ボーカル被り低減 |

---

## 🚀 F) Next Action (今すぐ実施)

### 1. sections_from_audio.py の改良パッチ適用
```bash
# 1) tempo_from_mix.py の統合
# 2) セクション細分化ロジック実装
# 3) key_hint 生成機能追加
```

### 2. テスト実行
```bash
python ops/sections_from_audio.py \
  --stems ujam \
  --out sections_improved.json \
  --exclude vocals \
  --min-bars 8 \
  --max-sections 7
```

### 3. 品質確認
```bash
# セクション数が5-7個に増えているか
# tempo_map, key_hint が追加されているか
cat sections_improved.json | jq '.sections | length'
cat sections_improved.json | jq '.tempo_map'
cat sections_improved.json | jq '.key_hint'
```

---

## 📝 G) ファイル分類一覧（完全版）

### Sections.json 改善 (直接活用)
- ✅ `utilities/tempo_from_mix.py` - テンポカーブ推定
- ✅ `utilities/tempo_utils.py` - テンポ補間
- ✅ `utilities/tempo_curve.py` - テンポカーブ処理
- ✅ `utilities/time_utils.py` - 秒↔QL変換
- ✅ `utilities/harmonic_utils.py` - キー推定（要改良）
- ✅ `utilities/section_validator.py` - バリデーション
- ◻️ `utilities/audio_analysis.py` - 任意（検証用）

### Stage2 即戦力 (薄いアダプタで直結)
- ✅ `utilities/accent_mapper.py` - Phase 23
- ✅ `utilities/articulation_mapper.py` - Phase 17
- ✅ `utilities/breath_mask.py` - Phase 29
- ✅ `generator/bass_utils.py` - Phase 11
- ✅ `utilities/timing_utils.py` - Phase 28
- ✅ `utilities/timing_corrector.py` - Phase 28
- ✅ `utilities/groove_sampler_v2.py` - Phase 27

### Stage2 大物 (要互換ハブ)
- 🧱 `utilities/arrangement_builder.py` - Stage1初期生成
- 🧱 `scripts/auto_pipeline_stage1.py` - ラッパー

### 保留/研究素材
- 🗃 `utilities/bass_timbre_dataset.py` - 音色データ
- 🗃 `utilities/duv_apply.py` - Dynamics/Velocity/Duration

---

## 🎯 結論

**即座に実施すべき最小改良**:
1. `tempo_from_mix.py` 統合 → `tempo_map` 付与
2. セクション細分化（3→7区間）
3. `harmonic_utils.py` でキー推定 → `key_hint` 付与

**効果**: sections.json が「最低限使える」から「Stage2が賢く動く」レベルに向上

**Stage2 統合**: accent_mapper, breath_mask, bass_utils を優先的に統合

この改良により、ChatGPTが指摘した「粒度が粗い」「フィル/ダッキング/グルーヴ最適化が出し切れない」問題が解決されます。

---

## 🎁 H) ChatGPT未指摘の追加発見ファイル（高価値）

### 【Sections.json 改善】構造解析・境界検出

#### 17. `utilities/peak_extractor.py` ⭐⭐⭐
**役割**: RMSピーク時刻抽出（セクション境界候補検出）  
**入力**: WAV  
**出力**: `[t0, t1, ...]` (秒)

**機能**:
- RMS smoothing（時間ウィンドウ平滑化）
- 閾値ベース検出（dB単位）
- 最小距離制約（30ms）

**活用方法**:
```python
from utilities.peak_extractor import extract_peaks, PeakExtractorConfig

cfg = PeakExtractorConfig(
    threshold_db=-20.0,
    min_distance_ms=30.0,
    rms_smooth_ms=20.0
)
peaks = extract_peaks(wav_path, cfg)

# sections_from_audio.py に統合
# ピーク群 = セクション境界候補（エネルギー変化点）
section_boundaries = detect_sections_from_peaks(peaks, bars)
```

**効果**:
- エネルギー変化点を精密検出（現在のlibrosa.beat.beat_trackより高精度）
- セクション境界の"音楽的妥当性"向上
- コーラス開始（エネルギー急上昇）を正確に捉える

---

#### 18. `utilities/onset_heatmap.py` ⭐⭐
**役割**: MIDIからオンセットヒートマップ生成  
**入力**: MIDI  
**出力**: `{"grid_index": count}` (小節内16分割)

**機能**:
- 小節内オンセット密度の可視化
- グリッド解像度可変（デフォルト16分割）
- アクセントマッピング用

**活用方法**:
```python
from utilities.onset_heatmap import build_heatmap, load_heatmap

# 既存MIDIから学習
heatmap = build_heatmap(reference_midi, resolution=16)

# accent_mapper と組み合わせて Phase 23 で使用
from utilities.accent_mapper import AccentMapper
mapper = AccentMapper(heatmap, global_settings)
```

**効果**:
- リファレンスMIDIのリズムパターン学習
- アクセント位置の統計的推定
- Stage2 Phase 23（Prosody）の精度向上

---

#### 19. `scripts/segment_phrase.py` (550行) ⭐⭐⭐
**役割**: Transformerベースのフレーズ境界検出  
**モデル**: BiLSTM + TCN + CRF  
**入力**: MIDI (pitch_class, velocity, duration)  
**出力**: フレーズ境界位置

**機能**:
- ML境界検出（学習済みチェックポイント使用）
- ピッチ範囲フィルタリング
- 楽器別処理（regex/index指定）

**活用方法**:
```python
from scripts.segment_phrase import segment_bytes

# MIDIからフレーズ境界検出
boundaries = segment_bytes(
    midi_bytes, 
    threshold=0.5,
    inst_regex="piano",
    pitch_range=(48, 84)
)

# sections_from_audio.py に統合
# → フレーズ境界 ≈ セクション境界（音楽的な句読点）
```

**効果**:
- 音楽理論に基づくセクション分割（ヒューリスティクスより高精度）
- フレーズ構造を考慮したセクション生成
- MLベースで楽曲パターンを学習

---

#### 20. `utilities/peak_synchroniser.py` ⭐⭐
**役割**: ドラムイベントと子音ピークの同期  
**入力**: ピーク時刻 + ベースイベント  
**出力**: 同期済みドラムイベント

**機能**:
- Lag補正（±10ms）
- 最小距離制約（0.25拍）
- Sustain検出（120ms閾値）
- 優先度ベース配置（kick > snare > hat）

**活用方法**:
```python
from utilities.peak_synchroniser import PeakSynchroniser
from utilities.peak_extractor import extract_peaks

# ボーカルWAVから子音ピーク抽出
peaks = extract_peaks(vocal_wav)

# ドラムパターンを同期
synced_events = PeakSynchroniser.sync_events(
    peaks, base_drum_pattern,
    tempo_bpm=120,
    lag_ms=10.0
)
```

**効果**:
- ボーカルとドラムの一体感向上
- Stage2 Phase 29（Vocal-Aware Ducking）と相性良い
- リアルタイム生成時のタイト感

---

### 【Stage2 強化】感情・スタイル・フレーズ処理

#### 21. `utilities/emotion_arranger.py` (166行) ⭐⭐
**役割**: 感情プロファイルに基づくパターン選択  
**対応**: Bass, Piano, Guitar  
**入力**: chordmap + emotion_profile  
**出力**: セクション毎のパターンキー

**機能**:
- 感情→パターン自動選択
- オクターブ優先度、長さ指定
- 3パート統合アレンジ

**使い所**: **Stage1〜Stage2 橋渡し**
```python
from utilities.emotion_arranger import generate_full_arrangement

arrangement = generate_full_arrangement(
    chordmap_path,
    rhythm_library_path,
    emotion_profile_path
)

# セクション毎に最適パターンを自動選択
# → Stage2 Phase 14-16（パート生成）の入力に
```

**効果**:
- 感情一貫性の自動維持
- 手動パターン指定不要
- 複数パート間のバランス最適化

---

#### 22. `utilities/phrase_filter.py` ⭐⭐
**役割**: フレーズクラスタリング・重複除去  
**手法**: 3-gram類似度 + HDBSCAN  
**入力**: フレーズイベント列  
**出力**: 保持マスク（重複は False）

**機能**:
- CountVectorizer（3-gram）
- Cosine類似度 + HDBSCAN
- フォールバック: Jaccard係数（sklearn不要時）

**使い所**: **Stage2 Phase 27（Style Adaptation）後の重複削減**
```python
from utilities.phrase_filter import cluster_phrases

# 生成された複数フレーズから代表選択
keep_mask = cluster_phrases(phrase_events_list, n=4)
unique_phrases = [p for p, k in zip(phrases, keep_mask) if k]
```

**効果**:
- グルーヴパターンの多様性確保
- 同じフレーズの繰り返し防止
- クラスタ代表のみ採用

---

#### 23. `utilities/emotion_profile_loader.py` + `rhythm_library_loader.py` ⭐⭐
**役割**: 感情プロファイル・リズムライブラリのYAML/JSON読み込み  
**バリデーション**: Pydantic（rhythm_library_loader）

**emotion_profile.yaml 例**:
```yaml
energetic:
  octave_pref: 3
  length_beats: 2.0
  velocity_range: [90, 110]
calm:
  octave_pref: 2
  length_beats: 4.0
  velocity_range: [50, 70]
```

**rhythm_library.yaml 例**:
```yaml
bass_patterns:
  walking:
    events:
      - {beat: 0.0, duration: 0.5, type: "root"}
      - {beat: 0.5, duration: 0.5, type: "fifth"}
```

**使い所**: **全Stage2パート生成の基盤**
- Phase 14-16（Bass/Piano/Guitar生成）
- emotion_arranger と組み合わせて自動化

**効果**:
- 設定の外部化（コード変更不要）
- プロダクション運用での柔軟性
- A/Bテスト容易化

---

#### 24. `utilities/style_db.py` ⭐
**役割**: スタイルカーブDB（velocity/CC）  
**ロード**: 環境変数 `STYLE_DB_PATH`

**style_db.yaml 例**:
```yaml
soft:
  velocity: [40, 60, 80]
  cc: [50, 80, 100]
hard:
  velocity: [80, 100, 120]
  cc: [90, 110, 127]
```

**使い所**: **Phase 27（Style Adaptation）のカーブ取得**
```python
from utilities.style_db import get_style_curve

curve = get_style_curve("soft")
velocities = curve["velocity"]  # [40, 60, 80]
```

**効果**:
- スタイル定義の一元管理
- 実行時切替（env変数）
- 複数プロジェクト共有

---

#### 25. `utilities/humanizer.py` (803行) ⭐⭐⭐
**役割**: 総合ヒューマナイズ（velocity/timing/swing/envelope）  
**拡張**: Cython最適化対応

**機能**:
- Velocity histogram適用（プロファイル指定）
- Swing offset（8分/16分）
- Ghost note jitter（再エクスポート from ghost_jitter）
- Envelope curve（ADSR風）
- CC11/Aftertouch対応

**使い所**: **Phase 28（Export Postprocess）の最終仕上げ**
```python
from utilities.humanizer import apply_velocity_histogram_profile, swing_offset

# Velocity histogram適用
apply_velocity_histogram_profile(part, profile_name="soft")

# Swing適用
swing_offset(notes, swing_ratio=0.6, swing_type="eighth")
```

**効果**:
- 機械的なMIDI → 人間的な演奏
- プロファイル切替で多様性
- Cython高速化（オプション）

---

### 【その他】音響解析・音色

#### 26. `utilities/consonant_extract.py` ⭐
**役割**: 子音抽出（Essentia SpectralFlux）  
**用途**: ボーカル分析、ピーク検出の高精度版

#### 27. `utilities/tone_shaper.py` ⭐
**役割**: MFCC→プリセット推定（KNN）  
**用途**: 音色自動選択（Stage2 音色マッピング）

---

## 📊 I) 追加ファイルの効果マトリクス

| ファイル | 用途 | 優先度 | 統合先 | 期待効果 |
|---------|------|--------|--------|---------|
| `peak_extractor.py` | セクション境界検出 | ⭐⭐⭐ | sections_from_audio.py | エネルギー変化点の精密検出 |
| `segment_phrase.py` | ML境界検出 | ⭐⭐⭐ | sections_from_audio.py | 音楽理論に基づくセクション分割 |
| `onset_heatmap.py` | オンセット密度 | ⭐⭐ | Phase 23 | アクセント位置の統計推定 |
| `peak_synchroniser.py` | ドラム同期 | ⭐⭐ | Phase 29 | ボーカル・ドラム一体感 |
| `emotion_arranger.py` | 感情パターン選択 | ⭐⭐ | Stage1→Stage2 | 感情一貫性自動維持 |
| `phrase_filter.py` | 重複除去 | ⭐⭐ | Phase 27 | グルーヴ多様性確保 |
| `emotion_profile_loader.py` | 感情設定読込 | ⭐⭐ | 全Phase | 外部設定化 |
| `rhythm_library_loader.py` | リズムDB読込 | ⭐⭐ | Phase 14-16 | パターン外部化 |
| `style_db.py` | スタイルカーブDB | ⭐ | Phase 27 | スタイル一元管理 |
| `humanizer.py` | 総合ヒューマナイズ | ⭐⭐⭐ | Phase 28 | 人間的演奏感 |

---

## 🚀 J) 改訂版実装プラン

### Phase 1: Sections.json 高精度化 ⭐⭐⭐
```python
# sections_from_audio.py 改良
from utilities.peak_extractor import extract_peaks
from utilities.tempo_from_mix import extract_tempo_curve
from scripts.segment_phrase import segment_bytes

# 1) テンポカーブ取得
tempo_map, beats = extract_tempo_curve(mix_wav, sr)

# 2) エネルギーピーク検出（境界候補）
peaks = extract_peaks(mix_wav, threshold_db=-20.0)

# 3) フレーズ境界検出（ML）- 既存MIDIがあれば
if reference_midi:
    phrase_bounds = segment_bytes(midi_bytes, threshold=0.5)
    # ピークとフレーズ境界の交点 → 高信頼度境界

# 4) セクション細分化（5-7区間目標）
sections = refine_sections(peaks, phrase_bounds, energy_curve)

# 5) キー推定 + tempo_map + timesig 付与
sections_json = {
    "unit": "bar",
    "sections": sections,
    "energy": energy_bar,
    "tempo_map": tempo_map,
    "timesig": {"num": 4, "denom": 4},
    "key_hint": key_hints
}
```

### Phase 2: Stage2 感情・スタイル統合 ⭐⭐
```python
# Stage1 → Stage2 橋渡し
from utilities.emotion_arranger import generate_full_arrangement

arrangement = generate_full_arrangement(
    chordmap_unified_path,
    rhythm_library_path,
    emotion_profile_path
)

# Phase 14-16 で使用
bass_pattern = arrangement["intro"]["bass_pattern_key"]
piano_pattern = arrangement["intro"]["piano_pattern_key"]
```

### Phase 3: ヒューマナイズ強化 ⭐⭐⭐
```python
# Phase 28 Export Postprocess
from utilities.humanizer import apply_velocity_histogram_profile, swing_offset

# 1) Velocity histogram
apply_velocity_histogram_profile(part, "soft")

# 2) Swing
swing_offset(part.recurse().notes, swing_ratio=0.6)

# 3) Ghost jitter (既存)
apply_ghost_jitter(part, density=0.3)
```

### Phase 4: フレーズ多様性確保 ⭐⭐
```python
# Phase 27 Style Adaptation 後
from utilities.phrase_filter import cluster_phrases

generated_phrases = [...]  # 複数候補生成
keep_mask = cluster_phrases(generated_phrases, n=4)
final_phrases = [p for p, k in zip(generated_phrases, keep_mask) if k]
```

---

## 📈 K) 総合効果予測

### Sections.json 品質（改訂版）
| 項目 | 現在 | Phase 1後 | 効果 |
|------|------|-----------|------|
| セクション数 | 3 | 7-9 | ピーク検出+ML境界で高精度 |
| 境界精度 | ヒューリスティック | ML+エネルギー複合 | 音楽理論準拠 |
| テンポ情報 | 無し | tempo_map + timesig | rubato完全対応 |
| キー情報 | 無し | key_hint | 転調自動検出 |

### Stage2 パイプライン強化（改訂版）
| Phase | 追加ファイル | 効果 |
|-------|-------------|------|
| Phase 11 | bass_utils.py | ゼロノート救済 |
| Phase 14-16 | emotion_arranger + rhythm_library_loader | パターン自動選択 |
| Phase 17 | articulation_mapper.py | 発音記号適用 |
| Phase 23 | accent_mapper + onset_heatmap | アクセント統計推定 |
| Phase 27 | groove_sampler_v2 + phrase_filter + style_db | グルーヴ多様化 |
| Phase 28 | humanizer + timing_utils | 人間的演奏感 |
| Phase 29 | breath_mask + peak_synchroniser | ボーカル一体感 |

---

## 🎯 結論（改訂版）

**ChatGPT未指摘の高価値ファイル11個を発見**:
- セクション検出: `peak_extractor`, `segment_phrase`
- 感情・スタイル: `emotion_arranger`, `phrase_filter`, `style_db`
- ヒューマナイズ: `humanizer` (803行の総合ツール)
- 設定管理: `emotion_profile_loader`, `rhythm_library_loader`

**最優先実装**:
1. `peak_extractor` + `segment_phrase` → sections_from_audio.py
2. `humanizer` → Phase 28
3. `emotion_arranger` → Stage1/Stage2橋渡し

**期待される追加効果**:
- セクション境界精度: ヒューリスティック → ML+物理解析の複合判定
- 感情一貫性: 手動 → 自動維持
- 演奏感: 機械的 → 人間的（velocity histogram + swing + jitter）
- グルーヴ多様性: 重複あり → クラスタリング重複除去

これらを統合することで、「ChatGPTが指摘した改良」+「未指摘の高度機能」の両方が実現され、world-classのStage1/Stage2パイプラインが完成します。
