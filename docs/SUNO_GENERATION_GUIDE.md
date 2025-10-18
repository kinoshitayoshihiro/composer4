# Suno AI Generation Guide

**目的:** Suno AIでWAV生成 → MIDI変換 → Pickle統合の段階的ワークフロー

**戦略:** 数曲ごとにPickle追加、プロンプト最適化を繰り返し、徐々にデータセットを育てる

---

## 🎯 生成戦略

### Phase 1: Proof of Concept (各奏法5-10曲)
- **目標:** Suno AIで狙った奏法が生成可能か検証
- **対象:** Critical Gaps優先（guitar_strum, strings_legato）
- **評価:** MIDI変換品質、Stage2スコア

### Phase 2: Incremental Growth (各奏法50-100曲)
- **目標:** 高品質プロンプトを確立、Pickle段階的拡張
- **対象:** Top 5 Gaps（guitar_strum, strings_legato, guitar_arpeggio, bass_pick, strings_spiccato）
- **評価:** Stage2合格率、Real Data+5%閾値達成

### Phase 3: Full Production (各奏法100-500曲)
- **目標:** Manifest残り不足を完全補完
- **対象:** 全奏法（25パターン）
- **評価:** 統合後のGap解消確認

---

## 🎸 Suno AI Prompt Templates

### 1. Guitar – Strum（最優先: 1,554件不足）

#### A. mid（ポップ定番・8分ストラム）

**Style:**
```
acoustic pop, clean strum, down-up 8th, bright, dry room, tight timing
```

**Prompt:**
```
Instrumental only. Acoustic guitar ONLY, no drums, no bass, no keys.
Tempo {TEMPO_BPM} BPM, 4/4, key {KEY}.
Play a steady down-up 8th-note strum with clear dynamics and minimal reverb.
Chord progression: {PROG} (4 bars), loop it for {LOOPS} cycles.
Keep voicings open-position pop chords; avoid arpeggios and fingerstyle.
Humanization subtle (±10–15 ms), velocity shaped for {EMOTION}.
Goal: clean, isolated strum texture that converts well to MIDI.
```

**パラメータ例:**
- TEMPO_BPM: 100, 110, 120, 130 (mid range)
- KEY: C, G, D, A, E (guitar-friendly keys)
- PROG: `C - Am - F - G`, `G - Em - C - D`, `D - Bm - G - A`
- LOOPS: 2-4
- EMOTION: neutral, happy, calm

**Expected Output:** 8-16秒のクリーンなストラムループ

---

#### B. slow（バラード・4分ストラム）

**Style:**
```
acoustic ballad, soft strum, quarter notes, warm, slight reverb, relaxed timing
```

**Prompt:**
```
Instrumental only. Acoustic guitar ONLY.
Tempo {TEMPO_BPM} BPM, 4/4, key {KEY}.
Play a gentle quarter-note strum pattern with soft dynamics and subtle reverb.
Chord progression: {PROG} (4 bars), repeat {LOOPS} times.
Use fingerstyle-influenced strum (thumb on bass, light brush on treble).
Timing relaxed (±15–25 ms humanization), velocity shaped for melancholic emotion.
Goal: warm, isolated ballad texture for MIDI conversion.
```

**パラメータ例:**
- TEMPO_BPM: 65, 75, 85 (slow range)
- KEY: Am, Em, Dm (minor keys for ballad)
- PROG: `Am - F - C - G`, `Em - C - G - D`
- LOOPS: 2-3
- EMOTION: sad, melancholic, calm

---

#### C. fast（ロック・16分ストラム）

**Style:**
```
rock, aggressive strum, 16th notes, bright attack, tight timing, no reverb
```

**Prompt:**
```
Instrumental only. Electric guitar ONLY, clean tone.
Tempo {TEMPO_BPM} BPM, 4/4, key {KEY}.
Play a driving 16th-note strum pattern with sharp attack and tight timing.
Chord progression: {PROG} (2 bars), loop {LOOPS} cycles.
Use power chord voicings (root + 5th), avoid full open chords.
Humanization minimal (±5 ms), velocity consistent and energetic.
Goal: percussive, isolated rock strum for MIDI.
```

**パラメータ例:**
- TEMPO_BPM: 140, 150, 160, 170 (fast range)
- KEY: E, A, D (power chord keys)
- PROG: `E5 - A5 - B5 - E5`, `A5 - D5 - E5 - A5`
- LOOPS: 3-5
- EMOTION: energetic, aggressive

---

### 2. Strings – Legato（最優先: 1,117件不足）

#### A. slow（長音・滑らかな移行）

**Style:**
```
string quartet, legato, slow sustained, warm hall reverb, smooth bowing
```

**Prompt:**
```
Instrumental only. String quartet ONLY (2 violins, viola, cello).
Tempo {TEMPO_BPM} BPM, 4/4, key {KEY}.
Play a sustained legato passage with smooth bow transitions (no staccato).
Chord progression: {PROG} (4 bars), hold each chord for 2 beats minimum.
Use full legato bowing (note overlap >90%), rich vibrato on sustained notes.
Humanization moderate (±20 ms), soft dynamics throughout.
Goal: smooth, isolated legato texture for MIDI conversion.
```

**パラメータ例:**
- TEMPO_BPM: 60, 70, 80 (slow range)
- KEY: D, G, C (string-friendly keys)
- PROG: `Dm - Am - F - C`, `G - Em - C - D`
- EMOTION: melancholic, serene, romantic

---

#### B. mid（中庸・表現豊かなレガート）

**Style:**
```
chamber strings, expressive legato, moderate tempo, warm room, dynamic swells
```

**Prompt:**
```
Instrumental only. String ensemble ONLY (violin section, viola, cello).
Tempo {TEMPO_BPM} BPM, 4/4, key {KEY}.
Play an expressive legato melody with dynamic swells and smooth phrasing.
Melodic line: {MELODY} (8 bars), no staccato or detached notes.
Use full legato technique (note overlap 80-100%), add crescendos/diminuendos.
Humanization natural (±15 ms), velocity ranges 40-80 for expression.
Goal: expressive, isolated legato for MIDI.
```

**パラメータ例:**
- TEMPO_BPM: 90, 100, 110 (mid range)
- KEY: A, E, B♭ (expressive keys)
- MELODY: Simple ascending/descending scales with neighbor tones
- EMOTION: romantic, hopeful, tender

---

### 3. Guitar – Arpeggio（不足: 1,007件）

**Style:**
```
fingerstyle guitar, arpeggio, 16th notes, clean tone, minimal reverb
```

**Prompt:**
```
Instrumental only. Acoustic guitar ONLY, fingerstyle technique.
Tempo {TEMPO_BPM} BPM, 4/4, key {KEY}.
Play a rolling 16th-note arpeggio pattern (thumb alternating with fingers).
Chord progression: {PROG} (4 bars), arpeggio each chord consistently.
Pattern: bass note (thumb) → 3rd → 5th → octave/melody note, loop smoothly.
Humanization subtle (±10 ms), velocity variance for natural fingerpicking.
Goal: clean, isolated arpeggio texture for MIDI.
```

**パラメータ例:**
- TEMPO_BPM: 100, 110, 120 (mid range)
- KEY: C, G, D, A
- PROG: `C - G/B - Am - F`, `G - D/F# - Em - C`
- EMOTION: neutral, calm, hopeful

---

### 4. Bass – Pick（不足: 900件）

**Style:**
```
electric bass, picked, 8th notes, tight attack, no slap, dry mix
```

**Prompt:**
```
Instrumental only. Electric bass ONLY, picked technique (no slap).
Tempo {TEMPO_BPM} BPM, 4/4, key {KEY}.
Play a steady 8th-note picked bassline with tight attack and consistent timing.
Root-5th pattern: {PATTERN} (4 bars), follow chord progression {PROG}.
Use picked technique (sharp attack, short sustain), no legato or hammer-ons.
Humanization minimal (±8 ms), velocity 70-85 for consistency.
Goal: clean, isolated picked bass for MIDI.
```

**パラメータ例:**
- TEMPO_BPM: 110, 120, 130 (mid-fast range)
- KEY: E, A, D (bass-friendly keys)
- PATTERN: `Root-Root-5th-Root`, `Root-5th-Octave-5th`
- PROG: `E - A - B - E`, `A - D - E - A`
- EMOTION: energetic, neutral

---

### 5. Strings – Spiccato（不足: 600件）

**Style:**
```
string ensemble, spiccato, bouncing bow, staccato, bright attack, dry room
```

**Prompt:**
```
Instrumental only. String ensemble ONLY (violins, viola, cello).
Tempo {TEMPO_BPM} BPM, 4/4, key {KEY}.
Play a spiccato (bouncing bow) pattern with short, detached notes.
Melodic rhythm: {RHYTHM} (4 bars), each note bounces cleanly off the string.
Articulation: sharp attack, quick release (note duration 20-30% of beat).
Humanization moderate (±12 ms), velocity 65-80 for brightness.
Goal: crisp, isolated spiccato texture for MIDI.
```

**パラメータ例:**
- TEMPO_BPM: 100, 110, 120 (mid range)
- KEY: D, A, G
- RHYTHM: 8th note patterns with rests
- EMOTION: playful, bright, energetic

---

## 🔄 WAV → MIDI 変換パイプライン

### Phase 1: Suno AI生成（WAV取得）

```bash
# 手動実行（Suno AI Web UI）
1. Prompt入力（上記テンプレート使用）
2. Style設定
3. 生成実行
4. WAVダウンロード → data/suno_wav/guitar_strum_mid/

# ファイル命名規則
{instrument}_{technique}_{tempo}_{key}_{id}.wav
例: guitar_strum_100_C_001.wav
```

---

### Phase 2: WAV → MIDI変換

**スクリプト:** `scripts/suno_wav_to_midi.py`（新規作成）

```bash
python scripts/suno_wav_to_midi.py \
  --input-dir data/suno_wav/guitar_strum_mid \
  --output-dir data/suno_midi/guitar_strum_mid \
  --instrument guitar \
  --technique strum \
  --method ensemble  # ensemble/basic/demucs
```

**変換メソッド:**
- **basic**: basic-pitch（シンプル、速い）
- **ensemble**: 複数モデル投票（高精度、遅い）
- **demucs**: Demucs前処理 + basic-pitch（ノイズ除去）

---

### Phase 3: MIDI品質検証 & Stage2スコアリング

```bash
# Stage1 クリーニング
python scripts/clean_midi.py \
  --input-dir data/suno_midi/guitar_strum_mid \
  --output-dir data/suno_clean/guitar_strum_mid \
  --quarantine-dir data/suno_quarantine/guitar_strum_mid

# Stage2 スコアリング
python scripts/test_instrument_metrics.py \
  --instrument guitar \
  --input-dir data/suno_clean/guitar_strum_mid \
  --config configs/lamda/guitar_stage2.yaml \
  --output-json output/suno_results/guitar_strum_mid.json

# 合格率確認
python scripts/analyze_stage2_results.py \
  output/suno_results/guitar_strum_mid.json \
  --threshold 45.0  # Real+5%
```

**品質ゲート:**
- Guitar: 45.0% (Real: 40.0 + 5%)
- Strings: 50.0% (Real: 45.0 + 5%)
- Bass: 45.0% (Real: 40.0 + 5%)
- Piano: 50.0% (Real: 45.0 + 5%)

**判定:**
- 合格率 ≥70% → Phase 4へ
- 合格率 <70% → プロンプト調整して再生成

---

### Phase 4: Pickle統合（段階的追加）

**スクリプト:** `scripts/append_to_pickle_shard.py`（新規作成）

```bash
# 既存Pickleに追加（Resume対応）
python scripts/append_to_pickle_shard.py \
  --input-dir data/suno_clean/guitar_strum_mid \
  --pickle-dir data/shards/hybrid \
  --instrument guitar \
  --technique strum \
  --source suno \
  --shard-size 5000 \
  --resume
```

**動作:**
1. 最新shardを読み込み（例: `guitar_shard_00003.pkl`）
2. 現在のバッファサイズ確認（例: 4,850件）
3. 新規データ追加（150件 → 5,000件到達）
4. 自動flush → `guitar_shard_00004.pkl`作成
5. 残りは次shardへ

**メタデータ:**
```python
{
    "instrument": "guitar",
    "technique": "strum",
    "tempo": 110,
    "key": "C",
    "source": "suno",  # real/suno/external
    "suno_prompt_id": "guitar_strum_mid_001",
    "conversion_method": "ensemble",
    "stage2_score": 48.5,
    "stage2_passed": True,
    "lamda": {
        "arpeggio_quality": 0.25,
        "chord_coherence": 0.62,
        "strumming_pattern": 0.58,
        # ...
    }
}
```

---

## 🔧 実装スクリプト

### 1. WAV → MIDI変換（scripts/suno_wav_to_midi.py）

**機能:**
- basic-pitch, Demucs, ensemble voting対応
- MIDI後処理（quantize, velocity normalization）
- メタデータJSON出力

**使用方法:**
```bash
# Basic変換（速い）
python scripts/suno_wav_to_midi.py \
  --input-dir data/suno_wav/guitar_strum_mid \
  --output-dir data/suno_midi/guitar_strum_mid \
  --method basic

# Ensemble変換（高精度）
python scripts/suno_wav_to_midi.py \
  --input-dir data/suno_wav/guitar_strum_mid \
  --output-dir data/suno_midi/guitar_strum_mid \
  --method ensemble \
  --num-models 3  # basic-pitch, MT3, BTC
```

---

### 2. Pickle段階的追加（scripts/append_to_pickle_shard.py）

**機能:**
- 既存shard読み込み & 追加
- Resume対応（重複回避）
- 自動shard分割（5,000件/shard）
- Source tracking（real/suno/external）

**使用方法:**
```bash
python scripts/append_to_pickle_shard.py \
  --input-dir data/suno_clean/guitar_strum_mid \
  --pickle-dir data/shards/hybrid \
  --instrument guitar \
  --technique strum \
  --source suno \
  --resume
```

---

### 3. Suno生成バッチスクリプト（scripts/generate_suno_batch.py）

**機能:**
- プロンプトテンプレート読み込み
- パラメータ組み合わせ生成
- Suno API実行（将来: 自動化）
- 現状: プロンプトリスト出力（手動コピペ用）

**使用方法:**
```bash
python scripts/generate_suno_batch.py \
  --template templates/guitar_strum_mid.yaml \
  --output suno_prompts/guitar_strum_batch_001.txt \
  --num-variations 10

# 出力例: 10パターンのプロンプト（手動でSuno UIにコピペ）
```

---

## 📊 段階的Pickle成長プロセス

### Iteration 1（PoC: 5曲 × 5奏法 = 25曲）

```bash
# 1. Suno生成（手動）
# guitar_strum: 5曲 (100, 110, 120, 130, 140 BPM)

# 2. WAV → MIDI
python scripts/suno_wav_to_midi.py \
  --input-dir data/suno_wav/guitar_strum_mid \
  --output-dir data/suno_midi/guitar_strum_mid \
  --method ensemble

# 3. Stage1 & Stage2
python scripts/clean_midi.py ...
python scripts/test_instrument_metrics.py ...

# 4. 結果確認
python scripts/analyze_stage2_results.py output/suno_results/guitar_strum_mid.json

# 5. 合格率 ≥70% → Pickle追加
python scripts/append_to_pickle_shard.py \
  --input-dir data/suno_clean/guitar_strum_mid \
  --pickle-dir data/shards/hybrid \
  --instrument guitar \
  --technique strum \
  --source suno
```

**期待結果:**
- 合格: 3-4曲（60-80%）
- Stage2平均: 45-50%（Real+5%達成）

---

### Iteration 2（Refinement: 20曲 × 5奏法 = 100曲）

```bash
# 1. プロンプト調整（Iteration 1のフィードバック反映）
# 2. Suno生成（手動 or 半自動）
# 3-5. 同上プロセス

# 累積Pickle統計確認
python scripts/pickle_statistics.py \
  --pickle-dir data/shards/hybrid \
  --instrument guitar

# Output:
# guitar_shard_00000.pkl: 5,000 (real: 963, suno: 4,037)
# guitar_shard_00001.pkl: 1,200 (real: 0, suno: 1,200)
# Total: 6,200 (real: 963, suno: 5,237)
```

---

### Iteration 3（Production: 100曲 × 5奏法 = 500曲）

**目標:** Critical Gaps完全補完

```bash
# Guitar strum: 1,554不足 → 500曲生成 → 350曲合格（70%）
# Strings legato: 1,117不足 → 400曲生成 → 280曲合格（70%）
# Guitar arpeggio: 1,007不足 → 350曲生成 → 245曲合格（70%）
# Bass pick: 900不足 → 300曲生成 → 210曲合格（70%）
# Strings spiccato: 600不足 → 200曲生成 → 140曲合格（70%）

Total生成: 1,750曲
Total合格: 1,225曲（70%）
```

**統合後のデータ構成:**
- Real Data: 3,559
- External Datasets: 2,160 (GuitarSet/URMP/MAESTRO/SMD)
- **Suno Synthetic: 1,225**
- **Total: 6,944** → Target: 7,888の88%達成

---

## 🎯 プロンプト最適化プロセス

### A. 初期プロンプト（テンプレートそのまま）

```
Instrumental only. Acoustic guitar ONLY, no drums, no bass, no keys.
Tempo 110 BPM, 4/4, key C.
Play a steady down-up 8th-note strum...
```

### B. Iteration 1結果

**問題:**
- ドラムが混入（5曲中2曲）
- ストラムパターン不明瞭（5曲中1曲）
- テンポずれ（±5 BPM）

**Stage2平均:** 38.5%（閾値45.0未達）

---

### C. プロンプト調整v2

```diff
- Instrumental only. Acoustic guitar ONLY, no drums, no bass, no keys.
+ [CRITICAL] Acoustic guitar SOLO ONLY. Absolutely NO drums, NO bass, NO percussion, NO keyboards.
+ Reject any arrangement with multiple instruments.
  Tempo 110 BPM, 4/4, key C.
- Play a steady down-up 8th-note strum...
+ Execute a precise down-up 8th-note strum pattern (down on beat, up on offbeat).
+ Metronome-locked timing with ±5ms tolerance.
+ Strum all 6 strings evenly, avoid muted strings or palm muting.
```

**Style調整:**
```diff
- acoustic pop, clean strum, down-up 8th, bright, dry room, tight timing
+ acoustic guitar solo, isolated instrument, clean strum only, down-up 8th pattern, 
+ bright attack, dry studio recording, no reverb, no effects, tight timing, 
+ metronome-accurate, studio microphone close-up
```

---

### D. Iteration 2結果

**改善:**
- ドラム混入なし（20曲中0曲）
- ストラムパターン明瞭（20曲中18曲）
- テンポ精度向上（±2 BPM）

**Stage2平均:** 47.2%（閾値45.0達成 ✅）
**合格率:** 75%（15/20）

→ **Production移行OK**

---

## 📋 実装チェックリスト

### Phase 1: 基盤整備（Week 1）
- [ ] `scripts/suno_wav_to_midi.py`作成（basic-pitch統合）
- [ ] `scripts/append_to_pickle_shard.py`作成（Resume対応）
- [ ] `scripts/generate_suno_batch.py`作成（プロンプト生成）
- [ ] `templates/`ディレクトリ作成（奏法別YAMLテンプレート）

### Phase 2: PoC実行（Week 1-2）
- [ ] Guitar strum 5曲生成（Suno AI手動）
- [ ] WAV → MIDI変換実行
- [ ] Stage1 & Stage2検証
- [ ] プロンプト調整v2作成

### Phase 3: Refinement（Week 2-3）
- [ ] Guitar strum 20曲生成（調整済みプロンプト）
- [ ] Strings legato 20曲生成
- [ ] 各奏法でPickle追加実行
- [ ] 合格率70%達成確認

### Phase 4: Production（Week 3-4）
- [ ] Critical Gaps完全補完（Top 5奏法）
- [ ] 累積Pickle統計確認
- [ ] Gap再評価（Manifest更新）
- [ ] Training Dataset準備

---

## 💡 重要な知見

### 1. Suno AIは「WAV生成」が正解
- ✅ **WAV出力推奨理由:**
  - Suno AIはWAV生成に最適化
  - MIDI直接生成は不可能
  - WAV → MIDI変換で柔軟性確保（複数モデル試行可能）

### 2. Ensemble Voting推奨
- basic-pitch単体より精度向上
- 3-5モデル投票で誤検出削減
- 計算コスト↑だが品質優先

### 3. プロンプトエンジニアリングが鍵
- Iteration 1で問題特定 → v2で解決
- StyleとPrompt両方調整が必要
- [CRITICAL]タグで楽器分離強制

### 4. 段階的Pickle追加が効率的
- 一気に1,000曲より、10曲 → 50曲 → 200曲が現実的
- プロンプト最適化しながら成長
- Resume対応でやり直し不要

---

**最終更新:** 2025年10月17日  
**次回更新:** PoC実行完了後（Guitar strum 5曲）
