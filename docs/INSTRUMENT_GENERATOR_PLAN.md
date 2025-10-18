# Instrument Generator Development Plan

楽器別Generator開発の全体計画

## 目標

各楽器の特性を理解し、スタイル・奏法・感情表現に応じた高品質なMIDIを生成するGeneratorを開発する。

## 対象楽器 & データセット

| 楽器 | データセット | ファイル数 | 平均スコア | 合格率 | 主要奏法 |
|------|-------------|-----------|-----------|--------|---------|
| **Piano** | POP909 (Melody+Chords) | 554 | 64.0% | 100% | pop_comping, ballad, jazz_voicing, arpeggio |
| **Bass** | SLAKH | 584 | 76.9% | 100% | walking, pick, slap, fingerstyle |
| **Guitar** | SLAKH | 963 | 42.9% | 67.7% | strum, arpeggio, fingerpicking, power_chord |
| **Strings** | SLAKH | 696 | 51.1% | 69.7% | legato, staccato, spiccato, sustained, tremolo |
| **Drums** | SLAKH + LAMDA | 873 | 55.7% | 9-100% | groove, fill, pattern, swing |

**Total:** 3,670 high-quality MIDI files

## アーキテクチャ

### Generator階層構造

```
InstrumentGeneratorBase (抽象クラス)
├── PianoGenerator
│   ├── MelodyGenerator (旋律特化)
│   └── CompinoGenerator (伴奏特化)
├── BassGenerator
├── GuitarGenerator
├── StringsGenerator
└── DrumsGenerator
```

### 共通インターフェース

```python
class InstrumentGeneratorBase:
    def generate(
        self,
        section: Section,
        technique: str,
        emotion: EmotionProfile,
        context: GenerationContext,
    ) -> List[NoteEvent]:
        """
        楽器別MIDI生成
        
        Args:
            section: セクション情報 (tempo, key, chord_progression)
            technique: 奏法 (strum/legato/walking等)
            emotion: 感情プロファイル (joy/sorrow/tension等)
            context: 生成コンテキスト (前セクションの情報、他楽器との協調)
        
        Returns:
            NoteEvent のリスト
        """
        pass
    
    def apply_technique(self, notes: List[NoteEvent], technique: str) -> List[NoteEvent]:
        """奏法を適用（velocity/timing/articulation調整）"""
        pass
    
    def apply_emotion(self, notes: List[NoteEvent], emotion: EmotionProfile) -> List[NoteEvent]:
        """感情表現を適用（dynamics/tempo_variation等）"""
        pass
    
    def validate(self, notes: List[NoteEvent]) -> ValidationResult:
        """生成結果の品質検証（Stage2メトリクス使用）"""
        pass
```

## Phase 1: Piano Generator（優先度★★★）

### 理由
- **最高品質データ**: 100%合格率、平均64%
- **作曲の中核**: Melody + Harmonyの両方を担当
- **段階的開発**: Melody → Chords → 統合の順で実装可能

### 実装内容

#### 1.1 MelodyGenerator
```python
class MelodyGenerator(InstrumentGeneratorBase):
    """Piano旋律生成"""
    
    def __init__(self, model_path: str):
        # POP909 v1 (277 files) で学習済みモデル読み込み
        self.model = load_model(model_path)
        self.metrics = PianoMetrics()  # Stage2メトリクス
    
    def generate(self, section, technique, emotion, context):
        # 1. コード進行からmelodyの候補音を抽出
        candidate_pitches = self._extract_chord_tones(section.chord_progression)
        
        # 2. Emotion → Melody contour（上昇/下降/波形）
        contour = self._emotion_to_contour(emotion)
        
        # 3. Rhythm pattern生成（8分/16分/シンコペーション）
        rhythm = self._generate_rhythm(section.tempo, technique)
        
        # 4. Pitch + Rhythm → Melody
        melody = self._combine_pitch_rhythm(candidate_pitches, rhythm, contour)
        
        # 5. Articulation適用（legato/staccato）
        melody = self._apply_articulation(melody, technique)
        
        # 6. Dynamics適用（pp → ff）
        melody = self._apply_dynamics(melody, emotion)
        
        return melody
    
    def _extract_chord_tones(self, chords):
        """コード構成音 + テンション音を抽出"""
        tones = []
        for chord in chords:
            # Root, 3rd, 5th, 7th, 9th等
            tones.extend(chord.get_tones(extensions=True))
        return tones
    
    def _emotion_to_contour(self, emotion):
        """Emotion → Melody輪郭"""
        if emotion.primary == "joy":
            return "ascending"  # 上昇傾向
        elif emotion.primary == "sorrow":
            return "descending"  # 下降傾向
        elif emotion.primary == "tension":
            return "wave_large"  # 大きな波形
        else:
            return "wave_small"  # 小さな波形
```

#### 1.2 CompingGenerator
```python
class CompingGenerator(InstrumentGeneratorBase):
    """Piano伴奏生成"""
    
    def generate(self, section, technique, emotion, context):
        # 1. Chord voicing決定（open/close/drop2等）
        voicing = self._select_voicing(technique, section.chord_progression)
        
        # 2. Rhythm pattern（pop_comping/ballad/jazz_voicing）
        if technique == "pop_comping":
            rhythm = self._pop_comping_pattern(section.tempo)
        elif technique == "ballad":
            rhythm = self._ballad_pattern(section.tempo)
        elif technique == "jazz_voicing":
            rhythm = self._jazz_voicing_pattern(section.tempo)
        
        # 3. Voicing + Rhythm → Comping
        comping = self._apply_voicing_to_rhythm(voicing, rhythm)
        
        # 4. Pedaling（CC64）
        comping = self._add_pedaling(comping, technique)
        
        return comping
```

#### 1.3 統合
```python
class PianoGenerator(InstrumentGeneratorBase):
    def __init__(self):
        self.melody_gen = MelodyGenerator("models/piano_melody.pt")
        self.comping_gen = CompingGenerator("models/piano_comping.pt")
    
    def generate(self, section, technique, emotion, context):
        # Melody + Comping統合生成
        melody = self.melody_gen.generate(section, "melody", emotion, context)
        comping = self.comping_gen.generate(section, technique, emotion, context)
        
        # 音域被り回避（Melody: C4-C6, Comping: C2-C4）
        melody = self._shift_to_high_register(melody)
        comping = self._shift_to_low_register(comping)
        
        # 統合
        return melody + comping
```

### 学習データ
- **Melody**: POP909 v1 (277 files)
- **Comping**: POP909 v2 (277 files)
- **Augmentation**: Transpose (±3 semitones), Tempo (±10%)

### 評価指標（Stage2メトリクス）
- melody_expression: 69.9% → 目標75%
- rhythm_diversity: 86.6% → 維持
- dynamics_range: 76.9% → 目標80%
- chord_progression_quality: 新規追加
- pedaling_quality: 17.7% → 目標50% (推定アルゴリズム改善)

---

## Phase 2: Bass Generator（優先度★★★）

### 理由
- **最高品質データ**: 100%合格率、平均76.9%
- **リズムの要**: Drums と協調してGroove形成
- **多様な奏法**: walking/pick/slap/fingerstyle

### 実装内容

#### 2.1 BassGenerator
```python
class BassGenerator(InstrumentGeneratorBase):
    """Bass生成"""
    
    def generate(self, section, technique, emotion, context):
        # 1. Root note抽出（コード進行から）
        root_notes = self._extract_roots(section.chord_progression)
        
        # 2. Bass line生成（technique別）
        if technique == "walking":
            bass_line = self._walking_bass(root_notes, section.tempo)
        elif technique == "pick":
            bass_line = self._pick_bass(root_notes, section.tempo)
        elif technique == "slap":
            bass_line = self._slap_bass(root_notes, section.tempo)
        elif technique == "fingerstyle":
            bass_line = self._fingerstyle_bass(root_notes, section.tempo)
        
        # 3. Drums同期（kick drumとタイミング合わせ）
        if context.drums_pattern:
            bass_line = self._sync_with_drums(bass_line, context.drums_pattern)
        
        # 4. Groove調整（velocity/timing微調整）
        bass_line = self._apply_groove(bass_line, emotion)
        
        return bass_line
    
    def _walking_bass(self, root_notes, tempo):
        """Walking bass (Jazz/Swing)"""
        # 4分音符でRoot → 3rd → 5th → 7th
        bass_line = []
        for root in root_notes:
            bass_line.append(NoteEvent(pitch=root, duration=0.5, velocity=70))
            bass_line.append(NoteEvent(pitch=root+4, duration=0.5, velocity=65))  # 3rd
            bass_line.append(NoteEvent(pitch=root+7, duration=0.5, velocity=68))  # 5th
            bass_line.append(NoteEvent(pitch=root+11, duration=0.5, velocity=66)) # 7th
        return bass_line
    
    def _pick_bass(self, root_notes, tempo):
        """Pick bass (Rock/Pop)"""
        # 8分音符でRoot反復 + 5th accent
        bass_line = []
        for root in root_notes:
            for i in range(4):
                if i == 2:  # 3拍目に5th
                    bass_line.append(NoteEvent(pitch=root+7, duration=0.25, velocity=75))
                else:
                    bass_line.append(NoteEvent(pitch=root, duration=0.25, velocity=70))
        return bass_line
    
    def _sync_with_drums(self, bass_line, drums_pattern):
        """Kick drumとBass rootをタイミング同期"""
        kick_times = [note.time for note in drums_pattern if note.pitch == 36]  # Kick = MIDI 36
        
        for bass_note in bass_line:
            # 最も近いkickに合わせる
            nearest_kick = min(kick_times, key=lambda t: abs(t - bass_note.time))
            if abs(bass_note.time - nearest_kick) < 0.05:  # 50ms以内
                bass_note.time = nearest_kick  # 同期
        
        return bass_line
```

### 学習データ
- **SLAKH Bass**: 584 files (100% pass)
- **Technique distribution**: walking(25%), pick(40%), slap(15%), fingerstyle(20%)

### 評価指標
- root_accuracy: 84.3% → 目標90%
- groove_quality: 64.3% → 目標75%
- pitch_range_fit: 87.1% → 維持
- drums_sync_quality: 新規追加（目標80%）

---

## Phase 3: Guitar Generator（優先度★★）

### 理由
- **大量データ**: 963 files (67.7% pass)
- **多彩な表現**: strum/arpeggio/fingerpicking
- **Sunoデータ補完必須**: guitar_strum (1,554 needed)

### 実装内容

#### 3.1 GuitarGenerator
```python
class GuitarGenerator(InstrumentGeneratorBase):
    """Guitar生成"""
    
    def generate(self, section, technique, emotion, context):
        # 1. Chord voicing（Guitar特有のオープンボイシング）
        voicing = self._guitar_voicing(section.chord_progression)
        
        # 2. Technique適用
        if technique == "strum":
            pattern = self._strum_pattern(section.tempo, emotion)
        elif technique == "arpeggio":
            pattern = self._arpeggio_pattern(section.tempo)
        elif technique == "fingerpicking":
            pattern = self._fingerpicking_pattern(section.tempo)
        elif technique == "power_chord":
            pattern = self._power_chord_pattern(section.tempo)
        
        # 3. Stringsパート協調（重複音域回避）
        if context.strings_notes:
            pattern = self._avoid_overlap_with_strings(pattern, context.strings_notes)
        
        return pattern
    
    def _strum_pattern(self, tempo, emotion):
        """Strum pattern生成"""
        if tempo < 90:
            # Slow: Quarter note down-strum
            return self._down_strum_quarter()
        elif tempo < 130:
            # Mid: Down-up 8th note
            return self._down_up_8th()
        else:
            # Fast: 16th note strumming
            return self._16th_strum()
    
    def _arpeggio_pattern(self, tempo):
        """Arpeggio pattern (Thumb-3rd-5th-Octave)"""
        pattern = []
        for chord in self.current_chords:
            # Thumb (bass note)
            pattern.append(NoteEvent(pitch=chord.root, velocity=75, duration=0.125))
            # 3rd
            pattern.append(NoteEvent(pitch=chord.root+4, velocity=70, duration=0.125))
            # 5th
            pattern.append(NoteEvent(pitch=chord.root+7, velocity=72, duration=0.125))
            # Octave
            pattern.append(NoteEvent(pitch=chord.root+12, velocity=68, duration=0.125))
        return pattern
```

### 学習データ
- **SLAKH Guitar**: 963 files (67.7% pass)
- **Suno補完（計画）**: guitar_strum (1,554), guitar_arpeggio (1,007)

### 評価指標
- arpeggio_quality: 45.2% → 目標60%
- chord_consonance: 38.7% → 目標50%
- strum_pattern_quality: 新規追加（目標65%）

---

## Phase 4: Strings Generator（優先度★★）

### 理由
- **豊かな表現**: legato/staccato/spiccato/tremolo
- **Sunoデータ補完必須**: strings_legato (1,117 needed)
- **Guitar協調**: 音域・和音の協調が重要

### 実装内容

#### 4.1 StringsGenerator
```python
class StringsGenerator(InstrumentGeneratorBase):
    """Strings生成"""
    
    def generate(self, section, technique, emotion, context):
        # 1. Voicing（Strings ensemble: Violin I/II, Viola, Cello）
        voicing = self._strings_voicing(section.chord_progression)
        
        # 2. Bowing technique
        if technique == "legato":
            notes = self._legato_bowing(voicing, section.tempo)
        elif technique == "staccato":
            notes = self._staccato_bowing(voicing, section.tempo)
        elif technique == "spiccato":
            notes = self._spiccato_bowing(voicing, section.tempo)
        elif technique == "tremolo":
            notes = self._tremolo_bowing(voicing, section.tempo)
        
        # 3. Vibrato追加（CC1 Modulation）
        notes = self._add_vibrato(notes, emotion)
        
        # 4. Dynamic swells（crescendo/diminuendo）
        notes = self._dynamic_swells(notes, emotion)
        
        return notes
    
    def _legato_bowing(self, voicing, tempo):
        """Legato bowing (Smooth, overlap >90%)"""
        notes = []
        for chord in voicing:
            for pitch in chord.pitches:
                duration = 2.0  # Long notes
                velocity = 65
                notes.append(NoteEvent(pitch=pitch, duration=duration, velocity=velocity))
        
        # Overlap調整（次の音が始まる前に前の音を0.1秒延長）
        for i in range(len(notes)-1):
            notes[i].duration += 0.1
        
        return notes
    
    def _spiccato_bowing(self, voicing, tempo):
        """Spiccato (Bouncing bow, short duration, high velocity variation)"""
        notes = []
        for chord in voicing:
            for pitch in chord.pitches:
                duration = 0.2  # Short
                velocity = random.randint(70, 95)  # High variation
                notes.append(NoteEvent(pitch=pitch, duration=duration, velocity=velocity))
        return notes
```

### 学習データ
- **SLAKH Strings**: 696 files (69.7% pass)
- **Suno補完（計画）**: strings_legato (1,117), strings_spiccato (600)

### 評価指標
- bowing_expression: 48.6% → 目標65%
- legato_quality: 42.0% → 目標60% (最優先改善)
- harmony_quality: 57.2% → 目標70%

---

## Phase 5: Drums Generator（優先度★）

### 理由
- **LAMDA特化**: 既存のLAMDA Drumsシステムが強力
- **大量データ**: SLAKH 412 + LAMDA 461 = 873 loops
- **Stage2完了済み**: 高品質ループを選抜済み

### 実装内容

#### 5.1 DrumsGenerator（LAMDA拡張）
```python
class DrumsGenerator(InstrumentGeneratorBase):
    """Drums生成（LAMDA Drums拡張）"""
    
    def __init__(self):
        # 既存LAMDAシステム統合
        from lamda_integration import LAMDADrumGenerator
        self.lamda_gen = LAMDADrumGenerator()
    
    def generate(self, section, technique, emotion, context):
        # 1. LAMDA Drumsでベースパターン生成
        pattern = self.lamda_gen.generate_pattern(
            tempo=section.tempo,
            style=technique,  # groove/fill/swing
            complexity=emotion.intensity,
        )
        
        # 2. Bass協調（kick × bass root同期）
        if context.bass_notes:
            pattern = self._sync_kick_with_bass(pattern, context.bass_notes)
        
        # 3. Emotion → Dynamics
        pattern = self._apply_emotion_to_drums(pattern, emotion)
        
        return pattern
    
    def _sync_kick_with_bass(self, drums_pattern, bass_notes):
        """Kick drumをBass rootに同期"""
        kick_notes = [n for n in drums_pattern if n.pitch == 36]  # Kick
        bass_roots = [n for n in bass_notes if n.is_root]
        
        # Bass rootのタイミングにkickを配置
        synced_kicks = []
        for bass in bass_roots:
            kick = NoteEvent(pitch=36, time=bass.time, velocity=90, duration=0.1)
            synced_kicks.append(kick)
        
        # 既存kickを削除、新kickを追加
        drums_pattern = [n for n in drums_pattern if n.pitch != 36]
        drums_pattern.extend(synced_kicks)
        
        return drums_pattern
```

### 学習データ
- **SLAKH Drums**: 412 loops
- **LAMDA Drums**: 461 loops (51,248 → Stage2選抜)
- **Total**: 873 high-quality loops

### 評価指標
- groove_quality: 64.3% → 目標75%
- pattern_diversity: 新規追加（目標70%）
- bass_sync_quality: 新規追加（目標85%）

---

## 実装スケジュール

### Week 1-2: Piano Generator
- [x] Stage2完了（277+277 = 554 files）
- [ ] MelodyGenerator実装
- [ ] CompingGenerator実装
- [ ] 統合 + 小規模テスト（10曲生成）

### Week 3-4: Bass Generator
- [x] Stage2完了（584 files）
- [ ] BassGenerator実装
- [ ] Drums同期機能実装
- [ ] 統合テスト（Piano+Bass）

### Week 5-6: Guitar Generator
- [x] Stage2完了（963 files）
- [ ] GuitarGenerator実装
- [ ] Suno AI補完（guitar_strum 1,554）
- [ ] Strings協調機能実装

### Week 7-8: Strings Generator
- [x] Stage2完了（696 files）
- [ ] StringsGenerator実装
- [ ] Suno AI補完（strings_legato 1,117）
- [ ] Guitar協調テスト

### Week 9: Drums Generator + 全楽器統合
- [x] Stage2完了（873 loops）
- [ ] DrumsGenerator実装（LAMDA拡張）
- [ ] Bass同期実装
- [ ] **全楽器統合テスト**（完全な楽曲生成）

---

## 技術的課題

### 1. Multi-instrument協調
**課題**: 各楽器が独立生成すると音域・リズムが衝突

**解決策**:
- **Context共有**: 前Generatorの出力を次Generatorに渡す
- **音域分離**: Piano(C2-C6), Guitar(E2-E5), Strings(G3-E6), Bass(E1-E3)
- **Timing同期**: Drums/Bass kickを基準に全楽器同期

### 2. Emotion表現の一貫性
**課題**: 各Generatorが異なるEmotion解釈をすると統一感なし

**解決策**:
- **EmotionProfile統一**: 全Generatorに同じEmotionProfileを渡す
- **Emotion → Parameter mapping**:
  - Joy: Velocity +10, Tempo +5%, Major scale
  - Sorrow: Velocity -10, Tempo -5%, Minor scale
  - Tension: Velocity variation +20, Dissonant intervals

### 3. データ不足への対応
**課題**: Guitar strum (1,554), Strings legato (1,117) 不足

**解決策**:
- **Suno AI補完**: Phase 1で実装済み（WAV→MIDI→Stage2→Pickle）
- **Data Augmentation**: Transpose, Tempo, Velocity variation
- **Transfer Learning**: 類似奏法から転移学習（strum → arpeggio）

---

## 評価方法

### 自動評価（Stage2メトリクス）
- 各Generator出力をStage2スコアリング
- 閾値: Real data平均 + 5%
- 合格率目標: ≥70%

### 人間評価（主観評価）
- 10曲生成 → 5段階評価（1-5点）
- 評価軸: Musicality, Emotion表現, Multi-instrument協調
- 目標: 平均4.0点以上

### A/Bテスト
- Generator生成 vs Real MIDI
- Blind test（どちらが生成か判別）
- 目標: 50%正解率（判別不可能）

---

## 次のアクション

**Immediate (今週):**
1. ✅ Piano Generator実装開始（MelodyGenerator）
2. ⏸️ POP909 v1データ読み込み・前処理
3. ⏸️ Emotion → Melody contour mapping実装

**Short-term (2週間):**
1. Piano Generator完成 + 小規模テスト
2. Bass Generator実装開始
3. Suno AI補完 PoC（Guitar strum 5曲）

**Mid-term (1ヶ月):**
1. Guitar/Strings Generator実装
2. 全楽器統合テスト
3. 実際の作曲デモ（3-5曲）

---

## 関連ドキュメント

- **Stage2結果**: `STAGE2_FULL_PRODUCTION_REPORT.md`
- **Suno AI統合**: `SUNO_GENERATION_GUIDE.md`
- **Manifest Runner**: `PHASE1_IMPLEMENTATION_REPORT.md`
- **Multi-Dataset Runner**: `MULTI_DATASET_RUNNER_GUIDE.md`

---

## まとめ

楽器別Generator開発により：

- ✅ **高品質データ活用**: Stage2で選抜された3,670 files
- ✅ **楽器特性反映**: 奏法・音域・Articulation適切に実装
- ✅ **Multi-instrument協調**: Context共有で音域・リズム衝突回避
- ✅ **Emotion表現統一**: EmotionProfileで一貫した表現
- ✅ **データ補完戦略**: Suno AI + Data Augmentation

**次フェーズ**: Piano Generator実装開始 → 実際の作曲デモへ！
