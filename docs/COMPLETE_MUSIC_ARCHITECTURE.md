# 🎼 Complete Music Generation Architecture
# =========================================
# 物語の感情 → コード理論 → 多重ステム統合システム

## 📖 全体ビジョン

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Complete Music Creation System                    │
│                                                                        │
│  [起点] 物語・歌詞 (小説家による創作)                                │
│    ↓                                                                   │
│  [感情分析] ChordMap + Rhythm + EmotionalYAML                        │
│    ↓                                                                   │
│  [コード理論] LAMDa統合 (先人の集大成)                               │
│    ↓                                                                   │
│  [楽器生成] 5つのComposer (Piano, Guitar, Bass, Drums, Sax)         │
│    ↓                                                                   │
│  [Suno統合] 12ステム分離 (Vocal含む完全楽曲)                         │
│    ↓                                                                   │
│  [統合システム] VocalSynchro + Mix + Mastering                        │
│    ↓                                                                   │
│  [出力] プロダクションレベル楽曲 (Avenger2級)                         │
└──────────────────────────────────────────────────────────────────────┘
```

## 🎯 3つの核心要素

### 1. 感情起点システム (ChordMap + EmotionalYAML)
**"作詞から生まれるchordmapと、それに命を吹き込むリズムとemotion"**

### 2. コード理論統合 (LAMDa)
**"先人たちの積み上げてきたコード理論という近代音楽の集大成"**

### 3. 多重ステム統合 (Suno AI 12 Stems)
**"vocalを含めた一つの楽曲、ステム分離wavで多様なdata"**

---

## 📊 データ構造の全体像

### 現在のデータ資産

#### 1. ChordMap (感情→コード変換)
```yaml
# mood.yaml, sections.yaml
verse:
  emotion: "melancholic"
  chords: ["Am", "F", "C", "G"]
  intensity: 0.6
  
chorus:
  emotion: "hopeful"
  chords: ["C", "G", "Am", "F"]
  intensity: 0.9
```

**役割**: 物語の感情をコード進行に変換
**現状**: `mood.yaml`, `sections.yaml`, `codex.yaml` に分散
**課題**: 感情とコードの対応が固定的

#### 2. EmotionalYAML (感情パラメータ)
```yaml
# emotion_humanizer.py で使用
emotions:
  joy:
    velocity_curve: [0.7, 0.9, 1.0, 0.8]
    timing_variance: 0.05
    swing: 0.03
    
  sadness:
    velocity_curve: [0.5, 0.6, 0.5, 0.4]
    timing_variance: 0.02
    swing: 0.0
```

**役割**: 感情を演奏パラメータに変換
**現状**: `emotion_humanizer.py` にハードコード
**課題**: YAMLによる柔軟な設定が未整備

#### 3. Rhythm Data (リズムパターン)
```yaml
# groove_profile.py で使用
jazz_swing:
  pattern: [1.0, 0.8, 1.2, 0.9]
  subdivision: 16
  accent_points: [0, 4, 8, 12]
```

**役割**: ジャンル別リズムテンプレート
**現状**: `groove_model.pkl` (訓練済み)
**課題**: YAML設定との統合

#### 4. LAMDa Dataset (コード理論)
```
CHORDS_DATA (15GB)     → 詳細MIDI 180,000曲
KILO_CHORDS (602MB)    → 整数シーケンス
SIGNATURES (290MB)     → 楽曲特徴量
```

**役割**: 先人のコード理論の集大成
**現状**: 統合アーキテクチャ構築済み
**課題**: ChordMapとの統合未実装

#### 5. 5つのComposer (楽器生成)
```python
# 現在実装済み
piano_composer    # modular_composer.py
guitar_composer   # (既存)
bass_composer     # (既存)
drums_composer    # (既存)
sax_composer      # train_sax_lora.py
```

**役割**: 楽器別MIDI生成
**現状**: 各々独立実装
**課題**: 統合制御システム未整備

---

## 🎼 新しい統合アーキテクチャ

### Phase 1: 感情→コード統合 (ChordMap + LAMDa)

```python
# scripts/emotion_to_chords.py (新規作成予定)

class EmotionChordMapper:
    """感情からコード進行を生成"""
    
    def __init__(self):
        self.chordmap = self.load_chordmap('mood.yaml')
        self.lamda = LAMDaUnifiedAnalyzer(Path('data/Los-Angeles-MIDI'))
    
    def generate_progression(self, emotion, story_context):
        """物語の感情からコード進行生成"""
        
        # 1. ChordMapから基本進行取得
        base_progression = self.chordmap.get_progression(emotion)
        # 例: ["Am", "F", "C", "G"]
        
        # 2. LAMDaで類似進行検索
        similar = self.lamda.search_similar_progressions(
            base_progression,
            filters={
                'emotion': emotion,
                'complexity': story_context.get('intensity', 0.5),
                'genre': story_context.get('genre', 'any')
            }
        )
        
        # 3. 最適な進行を選択 (または混合)
        enhanced_progression = self.select_best_progression(
            base_progression,
            similar,
            story_context
        )
        
        # 4. リズムとemotionパラメータを付与
        full_score = self.add_emotion_parameters(
            enhanced_progression,
            emotion,
            story_context
        )
        
        return full_score
```

### Phase 2: 統合Composer制御

```python
# scripts/unified_composer.py (新規作成予定)

class UnifiedComposer:
    """5つのComposerを統合制御"""
    
    def __init__(self):
        self.composers = {
            'piano': PianoComposer(),
            'guitar': GuitarComposer(),
            'bass': BassComposer(),
            'drums': DrumsComposer(),
            'sax': SaxComposer()
        }
        
        self.emotion_mapper = EmotionChordMapper()
    
    def compose_from_story(self, story_text, lyrics):
        """物語と歌詞から完全楽曲生成"""
        
        # 1. 感情分析
        emotions = self.analyze_emotions(story_text, lyrics)
        
        # 2. セクション分割 (verse, chorus, bridge)
        sections = self.segment_sections(lyrics)
        
        # 3. 各セクションでコード進行生成
        full_score = {}
        for section in sections:
            emotion = emotions[section.name]
            
            # ChordMap + LAMDa統合
            progression = self.emotion_mapper.generate_progression(
                emotion,
                section.context
            )
            
            full_score[section.name] = progression
        
        # 4. 5つのComposerで楽器生成
        tracks = {}
        for instrument, composer in self.composers.items():
            tracks[instrument] = composer.generate(
                full_score,
                emotion_params=emotions,
                sync_to='vocal'  # Vocal同期
            )
        
        return tracks
```

### Phase 3: Suno統合 (12ステム)

```python
# scripts/suno_stem_integration.py (新規作成予定)

class SunoStemIntegration:
    """Suno AI 12ステム統合"""
    
    STEM_TYPES = [
        'vocal_lead',
        'vocal_harmony', 
        'piano',
        'guitar',
        'bass',
        'drums',
        'synth',
        'strings',
        'brass',
        'percussion',
        'fx',
        'ambient'
    ]
    
    def integrate_stems(self, suno_stems, our_tracks):
        """Sunoステムと我々の生成トラックを統合"""
        
        integrated = {}
        
        for stem_name in self.STEM_TYPES:
            if stem_name in suno_stems:
                # Sunoのステム使用
                stem = suno_stems[stem_name]
                
                if stem_name in our_tracks:
                    # 両方ある場合は品質向上パイプライン適用
                    enhanced = self.enhance_stem(
                        suno_midi=stem['midi'],
                        our_midi=our_tracks[stem_name],
                        wav=stem['wav']
                    )
                    integrated[stem_name] = enhanced
                else:
                    # Sunoのみの場合は品質向上のみ
                    integrated[stem_name] = self.enhance_suno_stem(stem)
            
            elif stem_name in our_tracks:
                # 我々のトラックのみ
                integrated[stem_name] = our_tracks[stem_name]
        
        return integrated
    
    def enhance_stem(self, suno_midi, our_midi, wav):
        """ステム品質向上"""
        
        # 1. MIDIを比較分析
        comparison = self.compare_midis(suno_midi, our_midi)
        
        # 2. ベストプラクティス抽出
        if comparison['our_quality'] > comparison['suno_quality']:
            # 我々のMIDIをベースに
            base = our_midi
            reference = suno_midi
        else:
            # SunoをベースにLAMDa強化
            base = suno_midi
            base = self.enhance_with_lamda(base)
            reference = our_midi
        
        # 3. WAVと同期
        synced = self.sync_with_wav(base, wav)
        
        # 4. Humanization
        final = self.humanize(synced)
        
        return {
            'midi': final,
            'wav': wav,
            'quality_score': self.evaluate_quality(final)
        }
```

### Phase 4: VocalSynchro + Mix + Mastering

```python
# scripts/full_production_pipeline.py (新規作成予定)

class FullProductionPipeline:
    """プロダクションレベル統合パイプライン"""
    
    def __init__(self):
        self.composer = UnifiedComposer()
        self.suno_integration = SunoStemIntegration()
    
    def produce_song(self, story, lyrics, suno_export_dir):
        """完全楽曲プロダクション"""
        
        print("🎼 Full Production Pipeline")
        print("=" * 70)
        
        # Stage 1: 感情→コード→楽器生成
        print("\n[Stage 1] Emotion → Chords → Instruments")
        our_tracks = self.composer.compose_from_story(story, lyrics)
        
        # Stage 2: Sunoステム読み込み
        print("\n[Stage 2] Loading Suno AI stems (12 tracks)")
        suno_stems = self.load_suno_stems(suno_export_dir)
        
        # Stage 3: ステム統合
        print("\n[Stage 3] Integrating stems")
        integrated = self.suno_integration.integrate_stems(
            suno_stems,
            our_tracks
        )
        
        # Stage 4: Vocal同期
        print("\n[Stage 4] Vocal synchronization")
        synced = self.vocal_synchro(integrated)
        
        # Stage 5: ミキシング
        print("\n[Stage 5] Mixing")
        mixed = self.mix_tracks(synced)
        
        # Stage 6: マスタリング
        print("\n[Stage 6] Mastering")
        mastered = self.master(mixed)
        
        # 出力
        print("\n[Output] Rendering final production")
        return self.render_final(mastered)
    
    def vocal_synchro(self, tracks):
        """Vocal同期 (既存システム活用)"""
        # guide_vocal.mid を使用
        # ujam.sparkle_convert の --guide-vocal 機能
        return tracks
    
    def mix_tracks(self, tracks):
        """ミキシング"""
        # mixing_assistant/ モジュール活用
        return tracks
    
    def master(self, mixed):
        """マスタリング"""
        # 最終調整
        return mixed
```

---

## 🗂️ データ整理計画

### 優先度1: ChordMap統合

**現状の分散ファイル**:
- `mood.yaml` - 感情→コード基本マップ
- `sections.yaml` - セクション定義
- `codex.yaml` - コード詳細仕様

**統合先**: `config/emotion_chord_map.yaml` (新規作成)

```yaml
# config/emotion_chord_map.yaml

emotions:
  melancholic:
    primary_chords: ["Am", "F", "C", "G"]
    alternative_progressions:
      - ["Am", "Dm", "G", "C"]
      - ["Am", "F", "Dm", "E7"]
    intensity_modifiers:
      low: 
        velocity: [0.4, 0.5]
        tempo_factor: 0.85
      medium:
        velocity: [0.5, 0.7]
        tempo_factor: 1.0
      high:
        velocity: [0.7, 0.9]
        tempo_factor: 1.15
    rhythm_profile: "ballad"
    
  hopeful:
    primary_chords: ["C", "G", "Am", "F"]
    alternative_progressions:
      - ["C", "Am", "F", "G"]
      - ["C", "Dm", "G", "Am"]
    intensity_modifiers:
      # ...
    rhythm_profile: "upbeat"

# LAMDa統合設定
lamda_integration:
  enabled: true
  search_method: "kilo_chords"
  similarity_threshold: 0.7
  max_alternatives: 10
  
# リズムプロファイル
rhythm_profiles:
  ballad:
    tempo: 70-90
    groove: "straight"
    swing: 0.0
  
  upbeat:
    tempo: 120-140
    groove: "dance"
    swing: 0.05
```

### 優先度2: EmotionalYAML整備

**現状**: `emotion_humanizer.py` にハードコード

**統合先**: `config/emotion_parameters.yaml` (新規作成)

```yaml
# config/emotion_parameters.yaml

joy:
  velocity:
    curve: [0.7, 0.9, 1.0, 0.8]
    randomness: 0.1
  timing:
    variance: 0.05
    swing: 0.03
  articulation:
    staccato: 0.3
    legato: 0.7
  dynamics:
    crescendo: true
    accent_beats: [1, 3]

sadness:
  velocity:
    curve: [0.5, 0.6, 0.5, 0.4]
    randomness: 0.05
  timing:
    variance: 0.02
    swing: 0.0
  articulation:
    staccato: 0.1
    legato: 0.9
  dynamics:
    diminuendo: true
    rubato: 0.1

# ... 他の感情
```

### 優先度3: Rhythm Data統合

**現状**: `groove_model.pkl` (訓練済み)

**追加**: `config/rhythm_templates.yaml` (新規作成)

```yaml
# config/rhythm_templates.yaml

genres:
  jazz:
    swing:
      pattern: [1.0, 0.8, 1.2, 0.9]
      subdivision: 16
      accent_points: [0, 4, 8, 12]
    straight:
      pattern: [1.0, 1.0, 1.0, 1.0]
      subdivision: 16
      accent_points: [0, 8]
  
  rock:
    four_on_floor:
      pattern: [1.0, 0.9, 1.0, 0.9]
      subdivision: 16
      accent_points: [0, 4, 8, 12]

# groove_model.pkl との統合
model_integration:
  use_trained_model: true
  fallback_to_templates: true
```

---

## 🎯 実装ロードマップ

### Phase 1: データ統合 (今週-来週)
- [ ] `config/emotion_chord_map.yaml` 作成
- [ ] `config/emotion_parameters.yaml` 作成
- [ ] `config/rhythm_templates.yaml` 作成
- [ ] 既存YAML (`mood.yaml`, etc) マイグレーション

### Phase 2: 感情→コード統合 (来週-2週間後)
- [ ] `scripts/emotion_to_chords.py` 実装
- [ ] `EmotionChordMapper` class実装
- [ ] LAMDa統合テスト
- [ ] ChordMap ↔ LAMDa連携

### Phase 3: 統合Composer (2週間後-1ヶ月)
- [ ] `scripts/unified_composer.py` 実装
- [ ] 5つのComposer統合制御
- [ ] Vocal同期機能実装
- [ ] エンドツーエンドテスト

### Phase 4: Suno統合 (1ヶ月後-2ヶ月)
- [ ] `scripts/suno_stem_integration.py` 実装
- [ ] 12ステム読み込み
- [ ] MIDI/WAV統合
- [ ] 品質向上パイプライン

### Phase 5: フルプロダクション (2ヶ月後-)
- [ ] `scripts/full_production_pipeline.py` 実装
- [ ] ミキシング自動化
- [ ] マスタリング統合
- [ ] プロダクションレベル出力

---

## 💡 Avenger2級システムへの道

### Avenger2の特徴
- 複数オシレーター
- 高度なモジュレーション
- エフェクトチェーン
- プリセット管理

### 我々のシステムが目指すもの
```
┌────────────────────────────────────────────────────┐
│  Beyond Synthesizer - Complete Production System   │
├────────────────────────────────────────────────────┤
│                                                      │
│  [Input Layer]                                      │
│    • Story/Lyrics (感情起点)                        │
│    • ChordMap + EmotionalYAML                       │
│                                                      │
│  [Intelligence Layer]                               │
│    • LAMDa (180,000曲のコード理論)                 │
│    • Emotion→Parameter変換                          │
│    • Rhythm/Groove自動選択                          │
│                                                      │
│  [Generation Layer]                                 │
│    • 5 Composers (Piano/Guitar/Bass/Drums/Sax)     │
│    • Humanization (感情表現)                        │
│    • Vocal Synchro                                  │
│                                                      │
│  [Integration Layer]                                │
│    • Suno 12 Stems統合                              │
│    • MIDI/WAV自動統合                               │
│    • 品質向上パイプライン                           │
│                                                      │
│  [Production Layer]                                 │
│    • Mixing Assistant                               │
│    • Mastering                                      │
│    • Export (Pro quality)                           │
└────────────────────────────────────────────────────┘

= Avenger2 (シンセ) < Our System (完全作曲システム)
```

---

## 📚 関連ドキュメント

- [LAMDa統合](LAMDA_README.md) - コード理論の集大成
- [自己増殖システム](FUTURE_SELF_IMPROVING_SYSTEM.md) - Suno統合計画
- [Humanizer](../humanizer.md) - 感情表現
- [Groove](../groove.md) - リズム強化
- [Vocal Generator](../vocal_generator.md) - Vocal統合

---

## 🚀 次のアクション

1. **今すぐ**: LAMDaローカルテスト完了
2. **今週**: ChordMap/EmotionalYAML統合YAML作成
3. **来週**: `emotion_to_chords.py` プロトタイプ
4. **来月**: Suno 12ステムサンプル収集

---

**"物語の感情から、先人のコード理論を経て、12ステムの完全楽曲へ"**

**Avenger2を超える、完全音楽創造システムへ 🎼**
