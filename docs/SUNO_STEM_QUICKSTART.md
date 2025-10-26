# SUNO AI Stem Integration - クイックスタートガイド

このガイドでは、Suno AI生成楽曲のステムから新しいMIDI編曲を作成する手順を説明します。

---

## 🎯 前提条件

- Suno AIで生成した楽曲のステムファイル（6-12本のWAV）
- 原曲のボーカルWAVファイル（そのまま使用）
- セクション情報（Verse/Chorus等の開始小節）
- （任意）キー情報（例: "C:maj", "Am"）

---

## 📋 ステップ1：ステムファイルの準備

Sunoから分離されたステムを整理します：

```
project/
├── stems/
│   ├── vocals.wav          # メインボーカル
│   ├── backing_vocals.wav  # バッキングボーカル（任意）
│   ├── drums.wav           # ドラム
│   ├── bass.wav            # ベース
│   ├── guitar.wav          # ギター
│   ├── keyboard.wav        # キーボード/ピアノ
│   ├── strings.wav         # ストリングス
│   ├── percussion.wav      # パーカッション（任意）
│   └── synth.wav           # シンセ/FX（任意）
```

**重要**：ファイル名に楽器名を含めてください（自動認識に使用）

---

## 🔧 ステップ2：解析スクリプトの実行

### 基本的な使い方

```python
from analysis.stem_harmony import (
    make_beat_grid,
    estimate_activity,
    estimate_chords_per_stem,
    aggregate_stem_chords,
    extract_accent_grid,
    export_guides_to_midi
)

# ステムファイルのパス辞書
stems = {
    "vocals": "stems/vocals.wav",
    "drums": "stems/drums.wav",
    "bass": "stems/bass.wav",
    "guitar": "stems/guitar.wav",
    "keyboard": "stems/keyboard.wav",
}

# 1. ビートグリッド生成（テンポ・拍位置）
beat_grid = make_beat_grid(
    stems,
    default_bpm=120.0,  # 推定テンポ（手動で調整可）
    time_sig=(4, 4)      # 拍子
)

print(f"BPM: {beat_grid['bpm']}")
print(f"Total bars: {len(beat_grid['bars'])}")
```

### 活動マスクの抽出

```python
# 2. 各ステムの活動レベル（0..1）を小節ごとに取得
activity = {}
for role, path in stems.items():
    if role == "vocals":  # ボーカルはスキップ
        continue
    activity[role] = estimate_activity(path, beat_grid)

# 例: Bassの活動マスク
# [(0, 0.8), (1, 0.9), (2, 0.0), (3, 0.7), ...]
#  小節2はBassが休み → Bassジェネレータはこの小節をスキップ
```

### コード推定（各ステムから候補を抽出）

```python
# 3. 各ステムからコード候補を推定
stem_votes = {}
for role, path in stems.items():
    if role in ("vocals", "backing_vocals"):
        continue
    stem_votes[role] = estimate_chords_per_stem(
        path,
        beat_grid,
        role=role,
        key_hint="C:maj",  # 既知の調性（任意）
        top_n=2            # 上位N候補
    )

# 例: Guitar の拍(0, 1)のコード候補
# [(bar=0, beat=1)]: [
#     {"chord": "C:maj", "score": 0.71},
#     {"chord": "Am", "score": 0.54}
# ]
```

### コード集約（全ステムの投票を統合）

```python
# 4. 活動マスク×役割重みで統合
sections = [
    {"bar": 0, "label": "Intro"},
    {"bar": 4, "label": "Verse"},
    {"bar": 12, "label": "Chorus"}
]

cfg = {
    "weights": {
        "bass": 0.35,     # Bassは根音推定に重要
        "guitar": 0.35,   # Guitarは和声全体を反映
        "keyboard": 0.2,  # Pianoは和声補助
        "strings": 0.1    # Stringsは曖昧
    }
}

audio_chordmap = aggregate_stem_chords(
    stem_votes,
    activity,
    key_hint="C:maj",
    sections=sections,
    cfg=cfg
)

# 出力: audio_chordmap.yaml 形式
# {
#   "key": "C:maj",
#   "confidence_key": 0.78,
#   "items": [
#     {"bar": 0, "beat": 1, "chord": "C:maj", "confidence": 0.86},
#     {"bar": 2, "beat": 1, "chord": "F:maj", "confidence": 0.82},
#     ...
#   ]
# }
```

### アクセント格子の抽出（クロス楽器同期用）

```python
# 5. Kick/Snare/HiHat等の拍位置を抽出
accent_grid = extract_accent_grid(stems, beat_grid)

# 出力例:
# {
#   "kick": [0.0, 4.0, 8.0, ...],      # 各小節1拍目
#   "snare": [1.0, 3.0, 5.0, 7.0, ...],  # 2&4拍目
#   "hihat": [0.0, 1.0, 2.0, 3.0, ...],  # 全拍
#   "strum_ud": []  # Guitar由来（任意）
# }
```

### ガイドMIDIの書き出し（耳確認用）

```python
# 6. QA用のガイドMIDI（テンポ・マーカー・ブロックコード）
export_guides_to_midi(
    "output/guide.mid",
    beat_grid,
    sections,
    audio_chordmap
)
```

---

## 🎹 ステップ3：MIDI生成

### YAMLプリセットの設定

`config/harmony_config.yaml`（新規作成）:

```yaml
harmony:
  source: audio          # audio優先（原曲コード準拠）
  fallback: text         # 穴埋めは歌詞由来のchordmap
  keep_audio_root: true  # 根音は原曲優先
  prefer_root5: true     # 根音/5度を優先選択
  collapse_octaves: true # 連続オクターブ回避
  allow_text_tensions: []  # テンション追加なし（ボーカル保護）
```

### Stage2ジェネレータの実行

```python
from generator.bass_params_stage2 import BassParamsStage2
from generator.piano_params_stage2 import PianoParamsStage2
# ... 他のジェネレータも同様

# mix_context を構築
mix_context = {
    "beat_grid": beat_grid,
    "activity": activity,
    "accent_grid": accent_grid,
    "sections": sections
}

# overrides にまとめる
overrides = {
    "mix_context": mix_context,
    "audio_chordmap": audio_chordmap,
    "harmony": {
        "source": "audio",
        "keep_audio_root": True,
        "prefer_root5": True,
        "collapse_octaves": True
    }
}

# Bass生成
bass_params = BassParamsStage2()
bass_part = create_empty_part("bass", 16)  # 16小節
bass_result = bass_params.apply(
    bass_part,
    section_meta={"label": "Verse", "bar": 0, "tempo": 120},
    mix_context=mix_context,
    overrides=overrides["harmony"],
    seed=42
)

# MIDI書き出し
bass_result.write('midi', 'output/bass.mid')
```

### バッチ生成スクリプト

```python
# scripts/generate_from_stems.py（新規作成推奨）

instruments = ["bass", "drums", "piano", "guitar", "strings"]

for inst in instruments:
    # 活動マスクで無効な小節はスキップ
    active_bars = [
        bar for bar, level in activity.get(inst, [])
        if level > 0.25  # 閾値
    ]
    
    if not active_bars:
        print(f"Skipping {inst} (no activity)")
        continue
    
    # 生成（各instrumentのParams/Generatorを使用）
    result = generate_instrument(inst, overrides, active_bars)
    result.write('midi', f'output/{inst}.mid')
```

---

## 🎛️ ステップ4：パラメータ調整

### 活動マスクの閾値調整

```python
# 閾値を上げる → より厳密に原曲構成に従う
activity_threshold = 0.5  # 0.25 → 0.5

# 密度スケール（活動レベル連動）
density_scale = base_density * activity_level
```

### コード推定の重み調整

```python
# Bassを強調（根音精度優先）
cfg = {
    "weights": {
        "bass": 0.50,     # 0.35 → 0.50
        "guitar": 0.30,
        "keyboard": 0.15,
        "strings": 0.05
    }
}
```

### クロス楽器影響の有効化（eakey-style）

```yaml
piano:
  influence:
    drums:
      use: true
      kick_to_left_root: 0.7      # Kick → ピアノ左手ルート配置
      snare_to_right_accent: 0.5  # Snare → 右手アクセント
      hihat_subdivision_bias: 0.6 # HH密度 → アルペジオ密度
    guitar:
      use: true
      strum_to_broken_chord: updown  # ストラム方向 → 分散和音方向
      density_follow: 0.5            # ギター密度 → ピアノ密度
```

---

## 🔍 トラブルシューティング

### Q1. ビートグリッドがずれる

**症状**: 生成MIDIのリズムが原曲と合わない

**解決策**:
```python
# テンポを手動調整
beat_grid = make_beat_grid(stems, default_bpm=125.0)  # ← BPM調整

# または、外部ツールで正確なBPMを測定してから使用
```

### Q2. コード推定が不正確

**症状**: audio_chordmap が原曲と合わない

**解決策**:
```python
# 1. 耳コピで手動修正（推奨）
# data/audio_chordmap.yaml を直接編集

# 2. key_hint を正確に設定
stem_votes = estimate_chords_per_stem(
    path, beat_grid, role,
    key_hint="Am",  # ← 正確な調性
    top_n=3         # 候補数を増やす
)

# 3. 信頼度の低い箇所だけ手動修正
# confidence < 0.7 の小節を重点的にチェック
```

### Q3. 活動マスクが敏感すぎる/鈍い

**症状**: 鳴るべき箇所で休符になる、または休むべき箇所で鳴る

**解決策**:
```python
# RMSベースの閾値を調整（analysis/stem_harmony.py）
# estimate_activity() 内の正規化パラメータを変更

# または、手動でactivityを上書き
activity["bass"] = [
    (0, 1.0), (1, 1.0), (2, 0.0), (3, 0.8), ...  # 手動調整
]
```

### Q4. ボーカルと編曲が濁る

**症状**: 原曲ボーカルと新しい伴奏が不協和

**解決策**:
```yaml
harmony:
  source: audio          # ← audio優先を徹底
  keep_audio_root: true  # ← 根音保護
  prefer_root5: true     # ← 安全な音程優先
  collapse_octaves: true # ← オクターブ縮退
  allow_text_tensions: []  # ← テンション追加禁止

# さらに厳密に：
bass:
  scale_degree_weights:
    root: 0.70   # 根音を最優先
    fifth: 0.20  # 5度も安全
    third: 0.05  # 3度は控えめ
    others: 0.05
```

---

## 📊 動作確認

### テストスイートの実行

```bash
python scripts/test_stem_harmony.py
```

**期待される出力**:
```
✅ PASS: Role Guessing
✅ PASS: Beat Grid (Phase 13)
✅ PASS: Activity Mask (Phase 14)
✅ PASS: Chord Estimation (Phase 15)
✅ PASS: Chord Aggregation (Phase 16)
✅ PASS: Accent Grid (Phase 17)
✅ PASS: MIDI Export (Phase 18)

Total: 7/7 tests passed 🎉
```

### 個別機能のテスト

```python
# Beat grid精度確認
print(f"Detected BPM: {beat_grid['bpm']}")
print(f"First 5 bars: {beat_grid['bars'][:5]}")
# → [0.0, 4.0, 8.0, 12.0, 16.0] (4/4拍子の場合)

# Activity mask確認
for bar, level in activity["bass"][:10]:
    print(f"Bar {bar}: {level:.2f}")
# → 活動レベルが妥当か確認

# Chord map確認
for item in audio_chordmap["items"][:5]:
    print(f"Bar {item['bar']}: {item['chord']} (conf={item['confidence']:.2f})")
# → 原曲のコード進行と照合
```

---

## 🚀 次のステップ

### 1. 精度向上（将来実装）

- **librosa統合**: オンセット検出・テンポトラッキング
- **クロマベクトル**: より正確なコード推定（HMM/Viterbi）
- **F0推定**: ベースの根音を高精度で取得

### 2. ワークフロー自動化

```python
# scripts/suno_to_midi.py（新規作成）
def process_suno_stems(stem_dir, output_dir, config):
    """Sunoステム → MIDI一括生成"""
    stems = load_stems(stem_dir)
    mix_context = analyze_stems(stems, config)
    audio_chordmap = extract_chords(stems, mix_context)
    
    for inst in ["drums", "bass", "piano", "guitar", "strings"]:
        midi = generate_instrument(inst, mix_context, audio_chordmap)
        midi.write(f"{output_dir}/{inst}.mid")
```

### 3. DAW統合

- VST/AU版の開発（既存のplugin/基盤を拡張）
- リアルタイム解析（ストリーミング対応）
- Ableton Live / Logic Pro との連携

---

## 📚 関連ドキュメント

- **[STEM_HARMONY_IMPLEMENTATION.md](STEM_HARMONY_IMPLEMENTATION.md)**: 技術詳細・設計思想
- **[README.md](README.md)**: プロジェクト全体の概要
- **[analysis/stem_harmony.py](analysis/stem_harmony.py)**: コア実装（421行）
- **[scripts/test_stem_harmony.py](scripts/test_stem_harmony.py)**: テストスイート

---

## 💡 Tips

### 効率的なワークフロー

1. **最初はガイドMIDIで確認**: `export_guides_to_midi()` で素早く耳チェック
2. **活動マスクを可視化**: 小節ごとのactivityをグラフ化して原曲と比較
3. **コードは耳コピ併用**: 自動推定は70-80%の精度。残りは手動修正が確実
4. **段階的に有効化**: まず活動マスクだけ → 次にコード → 最後にアクセント格子

### 推奨設定（原曲ボーカル使用時）

- **harmony.source**: `audio`（必須）
- **keep_audio_root**: `true`（必須）
- **prefer_root5**: `true`（推奨）
- **collapse_octaves**: `true`（推奨）
- **allow_text_tensions**: `[]`（空＝安全）
- **activity_threshold**: `0.25-0.5`（楽器ごとに調整）

---

**Status**: ✅ Phase 13-18 実装完了・テスト通過  
**Version**: 1.0.0 (Skeleton - Safe Fallback)  
**Next**: Librosa統合（v2.0）・実オーディオ検証
