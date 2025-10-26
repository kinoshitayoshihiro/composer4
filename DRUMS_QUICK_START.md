# Drums Enhancement - Quick Start Guide

## 🚀 クイックスタート（3ステップ）

### ステップ1: emotion_profile.yaml編集

```yaml
# configs/emotion_profile.yaml
emotions:
  energetic:
    # ... 既存設定 ...
    
    # ★ここから追加
    drums_style: "tight_rock"  # オプション: スタイルプリセット
    
    drums_params:
      # HHをOpen化（裏拍40%）
      open_ratio:
        strong_beat: 0.1
        weak_beat: 0.2
        off_beat: 0.4
      
      # Crashをセクション冒頭に追加
      crash:
        downbeat_prob: 0.2
        with_kick: true
      
      # フィルを3,7,15小節目に挿入
      fills:
        insert_bars: [3, 7, 15]
        intensity: "medium"
```

### ステップ2: （既存コードそのまま）

```python
# scripts/suno_stem_arranger.py
# ★変更不要！自動的に適用されます

arranger = SunoStemArranger()
score = arranger.arrange_with_generators(
    chords=["C", "G", "Am", "F"],
    tempo=120,
    emotion="energetic",  # ← emotion_profile.yamlのenergeticが適用される
    bars=16,
    seed=42
)
```

### ステップ3: 実行

```bash
python scripts/suno_stem_arranger.py
```

**完了！** ドラムトラックに自動的に以下が適用されます:
- ✅ tight_rockスタイル（タイトな演奏、rimshot多用）
- ✅ HH Openが裏拍で40%確率
- ✅ セクション冒頭にCrash（Kick協調）
- ✅ 3,7,15小節目にmediumフィル

---

## 📚 主要機能の使い方

### 1. HH Open化（自然なハイハット表現）

```yaml
drums_params:
  open_ratio:
    strong_beat: 0.1   # 1,3拍目でのOpen確率
    weak_beat: 0.2     # 2,4拍目
    off_beat: 0.4      # 裏拍（8分裏）
  
  open_length:
    base_duration: 0.5  # Open時の長さ（quarter beats）
    bpm_dependent: true # BPM依存（高速→短く）
```

**効果**: テンポに応じて自動調整され、人間らしいハイハットワークを実現

### 2. Ghost Notes（ファンク/ジャズ風）

```yaml
drums_params:
  ghost_notes:
    snare_rate: 0.2  # 裏拍に低velocityスネア追加率
  
  ghost_caps:
    max_per_bar: 4          # 1小節最大4個まで
    velocity_threshold: 50  # 50以下をGhostと判定
```

**効果**: グルーヴ感が増し、ファンク/ジャズ風のニュアンスを実現

### 3. Rimshot混合（ロック/パンク風）

```yaml
drums_params:
  rimshot_rate: 0.2  # Snare→Rimshot置換率20%
  
  rim_snare_alternate_rate: 0.3  # 連続snareの2打目をRim化
```

**効果**: タイトなロックサウンド、スネアの表現力向上

### 4. Ride切替（長い曲で自動的に変化）

```yaml
drums_params:
  ride:
    switch_after_seconds: 16.0  # 16秒後からRideに切替開始
    decay_curve: "exp"          # 指数関数的に切替（"linear"も可）
```

**効果**: 曲の進行に合わせてHH→Rideに自然に移行

### 5. Push/Pull Feel（人間的なタイム感）

```yaml
drums_params:
  push_pull:
    push_amount: 0.03  # Snareを微妙に前へ（0.03 quarter beats = 約12ms @ 120BPM）
    pull_amount: 0.02  # Kickを微妙に後ろへ
```

**効果**: 機械的でない、人間らしいタイム感を実現

### 6. フィル（カスタムパターン定義）

```yaml
drums_params:
  fills:
    insert_bars: [3, 7, 15]  # 挿入小節（0-indexed）
    intensity: "heavy"        # light/medium/heavy
    
    # カスタムパターン定義
    patterns:
      heavy:
        - [{ drum: "kick", offset: 0.0, velocity: 90 },
           { drum: "snare", offset: 0.25, velocity: 95 },
           { drum: "kick", offset: 0.5, velocity: 90 },
           { drum: "crash1", offset: 0.75, velocity: 100 }]
```

**効果**: 任意の小節に、任意のパターンのフィルを挿入

---

## 🎨 スタイルプリセット一覧

### tight_rock
```yaml
drums_style: "tight_rock"
```
- タイトなロックドラム
- Rimshot多用（20%）
- Dynamics圧縮強め

### loose_indie
```yaml
drums_style: "loose_indie"
```
- ゆるいインディーロック
- Open HH多め（裏拍60%）
- Push feel強調

### edm_straight
```yaml
drums_style: "edm_straight"
```
- EDM/エレクトロ
- 4つ打ち強調
- Ride早期切替（8秒）

### jazz_swing
```yaml
drums_style: "jazz_swing"
```
- ジャズスイング
- Ride中心（最初から）
- Ghost多用（30%）

### funk_groove
```yaml
drums_style: "funk_groove"
```
- ファンクグルーヴ
- Pedal HH活用
- Accent map（1拍目強調）

---

## 🔧 トラブルシューティング

### Q: 設定が反映されない
**A**: `extra_intent`にパラメータが正しく渡されているか確認
```python
# suno_stem_arranger.py内で確認
print(f"extra_intent: {extra_intent}")
```

### Q: YAMLファイルが読めない
**A**: PyYAMLインストール or Built-in fallbackを使用
```bash
pip install pyyaml
```
YAMLなしでも動作します（Built-inプリセット使用）

### Q: エラーが出る
**A**: 全機能はtry/except包囲済み。エラーログを確認:
```
⚠️ postprocess_density failed: <エラー詳細>
```

### Q: 効果が強すぎる/弱すぎる
**A**: 各パラメータを微調整:
```yaml
# 例: Open HHを控えめに
open_ratio:
  strong_beat: 0.05  # デフォルト0.1から半減
  weak_beat: 0.1
  off_beat: 0.2
```

---

## 📝 ベストプラクティス

### 1. ジャンル別推奨設定

**ロック**: `tight_rock` + `rimshot_rate: 0.2-0.3`
**インディー**: `loose_indie` + `ghost_notes.snare_rate: 0.15`
**EDM**: `edm_straight` + `kick_on_change.emphasis_prob: 0.8`
**ジャズ**: `jazz_swing` + `ride.switch_after_seconds: 0.0`
**ファンク**: `funk_groove` + `pedal_hh.off_beat_rate: 0.4`

### 2. 段階的適用

**最小構成** → 効果確認 → 機能追加

```yaml
# ステップ1: 最小
drums_params:
  open_ratio: { off_beat: 0.3 }

# ステップ2: Ghost追加
drums_params:
  open_ratio: { off_beat: 0.3 }
  ghost_notes: { snare_rate: 0.15 }

# ステップ3: Fill追加
drums_params:
  open_ratio: { off_beat: 0.3 }
  ghost_notes: { snare_rate: 0.15 }
  fills: { insert_bars: [7, 15], intensity: "medium" }
```

### 3. Seedを固定して比較

```python
# 同じseedで比較することで、効果を正確に評価
score1 = arranger.arrange(..., seed=42)  # 設定A
score2 = arranger.arrange(..., seed=42)  # 設定B
```

---

## 🎯 次のステップ

1. **既存の曲で試す**: emotion_profile.yamlを編集して再生成
2. **カスタムプリセット作成**: drums_style_presets.yamlに追加
3. **フィルパターン拡張**: drums_fill_presets.yamlに追加
4. **パラメータ最適化**: 好みに合わせて微調整

---

**ハッピードラミング！** 🥁✨
