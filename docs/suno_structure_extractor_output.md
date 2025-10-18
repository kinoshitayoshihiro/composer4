# Suno構造抽出器のサンプル出力

このYAMLは `extract_structure.py` によって生成される構造情報のサンプルです。

```yaml
tempo_map:
  global_tempo: 120.0
  beat_times:
    - 0.0
    - 0.5
    - 1.0
    - 1.5
    # ... 全ビート位置（秒単位）
  downbeat_times:
    - 0.0
    - 2.0
    - 4.0
    # ... 小節頭の位置
  time_signature: [4, 4]

sections:
  - label: Intro
    start_time: 0.0
    end_time: 8.0
    start_measure: 0
    duration_measures: 4
  - label: Verse
    start_time: 8.0
    end_time: 24.0
    start_measure: 4
    duration_measures: 8
  - label: Chorus
    start_time: 24.0
    end_time: 40.0
    start_measure: 12
    duration_measures: 8
  - label: Bridge
    start_time: 40.0
    end_time: 48.0
    start_measure: 20
    duration_measures: 4
  - label: Chorus
    start_time: 48.0
    end_time: 64.0
    start_measure: 24
    duration_measures: 8

chords:
  Intro:
    - time: 0.0
      chord: C
      duration: 2.0
    - time: 2.0
      chord: F
      duration: 2.0
  Verse:
    - time: 8.0
      chord: C
      duration: 2.0
    - time: 10.0
      chord: G
      duration: 2.0
    - time: 12.0
      chord: Am
      duration: 2.0
    - time: 14.0
      chord: F
      duration: 2.0
  Chorus:
    - time: 24.0
      chord: F
      duration: 2.0
    - time: 26.0
      chord: C
      duration: 2.0
    - time: 28.0
      chord: G
      duration: 2.0

drums_hits:
  kick:
    - 0.0
    - 0.5
    - 1.0
    - 1.5
    # ... 全キック位置
  snare:
    - 0.5
    - 1.5
    - 2.5
    # ... 全スネア位置
  hihat:
    - 0.0
    - 0.25
    - 0.5
    - 0.75
    # ... 全ハイハット位置

bass_contour:
  Verse:
    - time: 8.0
      pitch: 48    # MIDI note (C3)
      duration: 0.5
      velocity: 80
    - time: 8.5
      pitch: 48
      duration: 0.5
      velocity: 75
    - time: 9.0
      pitch: 55    # G3
      duration: 0.5
      velocity: 80
  Chorus:
    - time: 24.0
      pitch: 53    # F3
      duration: 0.5
      velocity: 85
    - time: 24.5
      pitch: 53
      duration: 0.5
      velocity: 80
```

## 使用方法

### 基本的な使用

```bash
python scripts/audio2score/extract_structure.py \
    --vocal data/suno_stems/song1/vocal.wav \
    --accomp data/suno_stems/song1/accomp.wav \
    --output data/suno_structures/song1.yaml
```

### 伴奏のみ（フルミックス）

```bash
python scripts/audio2score/extract_structure.py \
    --accomp data/suno_stems/song1/full_mix.wav \
    --output data/suno_structures/song1.yaml
```

### オプション

- `--sr 44100`: サンプリングレート変更（デフォルト: 22050）
- `--n-sections 7`: セクション分割数変更（デフォルト: 5）
- `--quiet`: 詳細ログ抑制

## 出力構造の詳細

### tempo_map
- **global_tempo**: グローバルBPM（float）
- **beat_times**: 全ビート位置の配列（秒単位）
- **downbeat_times**: 小節頭（ダウンビート）の配列
- **time_signature**: 拍子記号 `[分子, 分母]`（デフォルト: [4, 4]）

### sections
各セクションは以下の情報を持つ：
- **label**: セクション名（Intro/Verse/Chorus/Bridge/Outro）
- **start_time**: 開始時刻（秒）
- **end_time**: 終了時刻（秒）
- **start_measure**: 開始小節番号
- **duration_measures**: セクションの長さ（小節数）

### chords
セクションごとにグループ化されたコード進行：
- **time**: コード変化の時刻（秒）
- **chord**: コード名（C, F, G, Am, etc.）
- **duration**: コード持続時間（秒）

### drums_hits
ドラムヒットの位置（秒単位の配列）：
- **kick**: キックドラム
- **snare**: スネアドラム
- **hihat**: ハイハット

### bass_contour
セクションごとのベースライン：
- **time**: ノート開始時刻（秒）
- **pitch**: MIDI pitch番号（28-64: E1-E4）
- **duration**: ノート持続時間（秒）
- **velocity**: ベロシティ（0-127）

## Stage2 Generatorとの連携

このYAML構造は、次のステップ（Todo #6）の `arrange_from_yaml.py` で使用されます：

1. **tempo_map** → 全Generator共通のテンポ設定
2. **sections** → セクションごとに適切なGenerator呼び出し
3. **chords** → CompingGenerator, GuitarGeneratorのコード進行指定
4. **drums_hits** → BassGeneratorのkick sync参照
5. **bass_contour** → BassGeneratorのpitch hint（オプション）

## 技術的な詳細

### Tempo抽出
- `librosa.beat.beat_track()` で動的テンポトラッキング
- ビート検出失敗時は120 BPMのデフォルトfallback

### Section分割
- Chromagram特徴量からrecurrence matrix作成
- Laplacian segmentationでセクション境界検出
- ヒューリスティックでセクションラベル推定

### Chord推定
- Chromagram（CQT）からroot note検出
- 簡易版：major/minor区別なし（将来拡張可能）

### Drum hits
- Onset detection + 周波数帯域フィルタリング
  - Kick: 20-120 Hz
  - Snare: 150-300 Hz
  - Hihat: 8000-16000 Hz

### Bass contour
- 低域（40-250 Hz）フィルタリング
- pYIN algorithmでピッチトラッキング
- Voiced segments抽出 → MIDI pitch変換
