# Emotion Profile & Humanize 機能の使い方

## 概要

Suno AI stem アレンジャーに**Emotion Profile** と **Humanize** 機能が追加されました。

### 主な機能

1. **コード自動推定** (優先度★★)
   - Suno stem WAV から自動的にコード進行を推定
   
2. **Humanize 機能** (優先度★)
   - タイミングのゆらぎ（±ms）
   - ベロシティのゆらぎ（標準偏差）
   - 5つの感情プロファイル対応

## 使い方

### 基本実行

```bash
python scripts/suno_stem_arranger.py \
  --input data/suno_stems/my_song \
  --output data/arranged_midi \
  --emotion energetic \
  --humanize
```

### 感情プロファイル指定

```bash
python scripts/suno_stem_arranger.py \
  --input data/suno_stems/ballad \
  --emotion melancholic \
  --emotion-profile configs/emotion_profile.yaml
```

### Humanize無効化

```bash
python scripts/suno_stem_arranger.py \
  --input data/suno_stems/my_song \
  --no-humanize
```

### 乱数シード指定（再現性）

```bash
python scripts/suno_stem_arranger.py \
  --input data/suno_stems/my_song \
  --seed 42 \
  --emotion calm
```

## 感情プロファイル

`configs/emotion_profile.yaml` に定義された5つの感情：

| 感情 | timing_ms | vel_sigma | 特徴 |
|-----|-----------|-----------|------|
| **energetic** | 10ms | 7 | 元気・活発・強いゆらぎ |
| **melancholic** | 8ms | 6 | 哀愁・メランコリック |
| **calm** | 6ms | 4 | 穏やか・静か・小さいゆらぎ |
| **aggressive** | 12ms | 8 | 激しい・最大のゆらぎ |
| **romantic** | 8ms | 6 | ロマンチック |

### パラメータ詳細

各感情プロファイルには以下が定義されています：

```yaml
emotions:
  energetic:
    humanize:
      timing_ms: 10        # タイミングのゆらぎ（±ms）
      vel_sigma: 7         # ベロシティの標準偏差
    density_multipliers:   # パート別密度
      drums:
        hh: 1.20          # ハイハット
        ghost: 1.30       # ゴーストノート
      bass: 1.10
      guitar: 1.15
      piano: 1.20
      strings: 1.05
    velocity_shift:        # ベロシティ調整
      drums: 4
      bass: 2
      guitar: 3
      piano: 3
      strings: -2
    swing:
      eighth: 0.04        # 8分音符のスイング
```

## CLI オプション

| オプション | 型 | デフォルト | 説明 |
|-----------|---|-----------|------|
| `--input` | Path | (必須) | Suno stem WAVディレクトリ |
| `--output` | Path | `data/arranged_midi` | 出力MIDIディレクトリ |
| `--tempo` | float | 120 | テンポ (BPM) |
| `--emotion` | str | `energetic` | 感情 (energetic/calm/melancholic/hopeful/intense) |
| `--bars` | int | 16 | 生成小節数 |
| `--emotion-profile` | str | `configs/emotion_profile.yaml` | 感情プロファイルYAMLパス |
| `--seed` | int | None | 乱数シード（再現性用） |
| `--humanize` | flag | True | Humanize機能を有効化 |
| `--no-humanize` | flag | - | Humanize機能を無効化 |

## 実装詳細

### Humanize処理

1. **タイミングのゆらぎ**
   ```python
   # ms → quarter length 変換
   ms_per_quarter = 60000.0 / tempo_bpm
   timing_ql = timing_ms / ms_per_quarter
   
   # 各ノートにランダムなオフセット
   offset_shift = random.uniform(-timing_ql, timing_ql)
   note.offset += offset_shift
   ```

2. **ベロシティのゆらぎ**
   ```python
   # ガウス分布でベロシティを変化
   vel_shift = int(random.gauss(0, vel_sigma))
   new_vel = max(1, min(127, note.volume.velocity + vel_shift))
   ```

### extra_intent の適用

各楽器ジェネレーターに以下の追加パラメータが渡されます：

```python
extra_intent = {
    "density_multipliers": {...},  # パート別密度調整
    "velocity_shift": {...},       # ベロシティ調整
    "swing": {...}                 # スイング
}
```

## カスタム感情プロファイル

独自のプロファイルを作成する場合：

1. `configs/emotion_profile.yaml` をコピー
2. 新しい感情を追加：

```yaml
emotions:
  my_emotion:
    humanize:
      timing_ms: 7
      vel_sigma: 5
    density_multipliers:
      drums: {hh: 1.1, ghost: 1.2}
      bass: 1.0
      guitar: 1.1
      piano: 1.1
      strings: 1.0
    velocity_shift:
      drums: 3
      bass: 1
      guitar: 2
      piano: 2
      strings: 0
    swing:
      eighth: 0.02
```

3. 実行時に指定：

```bash
python scripts/suno_stem_arranger.py \
  --emotion my_emotion \
  --emotion-profile configs/my_custom_profile.yaml
```

## 技術詳細

### 1. パート毎に決定的RNG（乱数相関解消）

同じseedでも各パートが独立した乱数系列を持つように実装：

```python
# seed + part名からMD5ハッシュ生成
part_tag = getattr(part, "id", getattr(part, "partName", "part"))
h = hashlib.md5(f"{seed}:{part_tag}".encode()).hexdigest()
rng = random.Random(int(h[:8], 16))

# パート固有のローカルRNGを使用
offset_shift = rng.uniform(-timing_ql, timing_ql)
vel_shift = int(rng.gauss(0, vel_sigma))
```

**メリット**:
- 再現性: 同じseedで同じ結果
- 独立性: パート間で乱数が相関しない
- 自然さ: 各パートが独自の"演奏クセ"を持つ

### 2. 負のオフセット回避（安全性保証）

ゆらぎ適用後、offset < 0.0の場合は0.0にクランプ：

```python
new_off = n.offset + offset_shift
n.offset = new_off if new_off >= 0.0 else 0.0
```

**保証事項**:
- すべてのノートoffset >= 0.0
- music21/MIDIエクスポートエラーを防止
- ごく稀な先行ズレを回避

### 3. swing.eighth簡易適用（表情付け）

8分音符の裏拍をわずかに後方へ押してスウィング感を演出：

```python
def _apply_swing_eighths(self, part, swing_ratio, tempo_bpm):
    eighth = 0.5  # 8分音符 = 0.5QL
    push = swing_ratio * (eighth * 0.5)  # 裏を後ろへ
    
    for n in list(part.flatten().notes):
        pos = n.offset / eighth
        # 裏拍判定（位置が "…+0.5" に近い）
        if abs((pos % 1.0) - 0.5) < 1e-6:
            n.offset += push
```

**適用条件**:
- emotion_profile.yamlで `swing.eighth` が定義されている場合のみ
- デフォルト値0.0なら無変更（後方互換）
- 4/4拍子前提（将来拡張可能）

**スウィング量の目安**:
- 0.00: 無変更（ストレート）
- 0.02: ごく軽いスウィング
- 0.04: 軽いスウィング（energetic推奨）
- 0.08: 中程度のスウィング
- 0.15: 強いスウィング（ジャズ風）

## トラブルシューティング

### YAMLファイルが見つからない

```
WARNING: Could not load emotion profile: [Errno 2] No such file or directory
INFO: Using default humanize params: timing_ms=8.0, vel_sigma=5.0
```

→ デフォルト値で実行されます（問題なし）

### 感情が定義されていない

```
WARNING: Emotion 'unknown_emotion' not found in profile
INFO: Using default humanize params: timing_ms=8.0, vel_sigma=5.0
```

→ デフォルト値で実行されます

### Humanize無効化したい

```bash
python scripts/suno_stem_arranger.py --no-humanize
```

### 再現性を確認したい

同じseedで2回実行し、結果を比較：

```bash
# 1回目
python scripts/suno_stem_arranger.py \
  --input data/suno_stems/test \
  --output out/arr1 \
  --seed 123 \
  --emotion energetic

# 2回目（同じseed）
python scripts/suno_stem_arranger.py \
  --input data/suno_stems/test \
  --output out/arr2 \
  --seed 123 \
  --emotion energetic

# 比較
diff <(python -m music21.midi out/arr1/*.mid) \
     <(python -m music21.midi out/arr2/*.mid)
# → ノートのoffset/velocityが一致
```

## 出力例

```
INFO: Loading emotion profile from: configs/emotion_profile.yaml
INFO: Humanize enabled: timing_ms=10.0ms, vel_sigma=7.0
INFO: Generating arrangement with 5 generators...
INFO: Generating drums...
INFO: Generating bass...
INFO: Generating piano...
INFO: Generating guitar...
INFO: Generating strings...
💾 Saved to: data/arranged_midi/my_song_arranged.mid
```

## 今後の拡張

- [ ] provenance.json へのパラメータ記録
- [ ] 感情間の自動遷移（A→B section）
- [ ] MIDI CC (Expression, Modulation) 対応
- [ ] パート別個別Humanize設定
