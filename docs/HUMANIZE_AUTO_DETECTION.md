# 自動判定機能ガイド（v1.2）

## 概要

「ギターストラム方向の自動判定」と「ピアノペダルのコードチェンジ連動」を実装しました。
音楽理論に基づく自動判定により、より自然で文脈に沿った演奏表現を実現します。

## 新機能

### 1. ギターストラム方向の自動判定

**目的**: 拍子・拍位置・セクション・コードチェンジを総合的に判断し、自然なストローク方向を自動決定

#### モード設定

```yaml
guitar:
  strum_direction_mode: auto  # 推奨：自動判定
  # その他のモード:
  #   beats        : 1/3拍=down, 2/4拍=up（固定）
  #   alternate    : 小節ごとに down/up を交互
  #   fixed_down   : 常にdown
  #   fixed_up     : 常にup
```

#### 自動判定の優先順位（`auto`モード）

1. **コードチェンジ優先**
   ```yaml
   strum_on_change_override: down  # コード変化時は down を優先
   # down / up / none（優先なし）
   ```

2. **セクション別バイアス**
   ```yaml
   section_direction_bias:
     chorus: down  # Chorusは力強いdown中心
     verse: down   # Verseも安定したdown
     bridge: up    # Bridgeは流れるようなup
   ```

3. **拍子別ヒューリスティク**
   - **4/4拍子**: 1拍目・3拍目=Down、2拍目・4拍目=Up
   - **3/4拍子**: 1拍目=Down、2・3拍目=Up
   - **6/8拍子**: 1拍目・4拍目=Down、その他=Up

#### 動作例

```
[Chorus, 4/4拍子]
小節1:
  1拍目: Cmaj → **Down**（コードチェンジ＋強拍）
  2拍目: Cmaj → Up（弱拍）
  3拍目: Gmaj → **Down**（コードチェンジ＋強拍）
  4拍目: Gmaj → Up（弱拍）

小節2（Bridge開始）:
  1拍目: Fmaj → **Down**（コードチェンジ優先）
  2拍目: Fmaj → Up（セクションバイアス + 弱拍）
```

---

### 2. ピアノペダルのコードチェンジ連動

**目的**: 確率的なペダルから、和声変化に連動した音楽的なペダリングへ

#### モード設定

```yaml
piano:
  pedal_mode: chord_change  # 推奨：コードチェンジ連動
  # probabilistic : 従来の確率方式（後方互換）
```

#### `chord_change` モードのパラメータ

```yaml
piano:
  pedal_mode: chord_change
  pedal_on_anticipation_ms: 12    # 和音変化の少し前から踏む
  pedal_release_ms: 35            # 変化直後に放す（濁り防止）
  pedal_hold_min_beats: 0.50      # 最小保持（短い和音の下限）
  pedal_hold_max_beats: 2.50      # 最大保持（伸びすぎ抑制）
```

#### 動作フロー

1. **コード変化点検出**
   - イベントの`chord`フィールドから変化を検出
   - 連続同一コードは統合（重複排除）

2. **セグメント化**
   ```
   Cmaj (小節1-2) → Gmaj (小節3) → Fmaj (小節4-5)
   └─ セグメント1 ─┘  └─ セグメント2 ─┘  └─ セグメント3 ─┘
   ```

3. **ペダルタイミング計算**
   ```
   セグメント1 (Cmaj):
     On  : 小節1の開始 - 12ms (anticipation)
     Off : 小節3の開始 - 35ms (release)
     Hold: 2小節分（clamp: 0.5〜2.5拍）
   ```

#### 効果

- **濁りの防止**: 和音変化直前にペダルを離す
- **音楽的な余韻**: コード内で自然な響き
- **KPI安全**: ノート数不変、CCイベントのみ追加

---

## 使用例

### 基本実行（すべて有効）

```bash
python3 scripts/midi_writer.py \
  --plan song_packages/suno_project/song_001/full_arrangement.json \
  --config configs/plan_humanize.yaml \
  --bars song_packages/suno_project/song_001/bars.parquet \
  --out song_packages/suno_project/song_001/full_arrangement_auto.mid
```

### 効果確認

#### ギターストラム方向の確認

```bash
# DAWのピアノロールで視認
# - コードチェンジ箇所でDown優先
# - Chorusで力強いDown傾向
# - 弱拍でUp傾向
```

#### ピアノペダルの確認

```python
from mido import MidiFile

mid = MidiFile('full_arrangement_auto.mid')
for i, tr in enumerate(mid.tracks):
    name = next((m.name for m in tr if m.type == 'track_name'), f'Track {i}')
    if 'Piano' not in name:
        continue
    
    cc64_events = [msg for msg in tr if msg.type == 'control_change' and msg.control == 64]
    print(f'{name}: {len(cc64_events)} CC64 events')
    
    # CC64の On/Off タイミングを確認
    tick = 0
    for msg in cc64_events[:10]:  # 最初の10イベント
        tick += msg.time
        print(f'  tick {tick}: CC64 = {msg.value}')
```

**期待される出力例**（`chord_change`モード）:

```
Piano: 48 CC64 events  ← コード変化の2倍（On/Off）
  tick 0: CC64 = 127     ← Cmaj開始
  tick 3820: CC64 = 0    ← Cmaj終了
  tick 3840: CC64 = 127  ← Gmaj開始
  tick 5740: CC64 = 0    ← Gmaj終了
  ...
```

---

## 後方互換性

### 既存設定との互換性

| 設定 | v1.1以前 | v1.2 | 動作 |
|------|---------|------|------|
| `strum_pattern: beats` | 使用中 | **非推奨** | `strum_direction_mode: beats` に自動変換 |
| `strum_direction_mode` 未設定 | - | デフォルト:`auto` | 自動判定が有効 |
| `pedal_mode` 未設定 | - | デフォルト:`chord_change` | コード連動が有効 |
| `pedal_prob` のみ設定 | 使用中 | 有効（`probabilistic`） | 確率方式を維持 |

### 旧動作への戻し方

```yaml
guitar:
  strum_direction_mode: beats  # 固定パターンに戻す

piano:
  pedal_mode: probabilistic   # 確率方式に戻す
  pedal_prob: 0.35
```

---

## パラメータチューニング

### ギターストラム自動判定

| パラメータ | デフォルト | 推奨範囲 | 効果 |
|----------|----------|---------|------|
| `strum_on_change_override` | `down` | `down`/`up`/`none` | コード変化時の方向優先 |
| `section_direction_bias.chorus` | `down` | `down`/`up`/`none` | Chorus での傾向 |
| `section_direction_bias.bridge` | `up` | `down`/`up`/`none` | Bridge での傾向 |

### ピアノペダル（コードチェンジ）

| パラメータ | デフォルト | 推奨範囲 | 効果 |
|----------|----------|---------|------|
| `pedal_on_anticipation_ms` | 12ms | 5-20ms | 踏み始めの先行時間 |
| `pedal_release_ms` | 35ms | 20-50ms | 離す余白（濁り防止） |
| `pedal_hold_min_beats` | 0.50 | 0.25-1.0 | 最小保持時間 |
| `pedal_hold_max_beats` | 2.50 | 2.0-4.0 | 最大保持時間 |

---

## トラブルシューティング

### Q: ギターストラムの方向が期待と違う

**A**: 以下を確認：
1. イベントに`chord`フィールドがあるか
2. `section_direction_bias`の設定
3. `strum_on_change_override`の値
4. デバッグ用に`strum_direction_mode: fixed_down`で固定してみる

### Q: ピアノペダルが挿入されない（chord_changeモード）

**A**: イベントに`chord`フィールドが必要です。
- Planに`chord`情報がない場合は`pedal_mode: probabilistic`に変更
- または、コード推定処理を事前に実行

### Q: ペダルが長すぎる/短すぎる

**A**: 以下で調整：
- 長い場合: `pedal_hold_max_beats`を下げる（2.5→2.0）
- 短い場合: `pedal_hold_min_beats`を上げる（0.5→0.75）

### Q: KPIへの影響が心配

**A**: すべて安全です：
- ギター: タイミング変更のみ（ノート数不変）
- ピアノ: CCイベント追加のみ（ノート数不変）
- 密度・バックビート指標は影響なし

---

## A/Bテスト推奨手順

### 1. ベースライン（v1.1相当）

```yaml
# plan_humanize_v11.yaml
guitar:
  strum_direction_mode: beats  # 固定パターン

piano:
  pedal_mode: probabilistic
  pedal_prob: 0.35
```

### 2. 自動判定版（v1.2）

```yaml
# plan_humanize.yaml（デフォルト）
guitar:
  strum_direction_mode: auto

piano:
  pedal_mode: chord_change
```

### 3. 生成と比較

```bash
# v1.1相当
python3 scripts/midi_writer.py --plan ... --config configs/plan_humanize_v11.yaml --out v11.mid

# v1.2（自動判定）
python3 scripts/midi_writer.py --plan ... --config configs/plan_humanize.yaml --out v12.mid
```

### 4. 聴き比べポイント

- **ギター**: コードチェンジでのストローク方向の自然さ
- **ピアノ**: 和音変化でのペダリングの音楽的妥当性
- **全体**: 濁りや不自然な響きがないか

---

## 実装詳細

### ギターストラム方向判定のアルゴリズム

```python
if mode == "auto":
    # 1. コード変化を最優先
    if chord_changed and on_change_override in ("down", "up"):
        direction = on_change_override
    else:
        # 2. セクションバイアス
        if section_bias in ("down", "up"):
            direction = section_bias
        else:
            # 3. 拍子ヒューリスティク
            if timesig == (3, 4):
                direction = "down" if beat == 1 else "up"
            elif timesig == (6, 8):
                direction = "down" if beat in (1, 4) else "up"
            else:  # 4/4など
                direction = "down" if beat in (1, 3) else "up"
```

### ピアノペダル（chord_change）のアルゴリズム

```python
# 1. コード時系列抽出
chord_points = [(start_beats, chord, channel) for event in events if event.chord]

# 2. 連続同一コードの重複排除
dedup = [(t, c, ch) for t, c, ch in chord_points if c != prev_chord]

# 3. セグメント化とペダルタイミング計算
for i, (t_on, chord, ch) in enumerate(dedup):
    t_next = dedup[i+1][0] if i+1 < len(dedup) else song_end
    hold = clamp(t_next - t_on - release_ms, hold_min, hold_max)
    
    cc_events.append((t_on - anticipation_ms, 64, 127, ch))  # On
    cc_events.append((t_on + hold, 64, 0, ch))               # Off
```

---

## 更新履歴

- **2025-11-01 v1.2**: 自動判定機能実装
  - ✅ ギターストラム方向の自動判定（拍子・コード・セクション統合）
  - ✅ ピアノペダルのコードチェンジ連動モード
  - 後方互換性維持（`beats`/`probabilistic`モード継続サポート）

- **2025-11-01 v1.1**: 上級機能実装
  - ギターストラム自動展開
  - ピアノペダル確率挿入
  - ストリングスCC11 ADSRカーブ

---

## 参考資料

- [上級ヒューマナイズ機能ガイド](./HUMANIZE_ADVANCED_FEATURES.md)
- [役割/セクション別ヒューマナイズガイド](./HUMANIZE_ROLE_SECTION.md)
- [KPI Gate仕様](./KPI_GATE_SPEC.md)
