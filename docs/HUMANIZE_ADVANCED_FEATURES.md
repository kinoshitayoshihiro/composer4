# 上級ヒューマナイズ機能ガイド（v1.1）

## 概要

「人間味」をさらに高める3つの上級機能を実装しました。すべてKPI安全性を確保しつつ、演奏の自然さを大幅に向上させます。

## 実装済み機能

### 1. ギターストラム自動展開

**目的**: 和音を「同時押し」ではなく、上下方向に"掃く"ように演奏

**設定例** (`configs/plan_humanize.yaml`):

```yaml
guitar:
  strum_width_ms:
    down: 28      # ダウンストラム：低音→高音へ28ms
    up: 18        # アップストラム：高音→低音へ18ms
  strum_group_epsilon_ms: 12  # この範囲内の同時発音を同一和音扱い
  strum_pattern: "beats"      # beats（拍位置判定）| alternate（小節交互）
```

**動作**:

1. 同一タイミング（±12ms以内）の複数ノートを検出
2. 拍位置に応じて方向判定：
   - `beats` モード: 1/3拍目=Down、2/4拍目=Up
   - `alternate` モード: 偶数小節=Down、奇数小節=Up
3. ピッチ順にオフセット適用：
   - Down: 低音→高音へ順次遅延
   - Up: 高音→低音へ順次遅延

**効果**:
- 機械的な同時発音を解消
- ストラム感のある自然な演奏
- ポップス/ロックの定石に合致

---

### 2. ピアノペダル自動挿入（CC64）

**目的**: 長めのノートに確率的にサスティンペダルを追加し、余韻を演出

**設定例**:

```yaml
piano:
  pedal_prob: 0.35        # 35%の確率で挿入
  pedal_min_beats: 0.75   # 0.75拍以上のノートが対象
  pedal_gap_ms: 25        # 離鍵の25ms前にペダルを離す
```

**動作**:

1. ノート長が `pedal_min_beats` 以上の場合に判定
2. `random.random() < pedal_prob` で確率的に挿入
3. タイミング：
   - **On**: ノート開始の少し前（`start - gap * 0.2`）
   - **Off**: ノート終了の少し前（`end - gap`）

**効果**:
- 濁りを防ぎつつ自然な余韻
- フレーズ感の向上
- 機械的な打ち込み感を解消

---

### 3. ストリングスCC11カーブ（Expression）

**目的**: 各ノートに簡易ADSRカーブを描き、弓使いの表現力を演出

**設定例**:

```yaml
strings:
  expr_cc: 11              # Expression CC番号
  expr_base: 84            # ベース値
  expr_depth: 18           # ピークまでの振れ幅
  expr_curve:
    attack_ms: 60          # 立ち上がり時間
    decay_ms: 120          # 減衰時間
    sustain_level: 92      # 維持レベル（0-127）
    release_ms: 120        # リリース時間
```

**動作**:

1. ノート長 > 0.75拍 の場合に適用
2. CC11イベントを4点折れ線で自動生成：

```
t0: expr_base (84)
  ↓ attack_ms (60ms)
t1: expr_base + expr_depth (102) ← ピーク
  ↓ decay_ms (120ms)
t2: sustain_level (92) ← 維持レベル
  ～ 保持 ～
t3: expr_base (84) ← ノート終了 - release_ms
```

**効果**:
- 弓の入り/抜けを自然に表現
- ベタ打ちの平坦さを解消
- 音量変化による生っぽさ

---

## KPI安全性の確認

| 機能 | 追加イベント | ノート数への影響 | 密度への影響 | バックビート |
|------|------------|---------------|------------|------------|
| ギターストラム | なし（タイミングのみ） | 不変 | 不変 | 不変 |
| ピアノペダル | CC64 × 2/ノート | 不変 | 不変 | 不変 |
| ストリングスCC11 | CC11 × 4/ノート | 不変 | 不変 | 不変 |

**結論**: すべてKPI指標（ノート数、密度、バックビート）を保持します。

---

## 無効化方法

すべての機能はYAML設定で無効化可能：

```yaml
guitar:
  strum_width_ms:
    down: 0   # ストラム無効
    up: 0

piano:
  pedal_prob: 0.0   # ペダル無効

strings:
  expr_depth: 0     # CC11無効
```

---

## 使用例

### 標準的な実行（すべての機能有効）

```bash
python3 scripts/midi_writer.py \
  --plan song_packages/suno_project/song_001/full_arrangement.json \
  --config configs/plan_humanize.yaml \
  --bars song_packages/suno_project/song_001/bars.parquet \
  --out song_packages/suno_project/song_001/full_arrangement_human.mid
```

### 効果の確認

```bash
# CCイベント数の確認
python3 -c "
from mido import MidiFile
mid = MidiFile('full_arrangement_human.mid')
for i, tr in enumerate(mid.tracks):
    name = next((m.name for m in tr if m.type == 'track_name'), f'Track {i}')
    cc_count = sum(1 for msg in tr if msg.type == 'control_change')
    note_count = sum(1 for msg in tr if msg.type == 'note_on')
    print(f'{name}: {note_count} notes, {cc_count} CC events')
"
```

**期待される出力例**:

```
Drums: 850 notes, 0 CC events
Bass: 320 notes, 0 CC events
Guitar: 450 notes, 0 CC events  ← ストラムはタイミング変更のみ
Piano: 280 notes, 156 CC events  ← ペダル（35%確率で挿入）
Strings: 190 notes, 520 CC events ← CC11カーブ（1ノート約4イベント）
```

---

## トラブルシューティング

### Q: ギターのストラムが効いていない

**A**: 以下を確認：
1. 和音が存在するか（単音では発動しない）
2. `strum_group_epsilon_ms` 内に複数ノートがあるか
3. `strum_width_ms.down/up` が 0 でないか

### Q: ピアノペダルが多すぎる/少なすぎる

**A**: `pedal_prob` を調整（0.0〜1.0）。推奨範囲は 0.2〜0.5。

### Q: ストリングスのCC11が密すぎて重い

**A**: `expr_depth` を下げる、または 0 にして無効化。

### Q: CCイベントが多すぎてファイルサイズが大きい

**A**: 以下のいずれかで対応：
- `expr_depth` を控えめに（18→12など）
- `pedal_prob` を下げる（0.35→0.2など）
- 特定の楽器のみ有効化（他は無効化）

---

## パラメータチューニングガイド

### ギターストラム

| パラメータ | 初期値 | 推奨範囲 | 効果 |
|----------|-------|---------|------|
| `down` | 28ms | 20-40ms | 広げるとゆったり、狭めるとタイト |
| `up` | 18ms | 10-30ms | Down より短めが自然 |
| `epsilon` | 12ms | 8-20ms | 大きいほど同一和音として扱いやすい |

### ピアノペダル

| パラメータ | 初期値 | 推奨範囲 | 効果 |
|----------|-------|---------|------|
| `pedal_prob` | 0.35 | 0.2-0.6 | 高いほど頻繁に挿入 |
| `pedal_min_beats` | 0.75 | 0.5-2.0 | 短いノートにも適用したい場合は下げる |
| `pedal_gap_ms` | 25ms | 15-40ms | 濁り防止の余白 |

### ストリングスCC11

| パラメータ | 初期値 | 推奨範囲 | 効果 |
|----------|-------|---------|------|
| `expr_depth` | 18 | 10-30 | 表情の強さ |
| `attack_ms` | 60ms | 40-120ms | 弓の入り速度 |
| `decay_ms` | 120ms | 80-200ms | ピーク後の減衰 |
| `sustain_level` | 92 | 80-110 | 維持レベル |
| `release_ms` | 120ms | 80-200ms | 弓の抜き速度 |

---

## A/Bテスト推奨手順

1. **ベースライン生成**（上級機能OFF）

```yaml
# plan_humanize_baseline.yaml
guitar:
  strum_width_ms: {down: 0, up: 0}
piano:
  pedal_prob: 0.0
strings:
  expr_depth: 0
```

```bash
python3 scripts/midi_writer.py --plan ... --config configs/plan_humanize_baseline.yaml --out baseline.mid
```

2. **上級機能版生成**（標準設定）

```bash
python3 scripts/midi_writer.py --plan ... --config configs/plan_humanize.yaml --out advanced.mid
```

3. **聴き比べ**
   - ギター：和音のストラム感
   - ピアノ：余韻と響き
   - ストリングス：弓使いの自然さ

4. **KPI確認**（念のため）

```bash
python3 scripts/kpi_gate.py --mid baseline.mid --bars bars.parquet --report baseline_kpi.json
python3 scripts/kpi_gate.py --mid advanced.mid --bars bars.parquet --report advanced_kpi.json
```

---

## 更新履歴

- **2025-11-01 v1.1**: 上級機能実装
  - ✅ ギターストラム自動展開
  - ✅ ピアノペダル確率挿入
  - ✅ ストリングスCC11 ADSRカーブ
  - イベント統合処理（Note + CC混在対応）

---

## 参考資料

- [役割/セクション別ヒューマナイズガイド](./HUMANIZE_ROLE_SECTION.md)
- [KPI Gate仕様](./KPI_GATE_SPEC.md)
- [MIDI Writer実装詳細](./研究手法_⇔_実装_対応表（v_2025_11_01_）.md)
