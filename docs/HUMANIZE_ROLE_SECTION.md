# 役割/セクション別ヒューマナイズ実装ガイド

## 概要

「人間味＝機械っぽさを減らす工夫」を5本柱で実現：

1. **時間の揺れ** - タイミングのマイクロシフト（±6〜9ms、役割別調整）
2. **強弱とアタック** - ベロシティのジッター＋セクション別シフト
3. **音価とアーティキュレーション** - ノート長の微調整（90〜98%）
4. **セクションごとの抑揚** - Intro/Chorus/Bridge別の表現力変化
5. **奏法のクセ** - ドラム前ノリ/後ノリ、ギターストラム（将来拡張）

## 実装済み機能

### ✅ 役割別ヒューマナイズ（Role-based）

- **Drums**
  - タイミング揺れ: ±6ms
  - ベロシティ揺れ: ±6
  - **HH前ノリ**: -6ms（キックより早く）
  - **スネア後ノリ**: +9ms（レイドバック・タメ感）
  
- **Bass**
  - タイミング揺れ: ±6ms
  - ベロシティ揺れ: ±5
  - **ノート長変動**: 92〜98%（レガート/スタッカートの自然な変化）

- **Guitar**
  - タイミング揺れ: ±7ms
  - ベロシティ揺れ: ±7
  - **ストラム幅**: Down=28ms, Up=18ms（将来のコード展開用）

- **Piano**
  - タイミング揺れ: ±6ms
  - ベロシティ揺れ: ±7
  - **アルペジオ範囲**: 5〜15ms（和音の微妙なばらし）
  - **ペダル確率**: 35%（将来のCC64自動挿入用）

- **Strings**
  - タイミング揺れ: ±5ms
  - ベロシティ揺れ: ±5
  - **エクスプレッション深さ**: 18（将来のCC11カーブ用）

### ✅ セクション別ニュアンス（Section-based）

- **Intro**
  - タイミングスケール: 1.2×（揺れを20%増幅→ゆったり）
  - ベロシティシフト: -4（控えめ）

- **Verse**
  - タイミングスケール: 1.0×（標準）
  - ベロシティシフト: 0

- **Chorus**
  - タイミングスケール: 0.8×（タイト、引き締め）
  - ベロシティシフト: +8（力強く）

- **Bridge**
  - タイミングスケール: 1.0×（標準）
  - ベロシティシフト: +4（やや強調）

## 使用方法

### 基本コマンド（セクション情報なし）

```bash
python3 scripts/midi_writer.py \
  --plan song_packages/suno_project/song_001/full_arrangement.json \
  --config configs/plan_humanize.yaml \
  --out song_packages/suno_project/song_001/full_arrangement_human.mid
```

### 推奨コマンド（セクション別ヒューマナイズ有効）

```bash
python3 scripts/midi_writer.py \
  --plan song_packages/suno_project/song_001/full_arrangement.json \
  --config configs/plan_humanize.yaml \
  --bars song_packages/suno_project/song_001/bars.parquet \
  --out song_packages/suno_project/song_001/full_arrangement_human.mid
```

`--bars` オプションで `bars.parquet` を指定すると、`bar_index` → `section_label` マッピングが有効化され、Intro/Chorus等のセクション別ニュアンスが自動適用されます。

### 上級機能の確認

生成されたMIDIファイルで以下を確認：

- **ギターストラム**: 和音が数十ms内で展開されているか（DAWのピアノロールで視認可能）
- **ピアノペダル**: CC64イベントが挿入されているか（長めのノート付近）
- **ストリングスCC11**: Expression カーブが描かれているか（4点の折れ線）

```bash
# MIDIファイルの詳細確認（Python）
python3 -c "
from mido import MidiFile
mid = MidiFile('full_arrangement_human.mid')
for i, tr in enumerate(mid.tracks):
    cc_count = sum(1 for msg in tr if msg.type == 'control_change')
    note_count = sum(1 for msg in tr if msg.type == 'note_on')
    print(f'Track {i}: {note_count} notes, {cc_count} CC events')
"
```

## 設定のカスタマイズ

### `configs/plan_humanize.yaml` の編集

#### 役割別パラメータ調整例

```yaml
humanize:
  roles:
    drums:
      timing_ms: 8           # 揺れを大きく（デフォルト6→8）
      hh_microshift_ms: -8   # HHをより前ノリに（-6→-8）
      snare_layback_ms: 12   # スネアをさらにタメる（9→12）
```

#### セクション別パラメータ調整例

```yaml
humanize:
  sections:
    chorus:
      timing_scale: 0.7      # さらにタイトに（0.8→0.7）
      vel_shift: 12          # より力強く（8→12）
```

## KPIへの影響（安全性の確認）

### ✅ KPI影響なし（±10ms/±10Vel以内）

- **密度（Density）**: ノート総数は変わらない
- **バックビート**: スネアの拍位置は微調整のみ（±9ms）
- **ノーツ数**: イベント数は維持
- **曲長**: `song_end_beats` クリップで保証

### 調整の安全域

| パラメータ | 安全範囲 | 推奨初期値 |
|----------|---------|----------|
| `timing_ms` | ±10ms以内 | ±6〜9ms |
| `vel_jitter` | ±10以内 | ±5〜7 |
| `vel_shift` (section) | ±12以内 | -4〜+8 |
| `noteoff_ratio_range` | 0.90〜1.00 | 0.92〜0.98 |

## 今後の拡張（将来実装予定）

### ✅ ギターストラム自動展開（実装済み）

和音を構成するノートを上下方向に"掃く"ように数十ms内で展開します。

```yaml
guitar:
  strum_width_ms:
    down: 28   # Down strokeで和音を28msかけて展開
    up: 18     # Up strokeは18ms
  strum_group_epsilon_ms: 12  # この範囲内の同時発音を同一和音扱い
  strum_pattern: "beats"      # beats（1/3拍=down, 2/4拍=up）| alternate（小節交互）
```

**動作**:
- 同一タイミング（±12ms以内）の複数ノートを検出
- 拍位置に応じて方向判定（1/3拍目=Down、2/4拍目=Up）
- ピッチ順にオフセット（Down: 低→高、Up: 高→低）

### ✅ ピアノペダル自動挿入（実装済み）

長めのノートに確率的にCC64（サスティンペダル）を挿入します。

```yaml
piano:
  pedal_prob: 0.5         # 50%の確率でCC64挿入
  pedal_min_beats: 1.0    # 1拍以上の長さで
  pedal_gap_ms: 25        # 離鍵の25ms前にペダルを離す（濁り防止）
```

**動作**:
- `dur_beats >= pedal_min_beats` のノートが対象
- `random.random() < pedal_prob` で確率判定
- ノート開始少し前にCC64=127、終了少し前にCC64=0

### ✅ ストリングスCC11カーブ（実装済み）

各ノートに簡易ADSRカーブを描き、表情豊かな弓使いを演出します。

```yaml
strings:
  expr_cc: 11              # Expression CC番号
  expr_base: 84            # ベース値
  expr_depth: 25           # ピークまでの振れ幅
  expr_curve:
    attack_ms: 100         # 立ち上がり時間
    decay_ms: 150          # 減衰時間
    sustain_level: 92      # 維持レベル
    release_ms: 150        # リリース時間
```

**動作**:
- ノート長 > 0.75拍 の場合に適用
- 時系列: ベース→ピーク（attack）→維持（decay）→ベース（release）
- CC11イベントを4点折れ線で自動挿入

### フラム（将来拡張）

スネアの二打ち効果（現在はデフォルト無効）。

```yaml
drums:
  flam_ms: 18  # >0でスネアに自動フラム（-18ms前打ち）
```

## 実装詳細

### ギターストラムのアルゴリズム

1. **グループ化**: 同一タイミング（`strum_group_epsilon_ms`以内）のノートを検出
2. **方向判定**:
   - `strum_pattern: "beats"`: 拍位置で判定（1/3拍目=Down、2/4拍目=Up）
   - `strum_pattern: "alternate"`: 小節の偶奇で交互（偶数小節=Down）
3. **オフセット適用**:
   - Downストローク: 低音→高音へ順次遅延
   - Upストローク: 高音→低音へ順次遅延
   - ステップ幅 = `strum_width_ms / (ノート数 - 1)`

### ピアノペダルの挿入タイミング

- **On**: `start_beats - pedal_gap_beats * 0.2`（打鍵の少し前）
- **Off**: `end_beats - pedal_gap_beats`（離鍵の少し前）
- → 濁りを防ぎつつ、自然な余韻を確保

### ストリングスADSRカーブ

CC11の時系列変化（4点折れ線）:

```
t0: expr_base (ベース値)
t1: expr_base + expr_depth (ピーク) ← attack_ms後
t2: sustain_level (維持レベル) ← t1 + decay_ms後
t3: expr_base (ベースに戻る) ← end_beats - release_ms
```

## 安全性確認

### KPIへの影響評価

| 機能 | 影響範囲 | KPI安全性 |
|------|---------|----------|
| ギターストラム | タイミング±28ms以内 | ✅ ノート数不変、密度不変 |
| ピアノペダル | CC64追加（2イベント/ノート） | ✅ ノート数不変、CC追加のみ |
| ストリングスCC11 | CC11追加（4イベント/ノート） | ✅ ノート数不変、表現力向上 |

### 無効化方法

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

### ピアノペダル自動挿入

```yaml
piano:
  pedal_prob: 0.5         # 50%の確率でCC64挿入
  pedal_min_beats: 1.0    # 1拍以上の長さで
```

### ストリングスCC11カーブ

```yaml
strings:
  expr_depth: 25          # CC11の振れ幅
  attack_ms: 100          # 立ち上がり
  release_ms: 150         # 減衰
```

## 検証方法

### A/Bテスト推奨手順

1. **ベースライン生成**（ヒューマナイズOFF）
   ```bash
   # 全パラメータを0に設定した plan_humanize_baseline.yaml で生成
   python3 scripts/midi_writer.py --plan ... --config configs/plan_humanize_baseline.yaml --out baseline.mid
   ```

2. **ヒューマナイズ版生成**（標準設定）
   ```bash
   python3 scripts/midi_writer.py --plan ... --config configs/plan_humanize.yaml --out humanized.mid
   ```

3. **聴き比べ**
   - タイミングの自然さ
   - 強弱の表現力
   - セクション間の抑揚

4. **KPI確認**
   ```bash
   python3 scripts/kpi_gate.py --mid baseline.mid --bars bars.parquet --report baseline_kpi.json
   python3 scripts/kpi_gate.py --mid humanized.mid --bars bars.parquet --report humanized_kpi.json
   ```

## トラブルシューティング

### Q: セクション別ニュアンスが効かない

**A**: `--bars` オプションで `bars.parquet` を指定していることを確認。`bars.parquet` に `section_label` カラムが必須。

### Q: タイミングの揺れが大きすぎる/小さすぎる

**A**: `configs/plan_humanize.yaml` の `timing_ms` を±1〜2ms刻みで調整。安全域は±10ms以内。

### Q: Chorusの音量が小さい/大きい

**A**: `sections.chorus.vel_shift` を調整（-12〜+12の範囲推奨）。

### Q: ドラムの前ノリ/後ノリが強すぎる

**A**: `hh_microshift_ms` / `snare_layback_ms` を±2ms刻みで調整。

## 参考資料

- [人間味の5本柱 詳細解説](./HUMANIZE_DETAILS.md)（別途作成予定）
- [KPI Gate仕様](./KPI_GATE_SPEC.md)
- [MIDI Writer実装詳細](./研究手法_⇔_実装_対応表（v_2025_11_01_）.md)

## 更新履歴

- **2025-11-01 v1.1**: 上級機能実装
  - ✅ ギターストラム自動展開（beats/alternate方向判定）
  - ✅ ピアノペダル確率挿入（CC64、濁り防止gap付き）
  - ✅ ストリングスCC11 ADSRカーブ自動生成
  - イベント統合処理（Note + CC混在対応）
  
- **2025-11-01 v1.0**: 役割/セクション別ヒューマナイズ初版実装
  - ドラムHH前ノリ/スネア後ノリ
  - ベースノート長変動
  - セクション別timing_scale/vel_shift
  - KPI安全域確認（±10ms/±10Vel）
