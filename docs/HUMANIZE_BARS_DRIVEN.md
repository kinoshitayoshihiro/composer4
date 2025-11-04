# bars.parquet駆動の深層人間味ガイド（v1.3）

## 概要

**bars.parquet / bars_extended.parquet**から直接**エナジー・セクション・アクセント・スウィング・密度・ドラム活性**を読み取り、音楽的文脈に応じた高度な人間味表現を実現します。

すべての機能は**既定OFF**なので、適用直後は既存挙動と完全一致します。

---

## 新機能一覧

### 📊 bars.parquet統合

| 列名候補 | 用途 | デフォルト動作 |
|---------|------|---------------|
| `energy_curve` / `energy` | バー別エネルギー（0..1） | イベント密度から推定 |
| `section_label` / `section` | セクション名（intro/verse/chorus/bridge） | セクション判定なし |
| `accent_score_target` / `accent_score` | アクセント強度（0..1） | 未使用 |
| `swing_pct` / `swing` | スウィング量（0..1） | セクション設定値 |
| `density_target` / `density` | 密度（参考値） | 未使用 |
| `drums_active` | ドラム活性（0/1） | sparse_bar_helpers用 |

#### 設定例（configs/plan_humanize.yaml）

```yaml
bar_features:
  use: true                 # false=従来のイベント密度推定にフォールバック
  prefer_extended: true     # bars_extended.parquet を優先
  path: null                # null=自動（song_dir/bars.parquet）
  columns:
    energy: ["energy_curve", "energy"]
    section: ["section_label", "section"]
    accent: ["accent_score_target", "accent_score"]
    swing: ["swing_pct", "swing"]
    density: ["density_target", "density"]
    drums_active: ["drums_active"]
```

---

## パフォーマンスレイヤ機能

### 1. ルバート（rubato）

**目的**: セクション別に小節内で軽く前後揺れ（呼吸感）

```yaml
performance:
  rubato:
    enable: false           # 既定OFF
    shape: sine             # sine | triangle
    depth_ms_by_section:
      intro: 6
      verse: 5
      chorus: 3             # タイトに
      bridge: 8             # 自由に
      outro: 4
```

**KPI影響**: ±数ms。ノート数不変。

---

### 2. 拍位置ナッジ（beat_nudge）

**目的**: ダウンビート=前ノリ、バックビート=後ノリで躍動感

```yaml
performance:
  beat_nudge:
    enable: false
    downbeat_ms: -6         # 1拍目を少し前へ
    backbeat_ms: 10         # 2/4拍を少し後ろへ
    scale_by_energy: true   # エナジー高で効果増幅
    max_abs_ms: 14
```

**KPI影響**: ±10ms程度。バックビート感増。

---

### 3. ベロシティ・エナジー連動（velocity_energy）

**目的**: エナジー高いバーで強く、低いバーで控えめ

```yaml
performance:
  velocity_energy:
    enable: false
    slope: 18               # vel += slope * (energy - 0.5)
    min_vel: 1
    max_vel: 127
```

**KPI影響**: ±10 vel程度。密度不変。

---

### 4. アクセント応答（accent_response）

**目的**: bars列の`accent_score`に応じて強弱・タイミング微調整

```yaml
performance:
  accent_response:
    enable: false
    center: 0.50
    vel_slope: 22           # vel += slope * (accent-0.5)
    time_ms_at_full: -5     # accent=1.0で-5ms（前ノリ）
    clamp_vel: [1, 127]
    max_abs_ms: 9
    apply_roles: ["guitar", "piano", "strings"]
```

**効果**: アクセント箇所で自然な強調。

---

### 5. スウィング（swing）

**目的**: 8分裏を遅らせて三連風グルーヴ

```yaml
performance:
  swing:
    enable: false
    source: "bars_then_section"   # bars_then_section | section_only | bars_only
    amount_by_section:            # 0.00..0.40（0.33≈三連）
      verse: 0.12
      chorus: 0.08
      bridge: 0.15
    max_ms: 28
    subdivision: 8                # 8 | 16
    apply_roles: ["guitar", "piano", "strings"]
```

**KPI影響**: 裏拍のみ遅れ。ノート数不変。

---

## 攻めの人間味レイヤ（enable: false推奨）

### 6. グルーヴテンプレート（groove_template）

**目的**: 拍ごとの前ノリ/後ノリと強弱倍率でセクション別グルーヴ

```yaml
performance:
  groove_template:
    enable: false
    source: "section"       # section | global
    templates:
      verse:
        beat_offsets_ms: { "1": -4, "2": 8, "3": -2, "4": 10 }
        velocity_scale:  { "1": 1.06, "2": 1.04, "3": 0.97, "4": 1.05 }
      chorus:
        beat_offsets_ms: { "1": 0, "2": 6, "3": 0, "4": 8 }
        velocity_scale:  { "1": 1.03, "2": 1.02, "3": 0.99, "4": 1.03 }
      bridge:
        beat_offsets_ms: { "1": -6, "2": 10, "3": -3, "4": 12 }
        velocity_scale:  { "1": 1.08, "2": 1.05, "3": 0.95, "4": 1.07 }
    clamp_ms: 15
    vel_minmax: [1, 127]
    apply_roles: ["drums","bass","guitar","piano","strings"]
```

**KPI影響**: ±15ms、vel倍率1.0±0.10。密度不変。

---

### 7. アンティシペーション（anticipation）

**目的**: ベース/ギターがコードチェンジを先取り（18ms前倒し）

```yaml
performance:
  anticipation:
    enable: false
    roles: ["bass","guitar"]
    ms_before_chord_change: 18
    beat_window: 0.55
    only_high_energy: true
    energy_gate: 0.60
```

**効果**: リズムセクションの前ノリ感。

---

### 8. ノート長シェイパー（length_shaper）

**目的**: ダウンビート=レガート、裏=スタッカート

```yaml
performance:
  length_shaper:
    enable: false
    legato_on_downbeat: 1.12
    staccato_on_offbeat: 0.80
    clamp_beats: [0.05, 3.50]
    apply_roles: ["guitar","piano","strings","bass"]
```

**KPI影響**: ノート長変更のみ。数不変。

---

### 9. ベロシティシェイパー（velocity_shaper）

**目的**: ダイナミクスの圧縮/拡張

```yaml
performance:
  velocity_shaper:
    enable: false
    mode: "expand"          # expand | compress
    mid: 88
    ratio: 1.25             # >1で拡張、<1で圧縮
    clamp: [1, 127]
    apply_roles: ["drums","bass","guitar","piano","strings"]
```

**効果**: `expand`=メカニカル感低減、`compress`=均一化。

---

### 10. ダイナミック・アーク（dynamic_arcs）

**目的**: セクション内で徐々に盛る/落とす

```yaml
performance:
  dynamic_arcs:
    enable: false
    by_section:
      verse:  { type: "crescendo", depth: 10 }
      chorus: { type: "plateau",   depth: 4 }
      bridge: { type: "decrescendo", depth: 8 }
    apply_roles: ["guitar","piano","strings","bass"]
```

**効果**: 小節進行に応じてvel±10程度。

---

### 11. ピンクジッター（pink_jitter）

**目的**: AR(1)モデルで持続的な揺れ（ピンクノイズ的）

```yaml
performance:
  pink_jitter:
    enable: false
    ms_std: 6               # 標準偏差（ms）
    correlation: 0.85       # 連続相関（AR(1)のρ）
    seed: null
    apply_roles: ["drums","bass","guitar","piano","strings"]
```

**KPI影響**: ±6ms程度の連続的揺れ。ノート数不変。

---

### 12. ギター sparse_bar_helpers

**目的**: ドラム休符バーでストラム幅を少し広げる

```yaml
roles:
  guitar:
    sparse_bar_helpers:
      enable: false
      widen_if_drums_inactive: true
      width_ms_bonus: 8
      energy_gate: 0.40
```

---

## 推奨設定プリセット

### 🛡️ 安全（KPI最優先）

```yaml
performance:
  accent_response: { enable: true }
  swing:
    enable: true
    source: bars_then_section
    subdivision: 8
    max_ms: 22
```

**効果**: bars列の`accent`/`swing`のみ適用。微細な変化。

---

### ⚡ 中程度（音楽性重視）

```yaml
performance:
  rubato: { enable: true }
  beat_nudge: { enable: true }
  velocity_energy: { enable: true }
  accent_response: { enable: true }
  swing: { enable: true }
  groove_template: { enable: true }
  anticipation: { enable: true }
```

**効果**: セクション・拍・エナジー駆動の総合人間味。

---

### 🔥 攻め（最大表現）

```yaml
performance:
  # 基本レイヤすべてON
  rubato: { enable: true }
  beat_nudge: { enable: true }
  velocity_energy: { enable: true }
  accent_response: { enable: true }
  swing: { enable: true }
  # 攻めレイヤ
  groove_template: { enable: true }
  anticipation: { enable: true }
  length_shaper: { enable: true }
  velocity_shaper: { enable: true, mode: expand, ratio: 1.18 }
  dynamic_arcs: { enable: true }
  pink_jitter: { enable: true, ms_std: 5, correlation: 0.88 }
```

**注意**: KPI影響大。まずは中程度から試してください。

---

## 実行例

```bash
# bars.parquet自動検出
python3 scripts/midi_writer.py \
  --plan song_packages/suno_project/song_001/full_arrangement.json \
  --config configs/plan_humanize.yaml \
  --out full_arrangement_human.mid

# bars.parquet明示指定
python3 scripts/midi_writer.py \
  --plan full_arrangement.json \
  --config configs/plan_humanize.yaml \
  --bars song_packages/suno_project/song_001/bars_extended.parquet \
  --out output.mid
```

---

## トラブルシューティング

### Q: bars.parquetが読み込まれない

**A**: 以下を確認：
1. `bar_features.use: true`
2. `bars.parquet`または`bars_extended.parquet`が存在
3. 列名が`columns`設定と一致
4. pandas (`_pd`)がインストール済み

### Q: 効果が感じられない

**A**: 確認ポイント：
- `enable: true`になっているか
- `apply_roles`に該当ロールが含まれているか
- bars列に実データがあるか（全部0だと効果なし）
- デフォルト値が小さい場合は`depth`, `slope`, `ms`を増やす

### Q: KPIへの影響が心配

**A**: 安全な順：
1. `accent_response` - ±9ms、±11vel
2. `swing` - 裏拍のみ+28ms以下
3. `beat_nudge` - ±14ms
4. `velocity_energy` - ±9vel
5. `rubato` - ±8ms
6. それ以外 - 影響中〜大

---

## A/Bテスト手順

### ステップ1: ベースライン

```yaml
# plan_humanize_baseline.yaml
bar_features: { use: false }  # bars.parquet無効
performance:
  # すべてenable: false
```

### ステップ2: bars駆動のみ

```yaml
bar_features: { use: true }
performance:
  accent_response: { enable: true }
  swing: { enable: true, source: bars_then_section }
```

### ステップ3: フル人間味

```yaml
# plan_humanize.yamlのデフォルト + 中程度プリセット
```

### 比較コマンド

```bash
python3 scripts/midi_writer.py --plan ... --config plan_humanize_baseline.yaml --out v0.mid
python3 scripts/midi_writer.py --plan ... --config plan_humanize_bars.yaml --out v1.mid
python3 scripts/midi_writer.py --plan ... --config plan_humanize.yaml --out v2.mid
```

---

## 実装済み内容（v1.3）

✅ bars.parquet複数列対応（energy/section/accent/swing/density/drums_active）  
✅ ルバート（セクション別深さ、sine/triangle）  
✅ 拍位置ナッジ（ダウン/バック、エナジー連動）  
✅ ベロシティ・エナジー連動  
✅ アクセント応答（bars列駆動）  
✅ スウィング（8/16分、bars→section優先）  
✅ グルーヴテンプレート（拍別オフセット・強弱倍率）  
✅ アンティシペーション（コードチェンジ先行）  
✅ ノート長シェイパー（レガート/スタッカート）  
✅ ベロシティシェイパー（expand/compress）  
✅ ダイナミック・アーク（crescendo/decrescendo/plateau）  
✅ ピンクジッター（AR(1)相関ノイズ）  
✅ sparse_bar_helpers（ドラム休符バー補正）  

**注**: すべて`enable: false`デフォルト。段階的に有効化推奨。

---

## 更新履歴

- **2025-11-01 v1.3**: bars.parquet駆動の深層人間味実装
  - bars.parquet複数列対応
  - パフォーマンスレイヤ5機能（rubato, beat_nudge, velocity_energy, accent_response, swing）
  - 攻めレイヤ7機能（groove_template, anticipation, length_shaper, velocity_shaper, dynamic_arcs, pink_jitter, sparse_bar_helpers）
  - 後方互換性維持（すべてenable: false既定）

- **2025-11-01 v1.2**: 自動判定機能実装
- **2025-11-01 v1.1**: 上級機能実装

---

## 参考資料

- [自動判定機能ガイド](./HUMANIZE_AUTO_DETECTION.md)
- [上級ヒューマナイズ機能ガイド](./HUMANIZE_ADVANCED_FEATURES.md)
- [役割/セクション別ヒューマナイズガイド](./HUMANIZE_ROLE_SECTION.md)
- [KPI Gate仕様](./KPI_GATE_SPEC.md)
