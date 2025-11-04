# 人間味機能実装ガイド（レビュー改善版）

## 📋 実装完了状況

### ✅ 完了項目

#### 1. YAML設定ファイル（plan_humanize.yaml v2）
- **Phase 4機能**: bars.parquet駆動の11レイヤ（rubato, beat_nudge, velocity_energy, accent_response, swing, groove_template, anticipation, length_shaper, velocity_shaper, dynamic_arcs, pink_jitter）
- **レビュー提案1**: `roles.drums.section_bias` - セクション別マイクロオフセット（hh_microshift_ms, snare_layback_ms, kick_anticipation_ms）
- **レビュー提案2**: `roles.guitar.section_direction_bias` - セクション別ストラム方向バイアス
- **レビュー提案4**: `performance.swing.reduce_if_ghost_hh` - ゴーストHH密度連動の自動調整
- **レビュー提案5**: `reproducibility` - humanizeセクションハッシュ焼き込み設定

**ファイル**: `configs/plan_humanize.yaml`  
**バージョン**: v2  
**行数**: 約350行

---

#### 2. CI検証スクリプト（レビュー提案3）

##### `scripts/validate_humanize_safety.py`
**目的**: 人間味適用後の安全性を3段階で検証

```bash
# 使用例
python3 scripts/validate_humanize_safety.py \
  --baseline baseline.mid \
  --humanized output.mid \
  --plan arrangement_plan.json
```

**検証項目**:
1. **無効化時ビット完全一致** (humanize.enabled=false時)
   - ±0 ticks / ±0 velocity / ±0 length
2. **境界制約チェック**
   - max|Δtime| ≤ 30ms
   - max|Δvel| ≤ 20
   - max|Δlen| ≤ 40ms
3. **KPI非劣化チェック**
   - Pass率悪化 ≤ 0.3%
   - notes_per_bar / backbeat_strength の安全域維持

**実装**: ✅ 完了  
**CI統合**: ⏳ Makefile / GitHub Actionsへの追加が必要

---

##### `scripts/stamp_humanize_tag.py`（レビュー提案5）
**目的**: YAML humanizeセクションのハッシュをMIDIメタデータに焼き込み

```bash
# 使用例
python3 scripts/stamp_humanize_tag.py \
  --config configs/plan_humanize.yaml \
  --midi output.mid \
  --version v2

# 出力例: "humanize_v2_abc12345"
```

**機能**:
- bar_features, performance, roles セクションをJSON化 → SHA256 → 8文字短縮
- MIDIメタイベント（text）に埋め込み
- トラック名の末尾に追記（オプション）

**実装**: ✅ 完了  
**midi_writer.py統合**: ⏳ --stamp-humanize-tag オプション追加が必要

---

### ⏳ 実装待ち項目

#### 3. midi_writer.py への機能統合

##### A. ドラムのセクション別バイアス（レビュー提案1）
**YAML設定**:
```yaml
roles:
  drums:
    section_bias:
      enable: true
      chorus:
        hh_microshift_ms: -2
        snare_layback_ms: 3
        kick_anticipation_ms: -1
```

**実装箇所**: `write_track_from_abs_notes()` 内のドラムイベント処理
**実装内容**:
```python
# セクション判定
current_section = get_section_at_beat(beat, sections)

# section_bias適用
section_bias = cfg.get("roles", {}).get("drums", {}).get("section_bias", {})
if section_bias.get("enable", False) and current_section in section_bias:
    bias = section_bias[current_section]
    
    # GM Drum Map: 42=Closed HH, 38=Snare, 36=Kick
    if pitch == 42:  # HH
        time_ms += bias.get("hh_microshift_ms", 0)
    elif pitch == 38:  # Snare
        time_ms += bias.get("snare_layback_ms", 0)
    elif pitch == 36:  # Kick
        time_ms += bias.get("kick_anticipation_ms", 0)
```

**状態**: ⏳ 未実装

---

##### B. ギターのセクション別ストラム方向バイアス（レビュー提案2）
**YAML設定**:
```yaml
roles:
  guitar:
    section_direction_bias:
      enable: true
      chorus: "down"
      verse: "up"
      bridge: null  # auto
```

**実装箇所**: ストラム方向決定ロジック（`write_track_from_abs_notes()` 内）
**実装内容**:
```python
# 既存の自動判定（転回/拍位置ベース）
auto_direction = determine_strum_direction(chord, beat, top_note)

# セクションバイアス適用
section_bias = cfg.get("roles", {}).get("guitar", {}).get("section_direction_bias", {})
if section_bias.get("enable", False):
    current_section = get_section_at_beat(beat, sections)
    biased_dir = section_bias.get(current_section, None)
    
    if biased_dir:
        direction = biased_dir  # "down" / "up" を優先
    else:
        direction = auto_direction  # null なら自動判定
else:
    direction = auto_direction
```

**状態**: ⏳ 未実装

---

##### C. スウィング × ゴーストHH 相互作用調整（レビュー提案4）
**YAML設定**:
```yaml
performance:
  swing:
    enable: true
    max_ms: 28
    reduce_if_ghost_hh: true
    ghost_hh_threshold: 0.60
```

**実装箇所**: `write_plan()` 内のswing適用時
**実装内容**:
```python
# bars.parquet から density_target を取得
ghost_hh_density = bars_df.iloc[current_bar]["density_target"] if bars_df is not None else 0.5

# ゴーストHH密度が高い場合にmax_msを減算
swing_config = cfg.get("performance", {}).get("swing", {})
max_ms = swing_config.get("max_ms", 28)

if swing_config.get("reduce_if_ghost_hh", False):
    threshold = swing_config.get("ghost_hh_threshold", 0.60)
    if ghost_hh_density > threshold:
        reduction = 3  # 固定値（将来的にはパラメータ化）
        max_ms = max(max_ms - reduction, 10)  # 下限10ms
```

**状態**: ⏳ 未実装

---

##### D. humanizeタグ自動焼き込み（レビュー提案5統合）
**実装箇所**: `write_plan()` の最後（MIDIファイル保存後）
**実装内容**:
```python
# reproducibility設定確認
repro_cfg = cfg.get("reproducibility", {})
if repro_cfg.get("enable", True) and repro_cfg.get("embed_in_midi_meta", True):
    from stamp_humanize_tag import generate_humanize_tag, embed_tag_in_midi_meta
    
    tag = generate_humanize_tag(config_path, version="v2")
    embed_tag_in_midi_meta(out_mid, tag, track_name_suffix=True)
```

**状態**: ⏳ 未実装

---

## 📊 実装優先度マトリクス

| 機能 | 優先度 | 理由 | 実装難易度 |
|-----|-------|-----|----------|
| **A. ドラムsection_bias** | ⭐⭐⭐ | 耳上の質感向上（chorus/verse差別化） | 低 |
| **B. ギターdirection_bias** | ⭐⭐ | 一貫性向上（chorusはdown統一等） | 低 |
| **C. swing×ghost_hh調整** | ⭐⭐ | 過剰な「跳ね×刻み」防止 | 中 |
| **D. humanizeタグ焼き込み** | ⭐ | 運用の頑健さ（差分追跡容易） | 低 |

---

## 🚀 実装手順（推奨）

### ステップ1: ドラムsection_bias実装（30分）
1. `write_track_from_abs_notes()` でドラムトラック判定時にセクション取得
2. GM Drum Map (36=Kick, 38=Snare, 42=HH) に応じてバイアス適用
3. テスト: chorus/verse間でHH/Snareのタイミング差を確認

### ステップ2: ギターdirection_bias実装（20分）
1. ストラム方向決定ロジックにセクション判定を追加
2. bias設定がnullなら既存の自動判定にフォールバック
3. テスト: chorusで全てdown、verseで全てupになることを確認

### ステップ3: swing×ghost_hh調整実装（40分）
1. bars.parquet から density_target を取得（既存の_load_bars_features()を利用）
2. threshold超過時にmax_msを減算（3ms固定）
3. テスト: 高密度HHバーでswing量が自動抑制されることを確認

### ステップ4: humanizeタグ焼き込み（15分）
1. `write_plan()` 最後に stamp_humanize_tag.py の関数を呼び出し
2. reproducibility.enableがfalseなら何もしない
3. テスト: 出力MIDIのメタイベント/トラック名にタグが含まれることを確認

### ステップ5: CI統合（30分）
1. `Makefile` に `test-humanize-safety` ターゲット追加
2. `.github/workflows/ci.yml` に検証ステップ追加
3. 例:
   ```yaml
   - name: Validate Humanize Safety
     run: |
       python3 scripts/midi_writer.py --plan test_data/test.json --out baseline.mid --config configs/plan_humanize_baseline.yaml
       python3 scripts/midi_writer.py --plan test_data/test.json --out humanized.mid --config configs/plan_humanize.yaml
       python3 scripts/validate_humanize_safety.py --baseline baseline.mid --humanized humanized.mid
   ```

---

## 🧪 テスト戦略

### 単体テスト
- `test_section_bias.py`: セクション別バイアスの正確性
- `test_direction_bias.py`: ストラム方向決定の一貫性
- `test_swing_ghost_hh.py`: density連動の自動調整

### 統合テスト
- `test_humanize_e2e.py`: full_arrangement.json → MIDI → 検証
- A/B比較: baseline.mid vs humanized.mid

### KPI回帰テスト
- `test_kpi_no_degradation.py`: humanize前後でKPI Gateが通過
- 許容範囲: ±0.3%

---

## 📝 使用例（全機能有効化）

### 最小構成（安全）
```bash
python3 scripts/midi_writer.py \
  --plan song_001/full_arrangement.json \
  --config configs/plan_humanize.yaml \
  --out output_safe.mid
```

**plan_humanize.yaml**:
```yaml
performance:
  accent_response: { enable: true }
  swing: { enable: true }
roles:
  drums:
    section_bias: { enable: false }  # OFF
```

### 中程度構成（推奨）
```yaml
performance:
  rubato: { enable: true }
  beat_nudge: { enable: true }
  velocity_energy: { enable: true }
  accent_response: { enable: true }
  swing: { enable: true, reduce_if_ghost_hh: true }
roles:
  drums:
    section_bias: { enable: true }
  guitar:
    section_direction_bias: { enable: true }
reproducibility:
  enable: true
```

### 最大構成（攻め）
```yaml
performance:
  # 全レイヤON
  rubato: { enable: true }
  beat_nudge: { enable: true }
  velocity_energy: { enable: true }
  accent_response: { enable: true }
  swing: { enable: true, reduce_if_ghost_hh: true }
  groove_template: { enable: true }
  anticipation: { enable: true }
  length_shaper: { enable: true }
  velocity_shaper: { enable: true, mode: expand, ratio: 1.18 }
  dynamic_arcs: { enable: true }
  pink_jitter: { enable: true, ms_std: 5 }
roles:
  drums:
    section_bias: { enable: true }
  guitar:
    section_direction_bias: { enable: true }
reproducibility:
  enable: true
```

---

## 🔗 関連ドキュメント

- [bars.parquet駆動ガイド](./HUMANIZE_BARS_DRIVEN.md)
- [自動判定機能ガイド](./HUMANIZE_AUTO_DETECTION.md)
- [上級ヒューマナイズ機能ガイド](./HUMANIZE_ADVANCED_FEATURES.md)
- [KPI Gate仕様](./KPI_GATE_SPEC.md)

---

## 📅 次のアクション

### 即座に実装可能（1-2時間）
1. ドラムsection_bias → **最優先**（耳上の効果大）
2. ギターdirection_bias → 一貫性向上
3. humanizeタグ焼き込み → 運用改善

### 追加検討が必要（2-4時間）
1. swing×ghost_hh調整 → density取得ロジックの確認必要
2. CI統合 → テストデータ準備

### オプション（将来的に）
1. Phase 4パフォーマンスレイヤの実装（rubato, beat_nudge等）
2. bars.parquet複数列対応の完全テスト
3. A/B試聴テストによる最適パラメータ探索

---

**最終更新**: 2025年11月1日  
**実装ステータス**: YAML完成 / スクリプト完成 / midi_writer.py統合待ち
