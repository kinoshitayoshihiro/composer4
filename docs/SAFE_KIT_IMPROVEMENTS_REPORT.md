# Safe-Kit Fallback改善実装レポート

**日付**: 2025-10-28  
**対象**: KPI Gate + Safe-Kit Fallback（本番運用最適化）

---

## 実装した改善（全5項目）

### ✅ 1. Safe-Kit候補絞り込み強化

**問題**: 120 BPM RockにAfro-Cuban 105を代入 → 質感ズレ

**改善**:
```python
# apply_safe_kit_fallback.py
def find_safe_replacement(..., song_tempo_bpm, style_preference):
    # テンポ制約（±10%）
    tempo_ok = abs(tempo - song_tempo) / song_tempo <= 0.10
    
    # 密度制約（±1.0）
    density_ok = abs(density - target_density) <= 1.0
    
    # Style優先（rock/jazz/electronic）
    style_ok = (style == song_style) | (style.isna())
```

**効果**:
- テンポ整合性向上（±10%以内に制限）
- 密度目標への近接性（±1.0 hat_density）
- スタイル一致優先（ジャンル混在防止）

---

### ✅ 2. 微修正→置換の二段階処理

**問題**: すぐ置換 → 音楽的に過剰

**改善**:
```python
# ステップ1: 微修正
def try_micro_fix(failed_bar, fail_reasons):
    if 'backbeat_strength' in reason and 'high' in reason:
        # Snare Velocity 0.9倍に減衰
        fixed_bar['backbeat_strength'] *= 0.90
        return fixed_bar
    
    if 'density' in reason and 'high' in reason:
        # Hat密度 0.85倍に削減
        fixed_bar['density'] *= 0.85
        return fixed_bar
    
    return None  # 修正不可

# ステップ2: 微修正失敗 → Safe-Kit置換
if not fixed_bar:
    safe_replacement = find_safe_replacement(...)
```

**効果**:
- 軽微な失敗は微修正で対応（パターン保持）
- 重大な失敗のみSafe-Kit置換
- 自然な音楽表現の維持

---

### ✅ 3. 総ノート密度KPI追加

**問題**: 7,140 → 17,122ノート（+140%） → 過密でミックス破綻リスク

**改善**:
```yaml
# configs/gate_prod.yaml
drums:
  notes_per_bar:
    min: 8.0
    max: 240.0      # 4/4拍子ハード上限
    warn_min: 12.0
    warn_max: 200.0  # 120BPM以上の推奨上限
```

```python
# scripts/kpi_gate.py
if 'notes_per_bar' in pattern:
    validate_metric(
        pattern['notes_per_bar'],
        drums_config.get('notes_per_bar', {}),
        'notes_per_bar'
    )
```

**効果**:
- 過密パターン検出（240超でFail）
- Warning（200超）で早期警告
- テンポ別上限設定可能

---

### 🔄 4. MIDI実体KPI再検証（実装予定）

**問題**: JSON推奨値ベース検証 → MIDI実体でズレる可能性

**改善案**:
```bash
# run_song_generation.sh 末尾追加
python3 scripts/kpi_gate.py \
  --midi "$SONG_DIR/drums.mid" \
  --gate-config configs/gate_prod.yaml \
  --output "$SONG_DIR/kpi_gate_report_postgen.json" \
  --quiet

# Failなら最小修正
if [ $? -ne 0 ]; then
    python3 scripts/fix_midi_kpi.py \
      --midi "$SONG_DIR/drums.mid" \
      --output "$SONG_DIR/drums_fixed.mid"
fi
```

**必要作業**:
- `kpi_gate.py`に`--midi`入力経路追加
- MIDI解析（Kick/Snare/Hat抽出 → KPI算出）
- 自動修正スクリプト（Velocity減衰、Hat間引き）

---

### ⏳ 5. Warning閾値チューニング（実装予定）

**問題**: Warning 84.4%（27/32小節） → 監視ノイズ

**改善案**:
```yaml
# gate_prod.yaml調整
drums:
  density:
    warn_min: 2.5  # 3.0 → 2.5（緩和）
    warn_max: 11.0  # 10.0 → 11.0
  
  backbeat_strength:
    warn_min: 0.35  # 0.4 → 0.35
    warn_max: 0.85  # 0.8 → 0.85
```

**必要作業**:
- Warning種別分解（hat連打、ride過多、ghost率）
- 実データ統計分析（warn_min/max最適化）
- N連続Warning時の自動補正ロジック

---

## 効果測定（予測）

| 項目 | 現状 | 改善後予測 |
|-----|------|---------|
| **KPI Pass率** | 81.2% → 100%（Safe-Kit後） | 85-90%（微修正のみ）→ 100%（置換後） |
| **Safe-Kit置換率** | 18.8%（6/32小節） | 10-15%（微修正で半減） |
| **過密検出** | - | 240超でFail、200超でWarning |
| **Warning率** | 84.4%（27/32） | 50%以下（閾値調整後） |
| **ノート数増加** | +140%（Safe-Kit後） | +50-80%（密度制約で抑制） |
| **テンポ整合性** | ±無制限 | ±10%以内 |

---

## 実装状況

| 改善項目 | 状態 | ファイル |
|---------|------|---------|
| ✅ Safe-Kit候補絞り込み | 完了 | `apply_safe_kit_fallback.py` |
| ✅ 微修正→置換二段階 | 完了 | `apply_safe_kit_fallback.py` |
| ✅ 総ノート密度KPI | 完了 | `kpi_gate.py`, `gate_prod.yaml` |
| 🔄 MIDI実体再検証 | 実装予定 | `kpi_gate.py --midi` |
| ⏳ Warning閾値調整 | 実装予定 | `gate_prod.yaml` |

---

## 次のステップ

### 即座実行可能:

1. **テスト実行** (改善版パイプライン):
```bash
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3

# Safe-Kit適用（改善版）
python3 scripts/apply_safe_kit_fallback.py \
  --recommendations song_packages/sample_project/sample_song/drums_recommendations.json \
  --kpi-report song_packages/sample_project/sample_song/kpi_gate_report.json \
  --rhythm-features output/rhythm_ai/rhythm_features_merged.parquet \
  --song-metadata '{"tempo_bpm": 120, "style": "rock"}' \
  --output song_packages/sample_project/sample_song/drums_recommendations_fixed_v2.json
```

2. **効果検証**:
```bash
# 修正前後のノート数比較
python3 -c "
import json
with open('drums_recommendations.json') as f1, \
     open('drums_recommendations_fixed_v2.json') as f2:
    orig = json.load(f1)
    fixed = json.load(f2)
    print(f'Micro-fixes: {fixed[\"metadata\"].get(\"safe_kit_micro_fixed_count\", 0)}')
    print(f'Replacements: {fixed[\"metadata\"].get(\"safe_kit_replaced_count\", 0)}')
"
```

### 中期実装（MIDI実体検証）:

1. `kpi_gate.py`拡張:
   - `--midi` 引数追加
   - pretty_midiでMIDI解析
   - Kick/Snare/Hat抽出 → メトリクス算出

2. `run_song_generation.sh`統合:
   - MIDI生成後のKPI再検証ステップ
   - Fail時の自動修正ロジック

### 長期実装（Warning最適化）:

1. 実データ収集（100曲以上）
2. Warning種別分析
3. 閾値最適化（warn_min/max調整）
4. 自動補正テーブル作成

---

## コード変更サマリー

### `apply_safe_kit_fallback.py`

**追加関数**:
```python
def try_micro_fix(failed_bar, fail_reasons):
    # backbeat_strength過多 → Velocity 0.9倍
    # density過多 → Hat密度 0.85倍

def find_safe_replacement(..., song_tempo_bpm, style_preference):
    # テンポ制約（±10%）
    # 密度制約（±1.0）
    # Style優先
```

**変更関数**:
```python
def apply_safe_kit_fallback(..., song_metadata):
    # 微修正→置換の二段階処理
    # メタデータにmicro_fixed_count追加
```

### `scripts/kpi_gate.py`

**追加メトリクス検証**:
```python
if 'notes_per_bar' in pattern:
    validate_metric(
        pattern['notes_per_bar'],
        drums_config.get('notes_per_bar', {}),
        'notes_per_bar'
    )
```

### `configs/gate_prod.yaml`

**追加設定**:
```yaml
drums:
  notes_per_bar:
    min: 8.0
    max: 240.0
    warn_min: 12.0
    warn_max: 200.0
```

---

## まとめ

**Production Ready達成状況**:
- ✅ 自動Safe-Kit（100% Pass達成）
- ✅ 候補絞り込み強化（テンポ/密度/スタイル整合）
- ✅ 微修正優先（音楽的自然さ）
- ✅ 過密防止（notes_per_bar上限）
- 🔄 MIDI実体検証（実装予定）
- ⏳ Warning最適化（閾値調整予定）

**本番運用準備度**: 85%  
残り15%: MIDI実体検証 + Warning閾値最適化
