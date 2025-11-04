# Phase 26/31 Mode/Scale統合 完了レポート

## ✅ 統合完了日
2025年10月21日

## 📋 実装サマリー

### 1. **InstrumentStage2Base 拡張**

#### 追加メソッド

**`_apply_mode_scale_constraint(part, section_meta)`** (line ~1500)
- **目的**: Phase 31でスケール外音を最近接スケール内音に修正
- **動作**: 
  - sections から key_hint/mode_hint を取得
  - 各ノートのピッチクラス(PC)をチェック
  - 平均マスク値の70%以下なら「スケール外」と判定
  - ±1半音、±2半音の順で最近接スケール内音を探索
  - 見つかれば自動修正
- **NO-OP保証**: mode_hint無し → 完全スキップ

**`_voice_leading_smooth(part, section_meta, chord_now, chord_prev, cfg)`** (line 1250)
- **拡張箇所**: 末尾に `self._apply_mode_scale_constraint(part, section_meta)` を追加
- **効果**: 既存のvoice-leading処理の後にmode/scale制約を適用

### 2. **対応楽器**

| 楽器 | Phase 31実装 | Mode/Scale統合 | 備考 |
|------|------------|---------------|------|
| Piano | ✅ line 606 | ✅ 自動適用 | `_voice_leading_smooth` 経由 |
| Guitar | ✅ line 587 | ✅ 自動適用 | `_voice_leading_smooth` 経由 |
| Strings | ✅ line 565 | ✅ 自動適用 | `_voice_leading_smooth` 経由 |
| Bass | ❌ 未実装 | ⏳ 将来対応 | Phase 31自体が無い |

### 3. **動作フロー**

```
各楽器の _phase_31()
    ↓
InstrumentStage2Base._voice_leading_smooth()
    ↓
    1. 既存処理: 和声音優先 + 跳躍制限
    ↓
    2. 新規追加: self._apply_mode_scale_constraint()
        ↓
        - sections から key_hint/mode_hint 取得
        - 各ノートのピッチクラスをチェック
        - スケール外音を検出
        - 最近接スケール内音に修正
```

## 🎯 実装詳細

### スケール外音判定ロジック

```python
# 平均マスク値の70%以下ならスケール外
avg_mask = sum(mask) / len(mask)
threshold = avg_mask * 0.70

if mask[pc] <= threshold:
    # スケール外音として修正対象
```

### 修正候補探索

```python
# ±1半音、±2半音の順で探索
candidates = []
for offset in [1, -1, 2, -2]:
    new_pc = (pc + offset) % 12
    if mask[new_pc] > threshold:
        candidates.append((abs(offset), pitch + offset))

# 最も近い音を選択
candidates.sort()
new_pitch = candidates[0][1]
```

## 🧪 テスト結果

### Test 1: D Ionian - スケール外音検出
```
Mask: [0.02, 0.14, 0.15, 0.02, 0.13, 0.02, 0.14, 0.09, 0.02, 0.14, 0.02, 0.11]
D#4/Eb4 (PC=3): スケール外 (mask=0.02) ✓
F4 (PC=5): スケール外 (mask=0.02) ✓
```

### Test 2: G Mixolydian - 特徴度数確認
```
F5 (PC=5): mask=0.14 - ♭7（特徴度数） ✓
F#5 (PC=6): mask=0.02 - スケール外 ✓
```

### Test 3: NO-OP確認
```
key_hint のみ → Ionian/Aeolian自動推定 ✓
key_hint/mode_hint 両方無し → None返却（完全NO-OP） ✓
```

### Test 4: 転調テスト
```
D Ionian: F音(PC=5) mask=0.02 (スケール外)
G Mixolydian: F音(PC=5) mask=0.14 (♭7として強調)
転調時のF音の扱いが適切に変化 ✓
```

## 📊 Phase 26 統合状況

### 現在の実装

**Phase 26 (Hybrid Harmony)** は以下の3ファイルで実装されています：

- `piano_params_stage2.py` line 518
- `guitar_params_stage2.py` line 497
- `strings_params_stage2.py` line 475

**現状**: `_blend_harmony()` を呼び出して和音情報を `self._overrides['harmony_blend']` に保存するのみ。

### Phase 26 への統合提案

Phase 26 は **ピッチ分布を直接操作していない** ため、以下のアプローチを推奨：

#### Option A: Phase 31 のみで完結（現在の実装） ✅
- **メリット**: シンプル、既存コードへの影響最小
- **適用タイミング**: ノート生成後の修正
- **効果**: スケール外音の自動修正

#### Option B: Phase 26 でピッチ分布にマスク適用（将来拡張）
```python
def _phase_26(self, part, section_meta, mix_context, params, seed):
    """Phase 26: Hybrid Harmony + Mode/Scale Mask"""
    # 既存の harmony blend 処理
    self._blend_harmony(...)
    
    # ★ 新規: ピッチ候補分布にマスク適用 ★
    # （実装には各楽器のピッチ生成ロジック詳細調査が必要）
```

**推奨**: まず Option A で運用し、必要に応じて Option B を検討。

## 🔄 今後の拡張

### 1. Bass への Phase 31 実装
```python
# generator/bass_params_stage2.py に追加
def _phase_31(self, part, section_meta, mix_context, params, seed):
    """Phase 31: Voice-Leading Guard（Bass対応）"""
    try:
        vl = params.get("voice_leading") or {}
        hints = part.get("hints") or {}
        chord_now = hints.get("blend_harmony") or {}
        chord_prev = section_meta.get("prev_chord") or {}
        self._voice_leading_smooth(part, section_meta, chord_now, chord_prev, vl)
    except Exception:
        return
```

### 2. 修正強度の調整
```python
# sections.json に追加
{
  "sections": [
    {
      "bar": 0,
      "key_hint": "D",
      "mode": "ionian",
      "scale_constraint_strength": 0.8  # 0.0=無効, 1.0=完全適用
    }
  ]
}
```

### 3. コード情報の活用
```python
# _apply_mode_scale_constraint で chord_root/chord_quality を使用
mask = scale_mask_for_point(
    t_ql=off_ql,
    sections=sections,
    chord_root=chord_now.get("root"),  # コードルート情報
    chord_quality=chord_now.get("quality")  # コード質情報
)
```

### 4. ログ強化
```python
# 修正統計の記録
corrections = {
    "total_notes": 0,
    "corrected_notes": 0,
    "corrections": []
}

# 修正時に記録
corrections["corrections"].append({
    "bar": bar_idx,
    "original_pitch": pitch,
    "corrected_pitch": new_pitch,
    "reason": "out_of_scale"
})
```

## 🎵 使用例

### 基本的な使い方

```json
{
  "sections": [
    {
      "bar": 0,
      "label": "intro",
      "key_hint": "D",
      "mode": "ionian"
    },
    {
      "bar": 43,
      "label": "chorus",
      "key_hint": "D",
      "mode": "lydian",
      "profile": "airy"
    },
    {
      "bar": 67,
      "label": "bridge",
      "key_hint": "G",
      "mode": "mixolydian"
    }
  ]
}
```

### 効果

1. **Intro (D Ionian)**: D, E, F#, G, A, B, C# のみ生成
2. **Chorus (D Lydian)**: D, E, F#, **G#**, A, B, C# - #4が映える
3. **Bridge (G Mixolydian)**: G, A, B, C, D, E, **F** - ♭7が映える

**スケール外音が自動的に修正され、キー・モードに沿った自然な旋律を生成**

## ✅ チェックリスト

- [x] `InstrumentStage2Base._apply_mode_scale_constraint()` 実装
- [x] `InstrumentStage2Base._voice_leading_smooth()` 拡張
- [x] Piano Phase 31 統合（自動）
- [x] Guitar Phase 31 統合（自動）
- [x] Strings Phase 31 統合（自動）
- [x] テスト作成 (`tests/test_phase31_integration.py`)
- [x] 全テストパス
- [x] NO-OP保証確認
- [ ] Bass Phase 31 実装（将来）
- [ ] 実楽曲でのテスト
- [ ] パフォーマンス測定

## 📚 関連ファイル

- `generator/instrument_stage2_base.py` - 基底クラス（メソッド実装）
- `generator/piano_params_stage2.py` - Piano Phase 31
- `generator/guitar_params_stage2.py` - Guitar Phase 31
- `generator/strings_params_stage2.py` - Strings Phase 31
- `ops/scale_modes.py` - マスク生成（完全版）
- `tests/test_phase31_integration.py` - 統合テスト
- `docs/sections_mode_scale_schema.json` - スキーマ例
- `README_MODE_SCALE_INTEGRATION.md` - 統合ガイド

## 🆘 トラブルシューティング

### Q1: スケール外音が修正されない

**A**: 
1. `sections.json` に `key_hint` があることを確認
2. `mode_hint` が無い場合は自動でIonian/Aeolianを推定
3. `voice_leading.enable` が `true` であることを確認

### Q2: 修正が強すぎる

**A**: 
1. 閾値を調整: `threshold = avg_mask * 0.70` → `0.50` など
2. または Phase 31 を無効化: `voice_leading.enable = false`

### Q3: Bass に適用されない

**A**: Bass には Phase 31 自体が未実装です。将来のアップデートで対応予定。

---

**作成日**: 2025年10月21日  
**バージョン**: 1.0.0  
**ステータス**: Phase 31統合完了 ✅
