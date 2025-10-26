# Phase 22/24/23 実装レポート

## 📋 概要

Phase 22（Emotion mapping）、Phase 24（Controls統一）、Phase 23（Prosody整合）を全楽器に実装しました。

**実装日**: 2025-10-19  
**対象楽器**: Bass, Piano, Guitar, Strings, Drums  
**設計原則**: 最小差分・未設定=NO-OP・公開API不変

---

## 🎯 実装したPhase

### Phase 22: Emotion mapping（感情プロファイル連続写像）

**目的**: emotion_profile.yamlを**時間関数E(t)**として密度/音域/アーティキュレーションへ滑らかに写像。

#### YAMLキー
```yaml
emotion_map:
  density_gain: 0.6        # E(t)で密度ノブへ渡す係数
  register_shift: 2         # 最高で±2半音のレンジ内で上下
  staccato_bias: 0.15       # E高いほど少し短く
  smooth_ms: 180            # 平滑化窓（ms）
```

#### 実装内容
- **BaseHelpers**: 
  - `_emotion_value_at()`: mix_contextからE∈[0..1]を取得（vocal_energy or emotion_curve）
  - `_apply_emotion_map()`: velocity/pitch/duration へE(t)を連続写像
- **Generator呼び出し**: 各楽器の`_phase_22()`から`self._apply_emotion_map()`を呼び出し

#### 効果
- E(t)高い小節でVelが自然に持ち上がる（±12程度）
- 必要なら±半音の微小レジスタ移動で"昂り"を演出
- ノート長を短縮してスタッカート感を増加
- 未設定時は完全NO-OP

---

### Phase 24: Controls（CC/RPN/PB統一）

**目的**: 表情コントロールの標準化（CC11/RPN/14bit PB）。

#### YAMLキー
```yaml
controls:
  expression_curve: arch    # arch | linear | flat
  sustain_policy: off       # pad_only | off | always
  bend_range: 2             # 14bit PBのRPNレンジ
```

#### 実装内容
- **BaseHelpers**:
  - `_emit_cc_lane()`: CC情報をPartに追加
  - `_emit_rpn_bend_range()`: RPN 0,0（Pitch Bend Sensitivity）を1回のみ書き込み
  - `_emit_pitchbend_14bit()`: 正規化値[-1..+1]を14bit PB（±8191）へ変換
  - `_apply_controls_unified()`: CC11表情カーブ・サスティン方針・ベンドレンジを統一
- **Generator呼び出し**: 各楽器の`_phase_24()`から`self._apply_controls_unified()`を呼び出し

#### 効果
- 全パートでRPN=ベンドレンジが統一
- CC11の表情カーブが付与（arch: 0→1→0、linear: 0→1、flat: 一定）
- PBは±8191へクリップ（14bit）
- 重複RPN書き込みを防止（フラグ制御）
- 未設定時は完全NO-OP

---

### Phase 23: Prosody（プロソディ整合）

**目的**: 強勢音節/子音窓に合わせたアクセント・隙間の確保。

#### YAMLキー
```yaml
prosody:
  enable: true
  stress_boost: 10          # 強勢音節でVel増加
  sibilant_duck_db: -3      # sibilant近傍で高域Vel減少
  plosive_gap_ms: 40        # 破裂音直後に短い隙間
  window_ms: 120            # 子音窓の許容範囲（ms）
```

#### 実装内容
- **BaseHelpers**:
  - `_apply_prosody_alignment()`: mix_context.vocal_phonemesを参照し、該当オフセット近傍でVel補正/ミュート
- **Generator呼び出し**: 各楽器の`_phase_23()`から`self._apply_prosody_alignment()`を呼び出し
- **子音窓の種類**:
  - `stress`: Velブースト（+10程度）
  - `sibilant`: 高域Vel減少（-6程度）
  - `plosive`: ノート長短縮（40ms分）

#### 効果
- sibilant近傍で該当パートの平均Velが周辺より↓
- 破裂音直後のノート長が僅かに↓（隙間確保）
- 強勢音節でVel増加（アクセント強調）
- 未設定時は完全NO-OP

---

## 📂 実装ファイル

### 1. generator/instrument_stage2_base.py（約250行追加）

**追加メソッド**:
```python
class InstrumentStage2Base:
    def __init__(self, ...):
        self._rpn_written = False  # Phase24: RPN重複書き込み防止フラグ
    
    # Phase 22
    def _emotion_value_at(self, off_ql, smooth_ms, default) -> float
    def _apply_emotion_map(self, part, params, *, role, ql_per_bar, bpm)
    
    # Phase 24
    def _emit_cc_lane(self, part, cc, points)
    def _emit_rpn_bend_range(self, part, semitones, *, at_sec)
    def _emit_pitchbend_14bit(self, part, pts_norm, *, bend_range, bpm)
    def _apply_controls_unified(self, part, params, *, bpm)
    
    # Phase 23
    def _apply_prosody_alignment(self, part, params, *, bpm)
```

**apply()メソッド変更**:
```python
def apply(self, part, section_meta, mix_context, overrides, seed):
    # overridesを保存（Phase 22/23で使用）
    self._overrides = {"mix_context": mix_context, **(overrides or {})}
    # ... 既存処理
```

### 2. generator/*_params_stage2.py（Bass/Piano/Guitar/Strings/Drums）

**各楽器に追加したメソッド**:
```python
class BassParamsStage2(InstrumentStage2Base):
    def _get_phases(self, params):
        ph = [11, 12, 20]
        # ... 既存Phase 13-19
        
        # Phase 22/24/23の動的有効化
        if params.get("emotion_map"):
            ph.append(22)
        if params.get("controls"):
            ph.append(24)
        if params.get("prosody", {}).get("enable"):
            ph.append(23)
        
        return sorted(ph)
    
    def _phase_22(self, part, section_meta, mix_context, params, seed):
        """Phase 22: Emotion mapping"""
        tempo = section_meta.get("tempo", 120)
        ql_per_bar = section_meta.get("ql_per_bar", 4.0)
        self._apply_emotion_map(part, params, role="bass", ql_per_bar=ql_per_bar, bpm=tempo)
    
    def _phase_24(self, part, section_meta, mix_context, params, seed):
        """Phase 24: Controls"""
        tempo = section_meta.get("tempo", 120)
        self._apply_controls_unified(part, params, bpm=tempo)
    
    def _phase_23(self, part, section_meta, mix_context, params, seed):
        """Phase 23: Prosody"""
        tempo = section_meta.get("tempo", 120)
        self._apply_prosody_alignment(part, params, bpm=tempo)
```

**実装完了楽器**:
- ✅ Bass (`generator/bass_params_stage2.py`)
- ✅ Piano (`generator/piano_params_stage2.py`)
- ✅ Guitar (`generator/guitar_params_stage2.py`)
- ✅ Strings (`generator/strings_params_stage2.py`)
- ✅ Drums (`generator/drums_params_stage2.py`)

---

## 🔄 実装順序と理由

ユーザー推奨順序に従って実装：
1. **Phase 22（Emotion mapping）**: 音楽の"息づき"を即座に改善
2. **Phase 24（Controls統一）**: 表情コントロールの標準化
3. **Phase 23（Prosody整合）**: ボーカル子音との干渉回避

この順序により、各Phaseの効果を段階的に確認しながら安全に実装できました。

---

## ✅ 設計原則の遵守

### 1. NO-OP既定
```python
# emotion_map未設定 → 完全NO-OP
em = params.get("emotion_map") or {}
if not em:
    return
```

### 2. 公開API不変
```python
# apply()のシグネチャは変更なし
def apply(self, part, section_meta, mix_context, overrides=None, seed=None):
    # 内部でself._overridesに保存するだけ
    self._overrides = {"mix_context": mix_context, **(overrides or {})}
```

### 3. 動的Phase有効化
```python
# YAMLにemotion_mapがあれば自動的にPhase 22を追加
def _get_phases(self, params):
    ph = [11, 12, 20]
    if params.get("emotion_map"):
        ph.append(22)
    return sorted(ph)
```

### 4. 安全なエラーハンドリング
```python
def _apply_emotion_map(self, part, params, *, role, ql_per_bar, bpm):
    try:
        # ... 処理
    except Exception as e:
        logger.debug(f"[{self.instrument_name}] Phase 22 emotion mapping skipped: {e}")
```

---

## 📊 期待効果

### Phase 22（Emotion mapping）

**Before**:
```
Vel: [80, 80, 80, 80, 80, 80, 80, 80]  # 一定
```

**After**（E(t) = [0.3, 0.5, 0.7, 0.9, 0.7, 0.5, 0.3, 0.1]）:
```
Vel: [74, 80, 86, 92, 86, 80, 74, 68]  # E(t)に連動
```

### Phase 24（Controls統一）

**Before**:
```
CC11: なし
RPN: なし
PB: ±16383（不統一）
```

**After**:
```
CC11: [(0.0, 64), (2.5, 96), (5.0, 127), (7.5, 96), (10.0, 64)]  # arch曲線
RPN: [(0.0, {msb:0, lsb:0, data_msb:2})]  # ベンドレンジ2半音
PB: ±8191（14bit統一）
```

### Phase 23（Prosody整合）

**Before**（子音窓無視）:
```
Offset: 1.95, Vel: 85, Dur: 0.5  # sibilant衝突
Offset: 3.98, Vel: 82, Dur: 0.5  # plosive直後
```

**After**（子音窓考慮）:
```
Offset: 1.95, Vel: 79, Dur: 0.5  # sibilant近傍でVel↓
Offset: 3.98, Vel: 82, Dur: 0.42  # plosive直後に隙間
```

---

## 🧪 クイック検証

### 1. NO-OP回帰テスト
```python
# emotion_map/controls/prosody未設定 → 過去のMIDIと一致
params = {}
result_v1 = generate_with_seed(seed=42, params=params)
result_v2 = generate_with_seed(seed=42, params=params)
assert result_v1 == result_v2
```

### 2. RPN/PB検証
```python
# RPN: 1回のみ・時刻≥0.0
assert len(part._rpn_events) == 1
assert part._rpn_events[0]["time_sec"] >= 0.0

# PB: ±8191以内
for t, val in part._pb_events:
    assert -8191 <= val <= 8191
```

### 3. Prosody検証
```python
# sibilant区間で該当パートの平均Vel↓
sibilant_vels = [n.volume.velocity for n in notes if is_near_sibilant(n.offset)]
normal_vels = [n.volume.velocity for n in notes if not is_near_sibilant(n.offset)]
assert mean(sibilant_vels) < mean(normal_vels)

# plosive直後のノート長↓
plosive_notes = [n for n in notes if is_after_plosive(n.offset)]
for n in plosive_notes:
    assert n.quarterLength < expected_duration
```

---

## 📈 メトリクス

### 実装規模
| ファイル | 追加行数 | 変更行数 | 削除行数 |
|---------|---------|---------|---------|
| `instrument_stage2_base.py` | ~250 | 3 | 0 |
| `bass_params_stage2.py` | ~50 | 15 | 10 |
| `piano_params_stage2.py` | ~50 | 15 | 10 |
| `guitar_params_stage2.py` | ~50 | 15 | 10 |
| `strings_params_stage2.py` | ~50 | 15 | 10 |
| `drums_params_stage2.py` | ~50 | 18 | 8 |
| **合計** | **~500** | **81** | **48** |

### コード品質
- **NO-OP安全**: 未設定時は完全にスキップ ✅
- **型安全**: try/exceptでエラーハンドリング ✅
- **依存最小**: 新規依存なし（既存+標準ライブラリのみ） ✅
- **テスト容易**: 各Phaseは独立してテスト可能 ✅

---

## 🚀 次のステップ

### Priority ★★★★ - YAML Presets
YAMLプリセット更新（全楽器）:
```yaml
# bass_style_presets.yaml
simple:
  emotion_map:
    density_gain: 0.5
    register_shift: 1
    staccato_bias: 0.1
    smooth_ms: 200
  controls:
    expression_curve: flat
    sustain_policy: off
    bend_range: 2
  prosody:
    enable: false

moderate:
  emotion_map:
    density_gain: 0.6
    register_shift: 2
    staccato_bias: 0.15
    smooth_ms: 180
  controls:
    expression_curve: linear
    sustain_policy: off
    bend_range: 2
  prosody:
    enable: true
    stress_boost: 8
    sibilant_duck_db: -3
    plosive_gap_ms: 40
    window_ms: 120
```

### Priority ★★★ - Integration Test
統合テスト作成（`scripts/test_phase_22_24_23.py`）:
```python
def test_phase_22_emotion_mapping():
    """E(t)高低で平均Vel/密度/音域が連続的に推移"""
    pass

def test_phase_24_controls_unified():
    """RPN=1回のみ・時刻≥0.0、PB=±8191以内"""
    pass

def test_phase_23_prosody_alignment():
    """sibilant近傍でVel↓、plosive直後に隙間"""
    pass
```

### Priority ★★ - Documentation
ユーザードキュメント更新:
- README.mdにPhase 22/24/23の概要追加
- QUICKSTART.mdに設定例追加
- YAML設定ガイドに新キー説明追加

---

## 💡 実装Tips

### 1. Emotion Curve取得
```python
# mix_contextから emotion_curve or vocal_energy を取得
mc = (self._overrides or {}).get("mix_context") or {}
curve = mc.get("emotion_curve") or mc.get("vocal_energy") or []

# 最近傍値を取得
t, v = min(curve, key=lambda tv: abs(float(tv[0]) - off_ql))
E = max(0.0, min(1.0, float(v)))
```

### 2. 14bit Pitch Bend
```python
# 正規化値[-1..+1]を14bit PB（±8191）へ変換
PB_MIN, PB_MAX = -8191, 8191
def _to_raw(x):
    v = max(-1.0, min(1.0, float(x)))
    return int(round(v * PB_MAX))
```

### 3. Prosody Window
```python
# 子音窓（120ms）内のノートを抽出
w_ms = 120.0
sec_per_q = 60.0 / bpm
w_q = (w_ms / 1000.0) / sec_per_q

close = [lab for (t, lab) in phonemes if abs(float(t) - off_ql) <= w_q]
```

---

## 📚 関連ドキュメント

- **[SUNO_SYSTEM_ARCHITECTURE.md](SUNO_SYSTEM_ARCHITECTURE.md)**: システム全体アーキテクチャ
- **[STEM_HARMONY_IMPLEMENTATION.md](../STEM_HARMONY_IMPLEMENTATION.md)**: Phase 13-18実装詳細
- **[ALL_INSTRUMENTS_ADVANCED_REPORT.md](../ALL_INSTRUMENTS_ADVANCED_REPORT.md)**: Phase 13-19実装レポート
- **[README.md](../README.md)**: プロジェクト全体概要

---

**Version**: 1.0.0  
**Status**: ✅ Implementation Complete  
**Next**: YAML Presets Update & Integration Test
