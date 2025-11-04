# Phase A実装完了 - 次ステップガイド

## ✅ 完了事項

### 1. Backend切替実装 (100%)
- **ops/stems_features.py**: FeaturesBackend統合完了
  - `--backend-config configs/arranger_weights.yaml`でmadmom/librosa_enhanced/pyloudnorm切替
  - `--extend-bars`でbars_extended.parquet生成(start_sec/end_sec/drums_active列)
- **ops/features_backends.py**: 470行、全backend実装
  - `extract_beats_madmom()`, `extract_downbeats_madmom()`
  - `extract_hat_density_librosa_enhanced()`: 5-12kHz bandpass
  - `extract_loudness_pyloudnorm()`: EBU R128 LUFS

### 2. E2E stems特徴配線 (A案必須)
- **scripts/e2e_suno_arrangement.sh**: 
  - `--stems-features $STEMS_FEATURES`自動検出・渡し
  - stem_features.parquet存在時にrecommend_drums.pyへ配線

### 3. 実グルーヴ stems重み付け (B案)
- **scripts/recommend_drums.py**:
  - 既存: `density_boost=0.6`, `fill_boost=0.3`でhat_density/fill_likelihood統合
  - **NEW**: `search_best_pattern(drums_active=False)`で低密度パターン優先(+0.5ボーナス)
  - drums_active取得: `stem_df["drums_active"]`から小節ごとに読み取り
- **scripts/drums_midi_to_plan_real.py**:
  - `--stems-features`オプション追加
  - ヘルパー関数追加(`_load_stems_features()`, `_build_density_override()`, `_density_target_for()`)
  - ※実際の重み付けはrecommend_drums.py経由で既に効いている

---

## 🔍 次の確認事項 (15~30分)

### 実行前準備
```bash
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3

# Python環境活性化（必要に応じて）
# source .venv311/bin/activate  # または適切なvenv

# stem_features.parquet再生成（backend有効化）
python3 ops/stems_features.py \
  --stems song_packages/suno_project/song_001/stemswav_001 \
  --bars song_packages/suno_project/song_001/bars.parquet \
  --output song_packages/suno_project/song_001/stem_features.parquet \
  --backend-config configs/arranger_weights.yaml \
  --tempo-bpm 74.68 \
  --extend-bars  # bars_extended.parquet生成

# bars.parquet拡張版で上書き（全工程で使用）
cp song_packages/suno_project/song_001/bars_extended.parquet \
   song_packages/suno_project/song_001/bars.parquet
```

### 確認1: Backend切替ログ確認
```bash
# 上記stem_features.py実行ログで以下を確認:
# - "Backend config loaded from: configs/arranger_weights.yaml"
# - "drums_active: XX active bars, YY break bars"
# - hat_density平均が3~5程度（librosa-onlyは1~2）

# stem_features.parquet内容確認
python3 -c "
import pandas as pd
df = pd.read_parquet('song_packages/suno_project/song_001/stem_features.parquet')
print('Columns:', df.columns.tolist())
print('hat_density stats:', df['hat_density'].describe())
print('drums_active counts:', df['drums_active'].value_counts() if 'drums_active' in df.columns else 'N/A')
"
```

**期待結果**:
- `drums_active`列が存在（1/0の値）
- `hat_density`平均が**3~5**（librosa_enhanced効果）
- active bars: 120~130 / break bars: 20~30

---

### 確認2: E2E stems統合テスト
```bash
# E2Eワークフロー実行（stems特徴自動配線）
./scripts/e2e_suno_arrangement.sh \
  song_packages/suno_project/song_001 \
  --drums-mode rule \
  --kpi

# ログで以下を確認:
# - "🎯 Stems features detected: song_packages/.../stem_features.parquet"
# - "Stem integration: ENABLED (density_boost=0.6, fill_boost=0.3)"
# - "Density boosted: XX/150 bars"
```

**期待結果**:
- stems特徴が自動検出・読み込まれる
- density boosted bars が30~50程度(全体の20~30%)
- KPI Pass rate向上（後述）

---

### 確認3: MIDI健全性サニティ
```bash
# MIDI基本検証
python3 -c "
import pretty_midi
mid = pretty_midi.PrettyMIDI('song_packages/suno_project/song_001/full_arrangement.mid')

# Track 0以外にset_tempo無いか（Strings二倍問題再発防止）
for i, inst in enumerate(mid.instruments):
    print(f'Track {i}: {inst.name}, is_drum={inst.is_drum}, notes={len(inst.notes)}')

# Downbeats確認
downbeats = mid.get_downbeats()
print(f'Downbeats: {len(downbeats)} (期待: ≈151)')

# 尺確認
print(f'Total duration: {mid.get_end_time():.1f}s (期待: ≈482s)')
"
```

**合格基準**:
- ✅ Track 0 = Tempo Map のみ
- ✅ Downbeats ≈ 151 (±2)
- ✅ Duration ≈ 482s (±5s)

---

### 確認4: KPI狙い撃ち検証
```bash
# relative density Fail削減確認
python3 scripts/kpi_gate_enhanced.py \
  --midi song_packages/suno_project/song_001/full_arrangement.mid \
  --bars song_packages/suno_project/song_001/bars.parquet \
  --gate-config configs/gate_prod.yaml \
  --tempo-bpm 74.68 \
  --output song_packages/suno_project/song_001/kpi_gate_phase_a.json

# 結果確認
python3 -c "
import json
with open('song_packages/suno_project/song_001/kpi_gate_phase_a.json', 'r') as f:
    kpi = json.load(f)
    
summary = kpi['summary']
print(f'Pass rate: {summary[\"pass_rate\"]*100:.1f}%')
print(f'Fail total: {summary[\"fail_total\"]}')

# Fail内訳
by_type = kpi.get('fail_by_type', {})
print('\\nFail breakdown:')
for k, v in sorted(by_type.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v} bars')
"
```

**目標値（SLO）**:
- ✅ **Pass rate ≥ 90%**（当面の本番基準）
- ✅ "density too low (relative)" が**全Failの50%未満**
- ✅ "notes_per_bar too low" が**連続3 bars以上に集中しない**

**Phase A前後比較**（Mock Simulation基準）:
- hat_density: 1.21 → **4.04** (+234%)
- relative density fail: 150 → **70 bars** (-53%)
- KPI Pass rate: 80.5% → **87~90%** (+6.5~9.5%)

---

## 🚀 A/B比較テンプレート

```bash
# A) Phase A前（stems無効）
./scripts/e2e_suno_arrangement.sh \
  song_packages/suno_project/song_001 \
  --drums-mode rule \
  --kpi
mv song_packages/suno_project/song_001/kpi_gate_postgen.json \
   song_packages/suno_project/song_001/kpi_gate_before.json

# B) Phase A後（stems有効）
# stem_features.parquetを配置してからE2E実行
./scripts/e2e_suno_arrangement.sh \
  song_packages/suno_project/song_001 \
  --drums-mode rule \
  --kpi
mv song_packages/suno_project/song_001/kpi_gate_postgen.json \
   song_packages/suno_project/song_001/kpi_gate_after.json

# 差分確認
python3 -c "
import json
with open('song_packages/suno_project/song_001/kpi_gate_before.json', 'r') as f:
    before = json.load(f)
with open('song_packages/suno_project/song_001/kpi_gate_after.json', 'r') as f:
    after = json.load(f)

print('Phase A効果:')
print(f'  Pass rate: {before[\"summary\"][\"pass_rate\"]*100:.1f}% → {after[\"summary\"][\"pass_rate\"]*100:.1f}% ({after[\"summary\"][\"pass_rate\"]-before[\"summary\"][\"pass_rate\"]:+.1%})')
print(f'  Fail total: {before[\"summary\"][\"fail_total\"]} → {after[\"summary\"][\"fail_total\"]} ({after[\"summary\"][\"fail_total\"]-before[\"summary\"][\"fail_total\"]:+d})')
"
```

---

## ⚠️ 既知の落とし穴（回避策つき）

### 1. set_tempoトラック分散
**症状**: Strings等が2倍速で再生される  
**原因**: Track 0以外にset_tempoイベントが紛れる  
**回避**: midi_writerでTrack 0限定を厳守（現在実装済み）

### 2. bars.parquet vs MIDI最終バー長ズレ
**症状**: Downbeats判定が149/150で揺れる  
**原因**: bars.parquetの最終小節end_secとMIDI実長の微差  
**回避**: `epsilon_sec=0.02`（20ms）許容（実装済み）

### 3. stems hat_densityが過小（平均1~2）
**症状**: librosa-onlyで生成されている  
**原因**: backend-config未指定または古いparquet再利用  
**回避**: `--backend-config`必須、`--extend-bars`で再生成

---

## 📊 次の推奨パッチ（1~2h）

### A. bars.parquet標準化
```bash
# 全曲でbars_extended.parquetを標準bars.parquetとして使用
# (start_sec/end_sec/drums_active列を全工程で参照可能に)

# 1. stems_features.py修正（--extend-barsをデフォルトONに）
# 2. E2Eワークフローでbars.parquet → bars_extended.parquetへリネーム自動化
```

### B. Real Groove密度マッチング精度向上
- drums_midi_to_plan_real.pyで実際にstems重み付けを使う場合（現状はrecommend_drums.py経由で効果あり）
- 期待効果: "density too low" **14 → 2~4 bars** (-85%)

### C. Phase B移行（YAMNet）
```bash
# 1. TensorFlow/YAMNet導入
pip install tensorflow>=2.13,<2.14 tensorflow-hub>=0.14

# 2. arranger_weights.yaml変更
features_backend:
  hat_density: yamnet  # librosa_enhanced → yamnet

# 3. 効果検証
# 期待: hat_density平均 4.0 → 5~7、KPI Pass 90~95%
```

---

## 📝 完了チェックリスト

- [ ] stem_features.parquet再生成（backend有効）
- [ ] drums_active列存在確認
- [ ] hat_density平均3~5（librosa_enhanced効果）
- [ ] E2E stems特徴自動配線ログ確認
- [ ] MIDI健全性サニティ（Track 0限定、Downbeats≈151、Duration≈482s）
- [ ] KPI Pass rate ≥ 90%達成
- [ ] relative density Fail -30~60%削減確認
- [ ] A/B比較でPhase A効果定量化

---

## 🎯 成功基準（Production Ready）

| 指標 | Phase A前 | Phase A後（目標） | 実測値 |
|------|-----------|------------------|--------|
| **KPI Pass率** | 80.5% | ≥ 90% | ___ % |
| **hat_density平均** | 1.21 | 4~5 | ___ |
| **relative density fail** | 150 bars | ≤ 70 bars | ___ bars |
| **Stem boost activation** | 0% | 20~30% | ___ % |
| **drums_active検出** | N/A | 120~130 active / 20~30 break | ___ / ___ |

---

## 📞 トラブルシューティング

### Q: stem_features.parquetにdrums_active列が無い
**A**: 古いバージョンで生成されています。`--backend-config`と`--extend-bars`付きで再実行してください。

### Q: hat_density平均が1~2のまま
**A**: librosa-onlyで動作しています。以下を確認:
1. `configs/arranger_weights.yaml`の`features_backend`セクション存在
2. `--backend-config`オプション指定
3. madmom/scipy/pyloudnormのインストール（`pip install -r requirements.txt`）

### Q: E2Eで"No stem_features.parquet found"警告
**A**: 正常です（optional）。stem_features.parquetを生成してからE2E再実行すると自動配線されます。

### Q: KPI Pass率が85%未満
**A**: 以下を確認:
1. bars.parquetのdensity_target値が適正か（平均5~8）
2. stems特徴が効いているか（ログで"Density boosted"確認）
3. Fail集中区間（Intro/Outro等）へのブースト追加検討

---

## 次アクション
1. ✅ **まず確認**: 上記「確認1~4」を実行してPhase A効果を定量化
2. 🚀 **90%達成したら**: 推奨パッチ（bars標準化・Real Groove精度向上）
3. 🎨 **95%目指すなら**: Phase B（YAMNet）移行

---

**Phase A実装完了お疲れ様でした！** 🎉
次は実データでの効果検証です。上記確認作業でKPI Pass率90%達成を目指しましょう！
