# LAMDA Stage2 Extractor 完全機能達成版 実装ロードマップ

**作成日**: 2025年1月  
**目標**: 半年間の資産を活かし、DAWを超える音楽家AI基盤を完成させる  
**スキーマ**: lamda_v2.5 → lamda_v3.0 (完全版)

---

## エグゼクティブサマリー

### 現状評価

| カテゴリ | 完成度 | 状態 | 評価 |
|---------|--------|------|------|
| **骨組み** | 100% | ✅ 完成 | 外部chordmap優先、WL検証、CSV/Pickle/JSON出力 |
| **旧資産配線** | 43% | ⚠️ 部分実装 | 3モジュール済み、2モジュール未配線 |
| **Phase要件** | 65% | ⚠️ スケルトン | 骨組み完成、精度向上未完 |
| **高度機能** | 20% | ❌ 未着手 | RNN/Transformer、novelty完全統合 |

### 資産インベントリ

```yaml
総資産: 312 Python modules (utilities/)
現在配線済み: 3 modules
  - harmonic_utils.py (✅)
  - groove_sampler_v2.py (✅)
  - pb_math.py (✅)

配線対象（優先度高）: 2 modules
  - tempo_utils.py (⭐)
  - timing_utils.py (⭐)

配線候補（高度化フェーズ）: 12+ modules
  - groove_rnn_v2.py
  - groove_transformer.py
  - bass_utils.py
  - humanizer.py
  - ops/sections_from_audio.py
  - その他307 modules
```

### ギャップ分析（ChatGPT提案の7項目）

| # | 項目 | 現状 | 完成度 | 優先度 |
|---|------|------|--------|--------|
| 1 | **テンポ/QL変換厳密化** | pretty_midi概算 | 0% | ⭐⭐⭐ |
| 2 | **1小節コード化精度** | PC多数決 | 0% | ⭐⭐⭐ |
| 3 | **key_hint/転調** | harmonic_utils優先パス | 40% | ⭐⭐ |
| 4 | **sections novelty併用** | sections_json受け口のみ | 30% | ⭐⭐ |
| 5 | **groove** | groove_sampler_v2優先パス | 50% | ⭐⭐ |
| 6 | **controls** | pb_math優先パス | 60% | ⭐ |
| 7 | **roles詳細** | GMプログラム簡易 | 20% | ⭐⭐ |

---

## 実装計画（3段階アプローチ）

### フェーズ1: 未配線資産の統合【1-2週間】

**目標**: tempo_utils, timing_utils を配線し、基礎精度を向上

#### タスク1.1: tempo_utils統合
```python
# scripts/lamda_stage2_extractor.py 行57-69に追加
try:
    from utilities import tempo_utils
except Exception:
    tempo_utils = None
```

**配線箇所**:
- `extract_tempo_grid()` (推定行数: 3800-4000)
  - 現状: `60.0 / tempo_bpm` の概算
  - 改善: `tempo_utils.sec_to_ql(sec, tempo_bpm)` で厳密計算
  - 効果: QL座標精度向上 → modulations, groove連動改善

**合格基準**:
- ✅ テンポ90/120/180BPMで±0.01QL以内
- ✅ フォールバック動作確認（tempo_utils無しでも動作）

#### タスク1.2: timing_utils統合
```python
# scripts/lamda_stage2_extractor.py 行57-69に追加
try:
    from utilities import timing_utils
except Exception:
    timing_utils = None
```

**配線箇所**:
- `chordmap_unify()` 後処理 (推定行数: 4200-4300)
  - 現状: 最短音価制約なし
  - 改善: `timing_utils.merge_min_dwell(chordmap, min_ql=2.0)`
  - 効果: "Phase 26/31: key制約（QLスナップ）" 完全実装

**合格基準**:
- ✅ 最短2QL未満の和音が消失
- ✅ 同一和音連結（C→C→Am → C→Am）
- ✅ フォールバック動作確認

#### タスク1.3: extract_bar_chords() 精度向上
**配線箇所**:
- `extract_bar_chords()` (推定行数: 4100-4200)
  - 現状: PC多数決のみ
  - 改善: `harmonic_utils.estimate_bar_chords(notes, downbeats_sec)`
  - 効果: 7th/sus/add9/13th拡張、トライトーン回避

**合格基準**:
- ✅ Dm7, Gsus4, Cadd9を正しく抽出
- ✅ F#減三 → Gへのトライトーン回避
- ✅ フォールバック動作確認（harmonic_utils無しは従来通り）

**マイルストーン1**:
- ✅ lamda_v2.6リリース
- ✅ テスト通過: `pytest tests/test_stage2_tempo_timing.py`
- ✅ ドキュメント更新: CHANGELOG.md

---

### フェーズ2: 精度向上【1ヶ月】

**目標**: Phase13-32の全要件を完全実装

#### タスク2.1: harmonic_utils完全統合
**強化内容**:
1. `estimate_local_keys_extended()` パラメータ調整
   - `win_bars=4` → `win_bars=8` (転調検出精度向上)
   - `confidence_threshold=0.6` 追加

2. `estimate_bar_chords()` 7th拡張
   - 現状: `harmonic_utils.estimate_bar_chords()` スケルトン呼び出し
   - 改善: `chord_vocab="extended"` パラメータ追加
   - 効果: Dm7, Cmaj7, Gsus4, Cadd9, F13 対応

**合格基準**:
- ✅ ジャズスタンダード（Autumn Leaves）で7thコード50%以上認識
- ✅ 転調検出: 8小節窓で転調タイミング±1小節以内
- ✅ confidence_threshold=0.6未満は "key_changes=[]"

#### タスク2.2: groove_sampler_v2詳細パラメータ
**強化内容**:
1. `analyze_groove_extended()` 拡張
   - 現状: `groove_sampler_v2.analyze(midi_data, downbeats_ql)`
   - 改善: `resolution=480, quantize=True` パラメータ追加
   - 効果: swing_pct, backbeat_strength 精度向上

2. E(t)連動（Phase22/25/29）
   - `energy_curve` を `groove_sampler_v2.analyze()` に渡す
   - `velocity_correlation` フィールド追加

**合格基準**:
- ✅ スイング率: 50%-75%の範囲を±5%以内で検出
- ✅ バックビート強度: 0.5-0.9の範囲を±0.1以内で検出
- ✅ E(t)相関係数: 0.7以上（強いビートとエネルギーピーク同期）

#### タスク2.3: bass_motion詳細計算
**強化内容**:
1. `estimate_roles_extended()` 拡張
   - 現状: `bass_motion = max(2, min(14, avg_interval))`
   - 改善: `bass_utils.analyze_bass_line(bass_notes)` 呼び出し
   - 効果: walking_bass, pedal_point, chromatic_approach 検出

2. bass_utils統合
   ```python
   try:
       from utilities import bass_utils
   except Exception:
       bass_utils = None
   ```

**合格基準**:
- ✅ ウォーキングベース検出（4小節平均5-8半音移動）
- ✅ ペダルポイント検出（8小節95%同音）
- ✅ クロマチックアプローチ検出（半音進行30%以上）

**マイルストーン2**:
- ✅ lamda_v2.8リリース
- ✅ テスト通過: `pytest tests/test_stage2_precision.py`
- ✅ ドキュメント更新: PHASE_PRECISION_UPGRADE.md

---

### フェーズ3: 高度化【2-3ヶ月】

**目標**: DAWを超える機能実装（RNN/Transformer、novelty完全統合）

#### タスク3.1: novelty完全統合
**強化内容**:
1. `ops/sections_from_audio.py` 統合
   - 現状: `--sections-json` 受け口のみ
   - 改善: `ops.sections_from_audio.analyze()` 直接呼び出し
   - 効果: オーディオ解析→RMS+novelty→自動セクション分割

2. `auto_sections_from_energy_extended()` 拡張
   ```python
   try:
       from ops import sections_from_audio
   except Exception:
       sections_from_audio = None
   
   if sections_from_audio and hasattr(sections_from_audio, "analyze"):
       sections = sections_from_audio.analyze(
           audio_path, 
           min_section_bars=8,
           novelty_kernel_size=64
       )
   ```

**合格基準**:
- ✅ セクション境界: 人間ラベルと±2小節以内で80%一致
- ✅ Chorus/Pre検出: Chorusエネルギー > Verse 平均+20%
- ✅ フォールバック動作確認

#### タスク3.2: groove高度化（RNN/Transformer）
**強化内容**:
1. `groove_rnn_v2.py` 統合
   - マイクロタイミング予測（±50tick範囲）
   - 時系列パターン学習（16小節窓）

2. `groove_transformer.py` 統合
   - アテンション機構（ダウンビート間相関）
   - グローバルグルーヴ特徴抽出

**合格基準**:
- ✅ マイクロタイミング予測: RMSE < 20tick
- ✅ スイング率予測: MAE < 3%
- ✅ 計算時間: 1曲 < 30秒（GPU不要）

#### タスク3.3: 外れ値スコア実計算
**強化内容**:
1. Stage2 MIDI→ヒストグラム変換
   - 現状: `"outlier_score": 0.0` 固定
   - 改善: `humanizer.py` 統合 → ベロシティ分散、タイミング分散計算

2. `humanizer.py` 統合
   ```python
   try:
       from utilities import humanizer
   except Exception:
       humanizer = None
   
   if humanizer and hasattr(humanizer, "calculate_outlier_score"):
       score = humanizer.calculate_outlier_score(
           midi_data, 
           reference_dataset="LAMDA_v1"
       )
   ```

**合格基準**:
- ✅ 量子化MIDI: score < 0.3
- ✅ 人間演奏MIDI: 0.4 < score < 0.7
- ✅ ランダムMIDI: score > 0.8

#### タスク3.4: SIGNATURES時間軸化
**強化内容**:
1. 転拍子対応
   - 現状: `(numerator, denominator)` 固定
   - 改善: `[(start_ql, numerator, denominator), ...]` リスト化

2. `extract_time_signatures()` 拡張
   - pretty_midi.TimeSignature のタイムスタンプ保持
   - QL座標変換（tempo_utils使用）

**合格基準**:
- ✅ 4/4 → 3/4 → 4/4 の転拍子を正しく検出
- ✅ 境界位置: ±0.5QL以内
- ✅ スキーマ互換性維持（単一拍子の場合は従来形式）

**マイルストーン3**:
- ✅ lamda_v3.0リリース（完全版）
- ✅ テスト通過: `pytest tests/test_stage2_advanced.py`
- ✅ ドキュメント更新: LAMDA_V3_COMPLETION_REPORT.md
- ✅ ベンチマーク: LAMDA_v1.2データセット全曲（1000曲+）で検証

---

## 優先度マトリクス

### 縦軸: 実装難易度 / 横軸: Phase要件充足度

```
高難度 │                 │ 3.2 groove_rnn  │ 3.4 SIGNATURES時間軸
       │                 │ 3.3 outlier実計 │
───────┼─────────────────┼─────────────────┼──────────────────
       │ 1.3 bar_chords  │ 2.2 groove詳細  │ 3.1 novelty完全統合
       │                 │ 2.3 bass_motion │
───────┼─────────────────┼─────────────────┼──────────────────
低難度 │ 1.1 tempo_utils │ 2.1 harmonic完全│
       │ 1.2 timing_utils│                 │
───────┴─────────────────┴─────────────────┴──────────────────
       低優先度          中優先度          高優先度
```

**実装順序の推奨**:
1. **1.1, 1.2** (低難度・中優先度) → 基礎固め
2. **1.3** (中難度・中優先度) → 精度向上
3. **2.1, 2.2** (中難度・高優先度) → Phase要件完全充足
4. **2.3** (中難度・中優先度) → 詳細機能拡張
5. **3.1** (中難度・高優先度) → DAW超え機能①
6. **3.2, 3.3, 3.4** (高難度・高優先度) → DAW超え機能②

---

## マイルストーン

### 3ヶ月目標（2025年4月）
- ✅ フェーズ1完了（tempo_utils, timing_utils配線）
- ✅ フェーズ2完了（harmonic, groove, bass詳細実装）
- ✅ lamda_v2.8リリース
- ✅ テストカバレッジ: 80%以上

### 6ヶ月目標（2025年7月）
- ✅ フェーズ3完了（novelty, RNN/Transformer統合）
- ✅ lamda_v3.0リリース（完全版）
- ✅ LAMDA_v1.2データセット全曲検証
- ✅ ベンチマーク: 人間ラベルとの一致率80%以上

### 1年目標（2026年1月）
- ✅ リアルタイム解析（1曲 < 10秒）
- ✅ Web API公開（lamda-stage2-api.example.com）
- ✅ プラグイン開発（DAW統合: Ableton Link対応）
- ✅ 論文執筆・学会発表

---

## テスト戦略

### ユニットテスト
```bash
# フェーズ1
pytest tests/test_stage2_tempo_timing.py -v
pytest tests/test_stage2_harmonic_basic.py -v

# フェーズ2
pytest tests/test_stage2_precision.py -v
pytest tests/test_stage2_groove_advanced.py -v

# フェーズ3
pytest tests/test_stage2_advanced.py -v
pytest tests/test_stage2_novelty.py -v
```

### 統合テスト
```bash
# 100曲サンプルセット
pytest tests/test_stage2_integration_100.py --slow

# LAMDA_v1.2全曲（1000曲+）
pytest tests/test_stage2_integration_full.py --very-slow
```

### ベンチマーク
```bash
# 精度評価
python scripts/benchmark_stage2_accuracy.py \
  --dataset lamda_v1.2 \
  --metrics key,chord,section,groove \
  --output results/benchmark_v2.8.json

# 速度評価
python scripts/benchmark_stage2_speed.py \
  --num-songs 100 \
  --output results/speed_v2.8.json
```

### 品質ゲート（各フェーズ）
| メトリクス | フェーズ1 | フェーズ2 | フェーズ3 |
|----------|---------|---------|---------|
| テストカバレッジ | 70% | 80% | 85% |
| key一致率 | 60% | 75% | 80% |
| chord一致率 | 50% | 65% | 75% |
| section一致率 | 55% | 70% | 80% |
| groove RMSE | - | 0.15 | 0.10 |
| 処理速度（1曲） | <60秒 | <45秒 | <30秒 |

---

## リスク管理

### 技術リスク
| リスク | 影響度 | 対策 |
|-------|-------|------|
| 旧資産の互換性問題 | 高 | フォールバック機能必須実装 |
| RNN/Transformer計算コスト | 中 | GPU不要アルゴリズム選定 |
| LAMDA_v1.2との整合性 | 高 | スキーマバージョン明示 |
| 転拍子MIDI希少性 | 低 | 合成データ生成 |

### スケジュールリスク
| リスク | 影響度 | 対策 |
|-------|-------|------|
| フェーズ2長期化 | 中 | タスク2.3を延期可能 |
| novelty統合複雑化 | 高 | ops/担当者と事前調整 |
| テストデータ不足 | 中 | LAMDA_v1.2から1000曲抽出 |

---

## 配線表（技術仕様）

### フェーズ1: 未配線資産
```python
# scripts/lamda_stage2_extractor.py
# 行57-69に追加

try:
    from utilities import tempo_utils
except Exception:
    tempo_utils = None

try:
    from utilities import timing_utils
except Exception:
    timing_utils = None
```

**使用箇所**:
1. `extract_tempo_grid()` (行3800-4000):
   ```python
   if tempo_utils and hasattr(tempo_utils, "sec_to_ql"):
       ql = tempo_utils.sec_to_ql(sec, tempo_bpm)
   else:
       ql = (sec * tempo_bpm) / 60.0  # fallback
   ```

2. `chordmap_unify()` 後処理 (行4200-4300):
   ```python
   if timing_utils and hasattr(timing_utils, "merge_min_dwell"):
       chordmap = timing_utils.merge_min_dwell(chordmap, min_ql=2.0)
   ```

3. `extract_bar_chords()` (行4100-4200):
   ```python
   if harmonic_utils and hasattr(harmonic_utils, "estimate_bar_chords"):
       bar_chords = harmonic_utils.estimate_bar_chords(
           notes, downbeats_sec, chord_vocab="extended"
       )
   else:
       # 従来のPC多数決
   ```

### フェーズ2: 精度向上
```python
# bass_utils統合
try:
    from utilities import bass_utils
except Exception:
    bass_utils = None

# estimate_roles_extended() 拡張
if bass_utils and hasattr(bass_utils, "analyze_bass_line"):
    bass_analysis = bass_utils.analyze_bass_line(bass_notes)
    roles["bass_motion"] = bass_analysis.get("avg_interval", 7)
    roles["walking_bass"] = bass_analysis.get("walking_bass", False)
    roles["pedal_point"] = bass_analysis.get("pedal_point", False)
```

### フェーズ3: 高度化
```python
# ops/sections_from_audio.py統合
try:
    from ops import sections_from_audio
except Exception:
    sections_from_audio = None

# auto_sections_from_energy_extended() 拡張
if sections_from_audio and hasattr(sections_from_audio, "analyze"):
    sections = sections_from_audio.analyze(
        audio_path=audio_path,
        min_section_bars=8,
        novelty_kernel_size=64
    )
else:
    # 従来のRMSピーク検出
```

---

## 成功基準

### 技術基準
- ✅ スキーマバージョン: lamda_v3.0
- ✅ テストカバレッジ: 85%以上
- ✅ 処理速度: 1曲 < 30秒（CPU環境）
- ✅ 精度: 人間ラベルとの一致率80%以上（key, chord, section）

### ビジネス基準
- ✅ LAMDA_v1.2データセット全曲対応（1000曲+）
- ✅ CI/CD完全自動化（GitHub Actions）
- ✅ ドキュメント完備（API仕様、チュートリアル）
- ✅ 外部ユーザーテスト（3名以上）

### ビジョン基準
- ✅ **DAWを超える機能**: novelty自動セクション、RNN/Transformerグルーヴ予測
- ✅ **音楽家AI**: 転調検出、7thコード認識、ウォーキングベース検出
- ✅ **半年間の資産活用**: utilities/312モジュールのうち10+モジュール統合

---

## 推奨ファイルリスト（Copilot参照用）

フェーズ1実装時に参照すべきファイル:
```
scripts/lamda_stage2_extractor.py         # メインファイル（5,955行）
utilities/tempo_utils.py                  # ⭐配線対象
utilities/timing_utils.py                 # ⭐配線対象
utilities/harmonic_utils.py               # ✅配線済み（拡張必要）
utilities/groove_sampler_v2.py            # ✅配線済み（パラメータ調整必要）
utilities/pb_math.py                      # ✅配線済み
LAMDA_STAGE2_PHASE_PATCHES.md             # 直前セッションの実装記録
PHASE_*.md                                # Phase13-32の要件定義
```

フェーズ2実装時に追加参照:
```
utilities/bass_utils.py                   # bass_motion詳細計算
utilities/humanizer.py                    # 外れ値スコア計算（フェーズ3でも使用）
```

フェーズ3実装時に追加参照:
```
ops/sections_from_audio.py                # novelty完全統合
utilities/groove_rnn_v2.py                # RNN統合
utilities/groove_transformer.py           # Transformer統合
```

---

## 次のアクション

1. **フェーズ1開始**:
   ```bash
   # ブランチ作成
   git checkout -b feature/phase1-tempo-timing-integration
   
   # tempo_utils, timing_utils読み込み
   cat utilities/tempo_utils.py
   cat utilities/timing_utils.py
   
   # 配線実装
   # scripts/lamda_stage2_extractor.py 編集
   ```

2. **テスト準備**:
   ```bash
   # テストファイル作成
   touch tests/test_stage2_tempo_timing.py
   
   # テストデータ準備
   python scripts/prepare_test_data.py --phase 1 --num-songs 10
   ```

3. **ドキュメント更新**:
   - CHANGELOG.md に "Phase1: tempo_utils, timing_utils統合" 追加
   - README.md の使用例更新

---

**作成者**: GitHub Copilot  
**レビュー**: 半年間の開発成果（utilities/312モジュール）を最大限活用  
**次回更新**: フェーズ1完了時（lamda_v2.6リリース時）
