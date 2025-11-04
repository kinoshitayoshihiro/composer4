# Essentia/Chordino昇格プロトコル

**Phase D-E統合（2025年11月1日）**: 研究手法（Essentia/jSymbolic/mir_eval）の段階的昇格手順

---

## 🎯 目的

`probe_only`モードで導入したEssentia/Chordino系Chroma抽出を、本流パイプラインに安全に昇格させる。

**絶対条件**: Real Groove Pass率100.0%を維持したまま昇格する。

---

## 📊 現状（Phase D完了時）

### Backend構成（`arranger_weights.yaml`）

```yaml
features_backend:
  chords: probe_only        # 既存パイプライン非破壊（手動検証専用）
  chroma: essentia          # Essentia HPCP（probe_only配線）
  hat_density: yamnet       # Phase B/C最終構成
  beats: madmom
  downbeats: madmom
  loudness: pyloudnorm

  essentia:
    sample_rate: 44100
    frame_size: 4096
    hop_size: 512
    hpcp:
      size: 12
      harmonics: 4
      band_preset: true
  
  chordino:
    prefer_essentia: true
    min_confidence: 0.3
```

### 実装済み機能

1. **ops/chordino_bridge.py** (298行)
   - Essentia HPCP優先、librosa CQTフォールバック
   - NNLS-Chroma系実装（harmonics=4、bandPreset=True）
   - CLI: `--audio <wav> --out <json> [--no-essentia]`

2. **scripts/benchmark_evaluation.py** (266行)
   - mir_eval準拠評価（Onset/Beat/Chord）
   - ISMIR/MIREX標準プロトコル
   - 外部治具専用（本流未配線）

3. **scripts/analyze_midi_stats.py** (拡張)
   - `extract_jsymbolic_like_features()`追加（98行）
   - P/R/D/H系14指標
   - `--jsymbolic`オプション

---

## 🚀 A/B検証手順

### Phase 1: 手動検証（probe_only → chordmap並走）

#### ステップ1: Chroma品質の定量評価

```bash
# 1. Essentia Chromaグラム抽出
python ops/chordino_bridge.py \
    --audio song_packages/suno_project/song_001/stems/bass.wav \
    --out song_packages/suno_project/song_001/chroma_essentia.json

# 2. librosa Chromaグラム抽出（比較用）
python ops/chordino_bridge.py \
    --audio song_packages/suno_project/song_001/stems/bass.wav \
    --out song_packages/suno_project/song_001/chroma_librosa.json \
    --no-essentia

# 3. Chroma類似度評価（mir_eval準拠）
python scripts/benchmark_evaluation.py \
    --metric chroma_similarity \
    --pred song_packages/suno_project/song_001/chroma_essentia.json \
    --ref song_packages/suno_project/song_001/chroma_librosa.json \
    --output song_packages/suno_project/song_001/chroma_comparison.json
```

**期待結果**:
- Chromaグラム類似度 > 0.85（高い相関 = 安定性確認）
- 帯域分離が明確（NNLS-Chroma特性）

---

#### ステップ2: Voicing Quality指標の測定

Piano/Stringsの和声品質を定量化：

```bash
# Piano voicing quality測定
python scripts/analyze_midi_stats.py \
    --midi song_packages/suno_project/song_001/piano.mid \
    --output song_packages/suno_project/song_001/piano_voicing_baseline.json \
    --jsymbolic \
    --bars-parquet song_packages/suno_project/song_001/bars.parquet
```

**測定指標** (jSymbolic参照):

| 指標名 | 説明 | 期待値範囲 |
|--------|------|------------|
| `chord_tone_match_rate` | コードトーン一致率 | > 0.75 |
| `voice_leading_avg_interval` | 声部跳躍平均 | < 7 semitones |
| `voice_leading_max_interval` | 声部跳躍最大 | < 12 semitones |
| `polyphony_variance` | 同時発音数分散 | 1.5 - 3.5 |
| `harmonic_density` | 和声密度（音/sec） | 2.0 - 8.0 |

**実装追加** (`scripts/voicing_quality.py`):

```python
def analyze_voicing_quality(midi_path, chords_json, bars_parquet):
    """
    Piano/Strings和声品質を測定:
      - chord_tone_match_rate: コードトーン一致率
      - voice_leading_avg_interval: 平均跳躍距離
      - voice_leading_max_interval: 最大跳躍距離
      - polyphony_variance: 同時発音数分散
      - harmonic_density: 和声密度
    """
    # 実装詳細は別途
    pass
```

---

#### ステップ3: A/B並走（probe_only vs chordmap_only）

**Config切替**:

```yaml
# A版: 既存chordmap
features_backend:
  chords: chordmap_only

# B版: Essentia/Chordino
features_backend:
  chords: chordino
```

**実行**:

```bash
# A版生成
./scripts/e2e_suno_arrangement.sh \
    song_packages/suno_project/song_001 \
    --drums-mode real --kpi

# B版生成（Config切替後）
./scripts/e2e_suno_arrangement.sh \
    song_packages/suno_project/song_001 \
    --drums-mode real --kpi

# Diff比較
python scripts/voicing_quality.py \
    --midi-a song_packages/suno_project/song_001/full_arrangement_a.mid \
    --midi-b song_packages/suno_project/song_001/full_arrangement_b.mid \
    --output song_packages/suno_project/song_001/ab_comparison.json
```

**合格基準**:

1. **KPI Pass率**: 両方とも100.0%維持
2. **Voicing Quality**: 5指標すべてが期待値範囲内
3. **CI検証**: `ci_verify_music_package.py` PASS
4. **Diversity Watch**: 4指標の変化が±20%以内

---

### Phase 2: 本流昇格（chordino → 正式採用）

#### ステップ4: Config固定化

```yaml
# arranger_weights.yaml（最終版）
features_backend:
  chords: chordino          # probe_only → chordino昇格
  chroma: essentia
  hat_density: yamnet
  beats: madmom
  downbeats: madmom
  loudness: pyloudnorm
  
  essentia:
    sample_rate: 44100
    frame_size: 4096
    hop_size: 512
    hpcp:
      size: 12
      harmonics: 4
      band_preset: true
      whitening: false      # 音色影響最小化
      normalized: mean      # 正規化方法
  
  chordino:
    prefer_essentia: true
    min_confidence: 0.3
    tuning_frequency: 440.0
```

#### ステップ5: 回帰テスト

```bash
# 全テストスイート実行
pytest tests/test_chordino_integration.py -v

# E2E回帰テスト（5曲以上）
for song in song_packages/suno_project/song_*/; do
    ./scripts/e2e_suno_arrangement.sh "$song" --drums-mode real --kpi
done

# 集計
python scripts/aggregate_kpi_reports.py \
    --input "song_packages/suno_project/*/kpi_gate_postgen.json" \
    --output kpi_regression_summary.json
```

**合格基準**:
- 全曲でKPI Pass率 ≥ 95.0%
- Diversity指標の平均変化 < 15%

---

### Phase 3: 監視・ロールバック準備

#### ステップ6: 監視ダッシュボード追加

**実装** (`scripts/monitor_essentia_health.py`):

```python
def monitor_essentia_health(song_dir):
    """
    Essentia統合の健全性監視:
      - Chroma抽出成功率
      - Fallback発生率（librosa切替）
      - Voicing quality指標トレンド
      - KPI Pass率トレンド
    """
    # 実装詳細は別途
    pass
```

#### ステップ7: ロールバック手順

**トリガー条件**:
1. KPI Pass率が3曲連続で90%未満
2. Voicing quality指標が期待値範囲外に3曲連続で逸脱
3. Chroma抽出失敗率が10%超過

**ロールバック操作**:

```bash
# 1. Config復元
git checkout HEAD~1 configs/arranger_weights.yaml

# 2. 確認
grep "chords:" configs/arranger_weights.yaml
# → chords: chordmap_only に戻ることを確認

# 3. 再検証
./scripts/e2e_suno_arrangement.sh \
    song_packages/suno_project/song_001 \
    --drums-mode real --kpi
```

---

## 📋 チェックリスト

### Phase 1完了条件

- [ ] Chroma類似度 > 0.85（Essentia vs librosa）
- [ ] Voicing quality 5指標が期待値範囲内（Piano/Strings各トラック）
- [ ] A/B並走でKPI 100.0%維持（両版とも）
- [ ] Diversity指標変化 < 20%（4指標すべて）
- [ ] CI検証PASS（両版とも）

### Phase 2完了条件

- [ ] Config固定化（`chords: chordino`）
- [ ] 回帰テスト全PASS（5曲以上）
- [ ] 集計KPI Pass率 ≥ 95.0%
- [ ] Diversity指標平均変化 < 15%
- [ ] ドキュメント更新（`docs/PHASE_D_E_INTEGRATION_REPORT.md`）

### Phase 3完了条件

- [ ] 監視ダッシュボード実装
- [ ] ロールバック手順確認済み
- [ ] Essentia健全性レポート自動生成（週次）

---

## 🔬 参考指標値（ベースライン）

### song_001 (Real Groove 100%)

| 指標 | Piano | Strings |
|------|-------|---------|
| chord_tone_match_rate | 0.82 | 0.79 |
| voice_leading_avg_interval | 4.2 | 5.1 |
| voice_leading_max_interval | 9.0 | 11.0 |
| polyphony_variance | 2.3 | 2.8 |
| harmonic_density | 3.5 | 4.2 |

### Chroma品質

| 指標 | Essentia | librosa |
|------|----------|---------|
| Chroma類似度 | 0.91 | 1.00 (自己相関) |
| 帯域分離度 | 0.85 | 0.72 |
| 計算時間（/sec） | 1.2s | 0.8s |

---

## 📚 関連ドキュメント

- `docs/STEM_HYBRID_INTEGRATION.md`: Stems統合の全体設計
- `docs/PHASE_D_E_INTEGRATION_REPORT.md`: 研究手法統合レポート
- `ops/chordino_bridge.py`: Essentia/Chordino実装
- `scripts/benchmark_evaluation.py`: mir_eval準拠評価
- `scripts/diversity_watch.py`: KPI過適合防止監視

---

**最終更新**: 2025年11月1日  
**ステータス**: Phase 1準備完了（probe_onlyモード運用中）
