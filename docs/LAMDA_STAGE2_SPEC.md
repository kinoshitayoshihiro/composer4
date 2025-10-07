# LAMDa Drum Loops Stage 2 Specification

> 更新日: 2025-10-07
> `score_distribution` は `score.threshold_passed = true` のループに限定して算出し、品質トレンド監視時は同一フィルタ条件を適用する。


Stage 2 は Stage 1 で得たドラムループを **学習・検索しやすい正規化データセット** に昇格させるフェーズです。本書では入出力、スキーマ、メトリクス、スコアリング、運用要件をまとめます。

## 1. 目的と全体像

- Stage 1 出力 (dedupe済み MIDI と shard metadata) を再パースし、**イベント粒度**と**ループ粒度**の正規化テーブルを作成する。
- 既存 `lamda_tools.metrics` に未実装の指標を追加し、**5軸スコア (Timing / Velocity / Groove Harmony / Drum Cohesion / Structure)** を 0–100 点で算出する。
- スコア閾値 (既定 70 点) を満たさないループを **再処理キュー** に自動登録する。
- 産物は Parquet を正とし、軽量 CSV/JSONL を併記する。すべての生成物に `pipeline_version`, `git_commit`, `data_digest` を付与する。

```
Stage1 (dedupe) ──▶ Stage2 extractor ──▶ canonical_events.parquet / loop_summary.csv / metrics_score.jsonl / retries/...
                                          └─▶ stage2_summary.json (統計)
```

## 2. 入力と前処理

| 種別 | パス | メモ |
| ---- | ---- | ---- |
| クリーン済み MIDI | `output/drumloops_cleaned/<hex>/<md5>.mid` | Stage1 で dedupe 済み。
| メタデータ | `output/drumloops_metadata/drumloops_metadata_v2*.pickle` | shard + index。
| Stage1 サマリ | `output/drumloops_metadata/stage1_summary.json` | 除外統計や設定値を Stage2 summary に伝播する。

プレ処理ステップ:
1. `scripts/normalize_midi_meta.py` を `output/drumloops_cleaned` 全体に適用し、テンポ/拍子をトラック 0 へ統合する (set_tempo 欠損率も集計)。
2. TMIDIX で MIDI を再パースし、リズム指紋など Stage1 に無かった特性を計算する。
3. シャードサイズは 1,000–2,000 ループを上限とし、Stage2 からは複数ファイルに分割する。

## 3. 出力アーティファクト

| ファイル | 形式 | 主目的 |
| -------- | ---- | ------ |
| `output/drumloops_stage2/canonical_events.parquet` | Parquet (列指向) | 全ノートイベント正規化テーブル。
| `output/drumloops_stage2/canonical_events_sample.csv` | CSV (オプション) | デバッグ用に先頭 n 行。
| `output/drumloops_stage2/loop_summary.csv` | CSV | ループ粒度メタデータと指標集約。
| `output/drumloops_stage2/metrics_score.jsonl` | JSONL | 5軸スコア結果と詳細 breakdown。
| `output/drumloops_stage2/retries/<loop_id>.json` | JSON | 閾値未達ループ向け再処理プリセット。
| `output/drumloops_stage2/stage2_summary.json` | JSON | 全体統計・除外理由・テンポ整流結果など。

## 4. スキーマ詳細

### 4.1 イベントテーブル (`canonical_events`)

| 列名 | 型 | 説明 |
| ---- | -- | ---- |
| `loop_id` | STRING | Stage1 `md5` を 16進小文字で保持。
| `source` | STRING | `"drumloops"` 固定 (将来 Suno や外部ソース追加時に活用)。
| `pipeline_version` | STRING | `"stage2"` バージョン番号 (例: `2025.10.0`)。
| `bar_index` | INT | 0 開始の小節番号 (テンポ/拍子から計算)。
| `beat_index` | INT | 小節内の拍番号 (0–拍子-1)。
| `beat_frac` | FLOAT | 拍内位置 (0–1).
| `grid_onset` | INT | 最小グリッド (ticks) に丸めたオンセット。
| `onset_ticks` | INT | 生オンセット値。
| `duration_ticks` | INT | ノート長 (tick)。
| `channel` | INT | 0–15。
| `program` | INT | 原ノートの program。
| `program_norm` | INT | `program % 128` (ドラム以外は 0–127)。
| `pitch` | INT | 0–127。
| `velocity` | INT | 0–127。
| `intensity` | FLOAT | velocity を 0–1 に正規化。
| `is_ghost` | BOOL | velocity ≤ ghost 閾値か。
| `swing_phase` | STRING | `"even"` / `"odd"` / `"triplet"` など。
| `microtiming_offset` | FLOAT | グリッドとの差 (ticks)。
| `ioi_bucket` | STRING | IOI (1/4, 1/8, 1/12, 1/16) ヒストグラム用バケット。
| `instrument_role` | STRING | kick / snare / closed_hat / open_hat / tom / cymbal / perc / other。
| `role_confidence` | FLOAT | 0–1。役割推定の確信度 (複合規則で算出)。
| `file_digest` | STRING | Stage1 由来の md5 ハッシュ（冗長だが join 用に保持）。

> `source` 列の値は `"drumloops"`, `"suno"`, `"external"` 等の限定集合から選択し、将来統合時のライセンス管理に備える。

### 4.2 ループサマリ (`loop_summary`)

| 列 | 型 | 説明 |
| --- | -- | ---- |
| `loop_id` | STRING | イベントと join。
| `filename`, `genre`, `bpm`, `note_count`, `duration_ticks` | 既存 Stage1 値。
| `bar_count` | INT | `duration_ticks / ticks_per_bar` を丸めて算出。
| `ticks_per_beat` | INT | MIDI header 情報。
| `tempo.initial_bpm` | FLOAT | 最初のテンポ (set_tempo 無ければ filename 推定値)。
| `tempo.variance` | FLOAT | bpm の標準偏差。
| `tempo.events` | JSON | set_tempo イベント配列。
| `time_signature` | STRING | 例: `"4/4"`。複数あれば先頭と変更点の配列。
| `program_set` | ARRAY<INT> | 参照したプログラム番号。
| `metrics.*` | FLOAT | `LoopMetrics` にある各指標 (swing_ratio, ghost_rate, layering_rate, ...)。
| `metrics.rhythm_fingerprint` | JSON | IOI ヒスト (1/4,1/8,1/12,1/16 グリッド別)。
| `metrics.drum_collision_rate` | FLOAT | kick/snare/hh の同時衝突率。
| `metrics.role_separation` | FLOAT | 役割ごとの分散指標。
| `metrics.microtiming_rms` | FLOAT | microtiming オフセットの RMS。
| `metrics.rhythm_hash` | STRING | Z-score 正規化後に複数グリッドを連結し SHA1 化したハッシュ。
| `tempo.lock_method` | STRING | `median` / `first` / `none`。テンポ整流の決定根拠。
| `tempo.grid_confidence` | FLOAT | 拍グリッドの信頼度 (0–1)。
| `score.total` | FLOAT | 0–100。
| `score.breakdown` | JSON | 各軸のスコア (Timing/Velocity/...)
| `score.threshold_passed` | BOOL | `score.total >= threshold`。
| `retry.preset_id` | STRING | 適用済みプリセット ID (再処理済みの場合)。
| `retry.seed` | INT | 自動再処理で使用した乱数 seed。
| `exclusion_reason` | STRING | Stage2 時点で除外された場合の理由。
| `pipeline_version` | STRING | Stage2 バージョン管理。
| `git_commit`, `data_digest` | STRING | トレーサビリティ用。

### 4.3 スコアレポート (`metrics_score.jsonl`)

JSONL 1 行構造:
```json
{
  "loop_id": "abcd...",
  "score": 78.5,
  "axes": {
    "timing": 16.0,
    "velocity": 19.5,
    "groove_harmony": 14.0,
    "drum_cohesion": 15.5,
    "structure": 13.5
  },
  "threshold": 70,
  "retry_preset_id": "timing_relax_quantize",
  "seed": 1337,
  "metrics_used": ["swing_ratio", "microtiming_std", ...],
  "created_at": "2025-10-07T11:32:00Z"
}
```json
{
  "loop_id": "abcd...",
  "score": 62.0,
  "reason": "timing_below_threshold",
  "metrics": {
    "swing_confidence": 0.2,
    "microtiming_std": 18.5,
    "drum_collision_rate": 0.34
  },
  "preset_id": "timing_relax_quantize",
  "seed": 233,
  "applied_at": "2025-10-07T12:02:11Z",
  "preset": {
    "add_swing": 0.08,
    "relax_quantize": true,
    "reharmonize": "omit_clashing"
  },
  "metrics_before": {
    "score": 62.0,
    "timing": 12.0
  },
  "metrics_after": {
    "score": 71.5,
    "timing": 16.5
  }
}
```
プリセットは 3–5 種固定 (例: `"over_quantized"`, `"bad_harmony"`, `"drum_collision"`)。

> `metrics_before` / `metrics_after` は少なくとも total score と原因軸 (例: timing) を記録し、適用差分を即時評価できるようにする。

### 4.5 Stage2 サマリ (`stage2_summary.json`)

```json
{
  "pipeline_version": "2025.10.0",
  "git_commit": "<hash>",
  "inputs": {
    "total_files": 1929,
    "normalized_tempi": 0.94,
    "tempo_missing": 118
  },
  "exclusions": {
    "size_too_big": 211,
    "too_few_notes": 87,
    "non_poly": 126,
    "duplicate_rhythm": 34
  },
  "score_distribution": {
    "population": "passed_loops",
    "min": 51.2,
    "p50": 76.4,
    "p90": 88.7
  },
  "retry_queue": 214,
  "created_at": "2025-10-07T11:40:00Z"
}
```

## 5. メトリクス拡張 (実装対象)

既存 `LoopMetrics` に以下を追加する:

- **Rhythm fingerprint**: IOI を 1/4, 1/8, 1/12, 1/16 グリッドに割り付けたヒストグラム (全体 / 先頭 1 小節)。
- **Microtiming RMS**: `microtiming_offset` の 2 乗平均平方根。
- **Drum collision rate**: kick/snare/hihat が 1–2 tick 以内で重なる割合。
- **Role separation**: 役割 (kick/snare/hihat/toms/etc) の同時発火パターンに基づく分散。
- **Tempo stability**: set_tempo の標準偏差 (bpm 換算)。

> 備考: Stage1 の Pitch-based dedupe で発生し得る偽陽性を減らすため、**リズム指紋ハッシュ** (IOI ヒストグラムを正規化し SHA1 化) を `loop_summary.metrics.rhythm_hash` として持たせる。

補足事項:

- Groove Harmony 軸はジャンル × BPM 帯別の期待レンジテーブルを参照し、未知ジャンル時は対称レンジを仮置き (情報欠損時は 10 点固定)。
- リズム指紋ハッシュは (1) Z-score 正規化 → (2) 1/8・1/12・1/16 の混合ベクトル連結 → (3) バージョン付き SHA1 (例: `rhythm_hash:v1:<sha1>` ) の手順で生成し、閾値変更でも再現性を確保する。

## 6. スコアリング (70 点閾値)

| 軸 | 重み | 代表指標 | 算出例 |
| --- | --- | -------- | ------ |
| Timing | 20 | `swing_confidence`, `microtiming_std`, `fill_density` | 正規化 / 平均。
| Velocity | 20 | `ghost_rate`, `accent_rate`, `velocity_range`, `unique_velocity_steps` | 貢献度合計で 0–20。
| Groove Harmony | 20 | `swing_ratio` の期待レンジ / `rhythm_fingerprint` 類似度 / (ドラムの場合は裏拍適合度) | NA の場合は 10 点基準。
| Drum Cohesion | 20 | `drum_collision_rate`, `role_separation`, `hat_transition_rate` | 衝突が多いと減点。
| Structure | 20 | `repeat_rate`, `variation_factor`, `breakpoint_count` | 4小節周期性をペナルティ化。

CLI 要件:
- `scripts/score_midi.py --in <midi|jsonl> --config configs/lamda/score.yaml --threshold 70 --out <json>`
- 複数ループを処理する際は `--batch` で loop_summary を読み込み、一括出力。
- スコアリング結果を `metrics_score.jsonl` と `loop_summary.score.*` の両方に書き込む。

## 7. CLI & Config

### 7.1 Stage 2 extractor

```
python scripts/lamda_stage2_extractor.py \
  --input-dir output/drumloops_cleaned \
  --metadata-dir output/drumloops_metadata \
  --output-dir output/drumloops_stage2 \
  --config configs/lamda/drums_stage2.yaml \
  --threshold 70 --limit 0
```

対応する YAML (`configs/lamda/drums_stage2.yaml`) は以下を定義:
- `events.parquet_path`, `loop_summary.csv_path`, `metrics_score.jsonl_path`
- メトリクス設定 (`ghost_velocity_threshold` など)
- スコアリング閾値と各軸のウェイト
- `retry_presets` のプリセット内容

### 7.2 追加ユーティリティ

- `python scripts/lamda_export_metadata.py --light --csv-out ...` (Stage2 移行後も軽量 CSV 出力に利用)
- `python scripts/analyze_drumloops_quality.py --sample -1 --report-path ...` (Stage2 で改善した指標を再確認)

### 7.3 役割推定ルール

1. GM 準拠トラック (channel 10) を優先的にドラム扱い。
2. 残りは `program_norm` を GM マッピング表で正規化し、Perc 系を抽出。
3. ピッチ集合 (kick=35/36、snare=38/40、hat=42/44/46 など) を閾値照合し役割スコアを算出。
4. 各ルールのスコアを 0–1 に正規化し、最大値の役割を `instrument_role`、スコアを `role_confidence` とする。
5. パーカッション等でスコア閾値未満の場合は `perc` or `other` へフォールバック。

### 7.4 スウィング / 三連・テンポ変化の扱い

- `tempo.events` が存在する場合は拍単位で再サンプリングし、IOI と `swing_ratio` を再計測。
- even / odd / triplet の 3 系統を同時計測し、最も適合度が高いラベルを `swing_phase` に設定。
- 三連適合度と microtiming RMS を併用し、`metrics.swing_confidence` を 0–1 で出力。
- BPM 変化が大きい場合は小節ごとに局所 BPM を推定し、Grid 信頼度 (`tempo.grid_confidence`) を計算。

## 8. 運用要件

- **除外理由ログ**: Stage1 で除外したケース (size, notes, polyphony, duplicates, rhythm_hash) を `stage1_summary.json` に追加し、Stage2 summary に引き継ぐ。
- **テンポ整流率**: set_tempo を持たない MIDI の割合を Stage2 summary に記録。median ロック処理の判定根拠とする。
- **リトライ方針**: 再処理は「自動でもう一手」を 1 回だけ実施 (プリセット適用 → 再スコア → 70 未満なら保留)。
- **リトライプリセット仕様**: `add_swing`, `relax_quantize`, `velocity_spread` 等のパラメータは YAML で min/max/step と適用順序を固定し、3–5 個のプリセットに限定。適用時は seed を固定し `retry_preset_id` と共に保存。
- **パフォーマンス**: 1,929 ループでイベント総数 ~20 万。Parquet 書き出しは 5 分以内を目安に `pyarrow` を使用。
- **テスト**: `pytest tests/test_stage2_metrics.py` を新設し、テンポ欠損・3/4 拍子・極端スウィング・ポリリズム・Perc 偏重・静音多め等の境界 MIDI を `tests/fixtures/stage2/` に配置して回帰テストを行う。

## 9. Stage 3 への橋渡し

- `loop_summary`, `metrics_score`, `rhythm_fingerprint` を軸に **クエリテンプレ (Top-K 100)** を構築し、Stage3 の変換学習データ抽出に利用する。
- `retry_queue` に蓄積されたケースから改善ワークフローをフィードバックし、Stage1 の Humanizer / Groove 前処理パラメータを調整する。
- `score_distribution` の時系列監視 (中央値, 分散) を CSV でロギングし、品質ドリフトを検知する。
- Parquet は `loop_id` 先頭 2 桁 × `bpm` 10 刻みバケットでパーティション分割し、将来のスケールと外部ソース統合に備える。

## 10. 完了指標 (Stage 2 Done)

- 品質分布: `score_distribution.p50 ≥ 75`、`p90 ≥ 88` を初回バッチの目安とする。
- 再処理効率: `retry_queue / total_loops ≤ 0.15`、自動リトライ後の中央値は +8 点以上。
- テンポ整流: `tempo_missing / total_loops ≤ 0.05`。
- 性能: 約 1.9k ループを 5 分以内で Parquet 書き出し (pyarrow、ローカル SSD 前提)。

---

以上が Stage 2 実装の仕様です。実装着手に合わせて本書を `docs/LAMDA_STAGE2_SPEC.md` として参照してください。
