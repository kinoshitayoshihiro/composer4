# LAMDA Stage2 Articulation Metrics

## 指標定義と実装メモ

| 指標 | 定義 | 実装ポイント |
| --- | --- | --- |
| `articulation.snare_ghost_rate` | スネアノートのうち `velocity <= V_ghost` の割合。 | 既定 `V_ghost = 34`（推奨範囲 28–36）。対象は channel=9 または GM スネア番号。Stage2 extractor で算出済み。 |
| `articulation.snare_flam_rate` | 各スネアノートの直前に同一ピッチで `IOI <= T_flam_ms` を持つ二連の割合。 | `T_flam_ms` デフォルト 22ms（12–35ms の範囲を想定）。テンポ依存のため ticks→ms 変換時は `ticks_per_beat` を使用。 |
| `articulation.detache_ratio` | 弦パートでレガート結合しない短い分離発音の割合。 | ギャップ `gap_ms >= 15` かつ `gate <= 0.75` を detache 候補とする。`VIOLIN_PITCH_RANGE` を使用し channel=9 以外を対象。 |
| `articulation.pizzicato_ratio` | 弦パートで pizzicato と推定される発音の割合。 | keyswitch / CC / プログラムで明示された pizzicato を優先。Fallback は `duration <= 0.65 * reference_step`。総発音が閾値未満ならラベル付与しない。 |

既存 `lamda_stage2_extractor.py` では上記指標を計算し、`metrics` に書き出しています。推奨閾値は `configs/thresholds/articulation.yaml` を参照してください。

## 固定閾値（MVP）

固定モードでは以下を初期値として使用します（GM キット・一般的テンポ帯を想定）。

- ゴースト: `snare_ghost_rate >= 0.22` かつ `backbeat_strength >= 6`。安全弁: スネア総発音 ≥ 8。
- フラム: `snare_flam_rate >= 0.035`。安全弁: スネア総発音 ≥ 24。
- ドラッグ: `drag_triplet_like >= 0.015`。安全弁: スネア総発音 ≥ 24。
- ロール: `snare_notes_per_sec >= 9.5` で 240ms 以上継続。安全弁: スネア総発音 ≥ 24。
- ハイハット開き: `hat_cc4_mean >= 40` または `open_hat_ratio >= 0.18`。安全弁: ハイハット総発音 ≥ 32。
- 弦 detache: `detache_ratio >= 0.55`。安全弁: 弦トラックかつ総発音 ≥ 16。
- 弦 pizzicato: `pizzicato_ratio >= 0.12` または keyswitch/CC で確定。安全弁: 弦トラックかつ総発音 ≥ 16。

## 自動キャリブレーション

1. ループをテンポ帯 `tempo_bin = [0, 90, 110, 130, 999]` で分類。
2. 各 bin で対象指標の `Q1, Q2, Q3` と `IQR` を計算。
3. ラベル付与条件：
   - 高確: `metric >= Q3 + 0.25 * IQR`
   - 中立: `Q3 <= metric < Q3 + 0.25 * IQR`
   - 否決: `metric < Q2`
4. ヒステリシス: 付与後に取り消す場合は `metric < Q2 - 0.1 * IQR` を要求。
5. `min_support`: ドラム系 ≥ 24 発音、弦系 ≥ 16 発音（2 小節相当）で集計。

`configs/thresholds/articulation.yaml` の `mode: auto` と `quantiles` 設定を参照し、分位点計算の実装を追加してください。

## スコアリング統合

`Stage2` の総合スコアに奏法適合度を追加して Selector を拡張します。

```python
articulation_score = (
    0.15 * ghost_presence +
    0.10 * flam_presence +
    0.15 * detache_presence +
    0.10 * pizz_presence
)
```

- `presence` は 0/1 でも連続値（クリップ後）でもよい。
- 学習採用条件の例: `base_score >= 70` かつ `articulation_score >= 0.20`。
- 要求された奏法が満たされるかをチェックし、欠落時は罰則を加えると制御しやすい。

## 検証プロトコル（目安 4 時間）

1. ランダムサンプリング: ドラム 35 件、弦 15 件で合計 50 ループ。
2. 付与ラベルを波形 + MIDI で目視確認し、Precision ≥ 0.85 を目標に高閾値のみ調整。
3. 生成器に条件（例: ghost + accent）を付与し、`technique_match@top1` を記録。
4. A/B 比較: 既存フロー vs 奏法条件指定で Groove / 主観スコアを評価。
5. 結果を `docs/eval/` などに記録して閾値改善ループを回す。

## 今後のタスク候補

- Tempo bin × 曲種別での自動閾値更新処理の追加。
- `Stage2` retry ロジックに奏法指標を組み込んだプリセットを用意。
- Pizzicato 判定のための keyswitch / CC 収集スクリプト作成。
