# Real Song Roadmap v2 — Humanize-First Refresh (2025-11-19)

目的: Stage2/Stage3 で整った最新データと AI 群 (OtobonAI, DUV, Magenta, Stage3 GPT-2 等) を結集し、テンポ同期・Humanize・QA 合格を満たす 1 曲自動生成パイプラインを段階的に完成させる。

---

## 0. スナップショット (現状整理)

| 項目 | 状態 (2025-11-19) |
| --- | --- |
| Stage2 抽出 | `tempo_map.json`, `bars.parquet`, `sections.json`, `bars_with_slots.parquet` など再生成済み。ループ要約 CSV / Parquet も揃っており再利用可。 |
| Stage3 条件 | `outputs/stage3/conditions.parquet` と `conditions_for_training.csv` が XMIDI アライン済み。`stage3_generator.py` は LoRA 対応 (peft 導入済み)。 |
| DUV チェックポイント | `checkpoints/bass_duv_v2.ckpt` など既存モデルあり。`scripts/predict_duv.py` / `scripts/train_phrase.py` で推論・再学習可能。 |
| EmotionAI / GuideToneAI | OtobonAI 実装 (`emotion_ai_v2.py`, `guide_tone_ai_v2.py`) は存在。V2 plan への完全統合は道半ば。 |
| Magenta 系 | 仕組み (Groove / Drumify / Continue / Interpolate / Generate) の構想あり。コード雛形やランナーは部分的。 |
| QA / KPI | `quality_gate_fill_riff.py`, `kpi_gate_enhanced.py`, `deep_harmony_audit.py` 等はあるが、最新パイプラインへの自動組み込みは未完。 |

---

## 1. 直近 4 週間の優先テーマ

1. **DurationHumanizeAI MVP**: DUV を核に Humanize profile + EmotionAI を反映できる AI レイヤを先行実装。Magenta より前に人間味を担保する。
2. **Stage3 再訓練 & 評価**: `outputs/stage3/conditions_for_training.csv` + technique metadata を使い、小型で通ったコマンドを LoRA 付き中規模設定に拡張。`quick_eval` でデモ生成。
3. **Rhythm / Groove 基盤補強**: RhythmAI 辞書、Drumify、Groove (DUV + Humanize) を Phase1/2 のアウトプットへ組み込み、Continue/Interpolate/Generate はその後に着手。
4. **Magenta 装飾パイプ**: Humanize / Groove / Drumify が安定したら、Magenta は Fill / Riff 補助レイヤとして限定導入。

---

## 2. リフレッシュ後のフェーズ計画

### Phase 0 — リポジトリ同期 & ガードレール整備 (完了率 70%)
- ドキュメント: `OTOBON_AI_IMPLEMENTATION.md`, `OTOBON_AI_PHASE2_SUMMARY.md` を docs へ集約済み。残タスクは AI 一覧表 (`docs/AI_COMPONENTS.md`) の追加。
- スクリプト: `make_song_phase_a.sh` (tempo/bars再抽出)、`e2e_suno_arrangement.sh` リフレッシュが必要。
- CI: `ci_quality_gate.sh` を composer4 仕様に合わせる (tempo_map, bars_with_slots, plans 5 種の存在チェック)。

### Phase 1 — Humanize 土台と Slot 安定化 (進行中)
1. `plan_humanize.yaml` を楽器×セクションで再定義。
2. `json2midi.py` 直前に Humanize hook を挿入。初期版は DUV 無しの deterministic テーブル + 乱数。
3. Fill/Riff slot 充足率: 境界 fill ≥ 0.9, chorus riff ≥ 0.5 を KPI 化。
4. KPI Gate (Mute ゼロ, per-instrument density 下限, tension events) を Phase1 QA として固定。

### Phase 2 — OtobonAI (Emotion / GuideTone) 統合
1. EmotionAI / GuideToneAI API を section×bar ループに統合 (Strings/Piano → Guitar/Bass/Drums)。
2. LyricAnchorIndex (phrase roles) を rulebook に反映。
3. CREPE / Onsets-and-Frames は参照レイヤとして hook のみ配置。
  - 2025-11-20: `generate_piano_plan_v2.py` / `generate_strings_plan_v2.py` に `--piano-oaf` フラグと `reference_layers` メタデータを追加し、CREPE F0 / Onsets-and-Frames サマリを AI context へ共有可能にした。
  - 2025-11-20: `arrangement_orchestrator.py` と `scripts/midi_writer.py` が `reference_layers` サマリを集約・読取できるようになり、Humanize 手前の段階でも CREPE/OAF メタデータが参照可能になった。

### Phase 3 — RhythmAI / DurationHumanizeAI 完成 & Continue/Interpolate
1. Rhythm vocab (`rhythm_vocab.yaml`) + AI 推薦 (`rhythm_ai.py`) を V2 テンプレ決定フェーズへ挿入。
  - 2025-11-20: RhythmAI が `rhythm_vocab.yaml` のベース/BPM メタデータを読み込み、`generate_bass_plan_v2.py` が CLI から vocab/manifest を受け取って rhythm_pattern_id を各イベントへ付与できるようになった。
  - 2025-11-20: Bass 用語彙 (`bass.root_quarter` など) を vocab に追加し、policy の `instruments.bass.rhythm_vocab` でセクション別の推奨 ID/density を記述した。
  - 2025-11-20: Funk/Disco/Shuffle/Latin 4 種の bass フィール (`bass.funk_syncopated_ghost` など) を `rhythm_vocab.yaml` に追加し、他ジャンルの Continue/Interpolate 事前準備を完了。
2. **DurationHumanizeAI** を完成 (DUV + EmotionAI + profile)。
3. Continue: 1 小節モチーフを 8 小節以上へ伸ばす extend モジュール (RhythmAI + Stage3 条件を活用)。
  - 2025-11-21: `generate_guitar_plan_v2.py` が ContinueModule を直接呼び出し、policy セクション or CLI `--continue-enable` でギターリフを差し替え可能になった。`scripts/make_song_package_phase_b.sh` からも Continue フラグを伝播できる。
4. Interpolate: パターン A↔B の橋渡し生成を groove 生成器へ追加 (良質パターン確保後)。

### Phase 4 — Magenta & 装飾レイヤ + Generate (最後)
1. Magenta Fill/Arp レイヤ (`magenta_fill_generator.py` 想定) を slot に限定適用。
2. Diversity コントロール (magenta event 数、使用確率) を policy YAML に追記。
3. Generate 機能は Groove/Drumify/Continue/Interpolate が安定後に補助的に使う。

### Phase 5 — QA/CI 拡張 & ワンボタン化
1. `quality_gate_fill_riff.py` と `kpi_gate_enhanced.py` を統合し JSON KPI を出力。
2. `make_song.sh` (Phase A→B→C) を strict/no-ai/ai-only フラグ付きで実装。
3. `song_004` をゴールデンマスターとして diff 回帰テストを自動化。

---

## 3. DurationHumanizeAI MVP 設計

### 3.1 データ & 依存
- **入力ソース**: `plans/*/*.json` (note events), EmotionAI 出力, `plan_humanize.yaml` プロファイル, optional CREPE hints。
- **学習データ**: Stage2 または Los Angeles MIDI から抽出した note CSV (`notes.duv.csv` 等)。`scripts/train_phrase.py` が出力。
- **モデル**: DUV (velocity/duration) multi-head ネット。`--duv-mode both`, `--use-duv-embed` を推奨。
- **推論**: `scripts/predict_duv.py` で CSV→MIDI を変換し、Humanize hook で JSON プランへ戻す。

### 3.2 処理フロー (MVP)
1. `generate_*_plan_v2.py` が各イベントを出力。
2. `plan_humanize.yaml` から instrument/section ごとの初期 sway, swing, velocity curve を取得。
3. EmotionAI の valence/arousal を feature に組み込み、DUV 推論へ渡す。
4. DUV から得た `{delta_start, duration_scale, velocity}` を JSON plan の `humanize` フィールドとして保存。
5. `json2midi.py` で `humanize` を適用し最終 MIDI を出力。

### 3.3 DUV 推論ラッパーテンプレ (Python)

```python
# duration_humanize_ai.py (template)
from __future__ import annotations
from dataclasses import dataclass
import argparse
from pathlib import Path

import pandas as pd

from scripts import predict_duv


@dataclass
class HumanizeRequest:
    csv_path: Path
    ckpt: Path
    stats: Path
    out_midi: Path
    vel_smooth: int = 3
    dur_quant: str = "1/32"
    device: str = "cpu"

    def to_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(
            csv=self.csv_path,
            ckpt=self.ckpt,
            stats_json=self.stats,
            out=self.out_midi,
            vel_smooth=self.vel_smooth,
            dur_quant=self.dur_quant,
            device=self.device,
            smooth_pred_only=True,
            batch=64,
            filter_program=None,
            limit=0,
            verbose=False,
        )


def run_humanize(req: HumanizeRequest) -> Path:
    predict_duv.run(req.to_namespace())
    return req.out_midi


def attach_humanize_to_plan(plan_path: Path, humanize_csv: Path, midi_path: Path) -> None:
    plan = pd.read_json(plan_path)
    humanize_df = pd.read_csv(humanize_csv)
    merged = plan.merge(humanize_df[["event_id", "start_delta", "duration_scale", "velocity_delta"]], on="event_id", how="left")
    merged.to_json(plan_path, orient="records", force_ascii=False, indent=2)
```

> 実際の schema に合わせて `event_id` やフィールド名を調整する。`predict_duv.run` は CLI と同じロジックのため、`subprocess.run(["python", "scripts/predict_duv.py", ...])` で代用しても良い。

### 3.4 DUV 再学習コマンドテンプレ

```bash
# 1) CSV 準備 (例: Los Angeles MIDI 500 曲)
PYTHONPATH=. .venv311/bin/python scripts/prepare_phrase_csv.py \
  --midi-root data/Los-Angeles-MIDI \
  --out data/phrase_csv/bass_duv.csv \
  --max-bars 8 --step-bars 2 --ppq 480 --tempo 90

# 2) DUV 学習
PYTHONPATH=. .venv311/bin/python scripts/train_phrase.py \
  --csv data/phrase_csv/bass_duv.csv \
  --duv-mode both --use-duv-embed --use-bar-beat \
  --vel-bins 16 --dur-bins 16 \
  --epochs 4 --batch-size 128 --lr 3e-4 \
  --arch lstm --hidden 512 --layers 4 \
  --out checkpoints/bass_duv_v3.ckpt \
  --stats-json checkpoints/bass_duv_v3.stats.json \
  --save-best

# 3) 評価 / 可視化
PYTHONPATH=. .venv311/bin/python scripts/eval_duv.py \
  --csv data/phrase_csv/bass_duv_valid.csv \
  --ckpt checkpoints/bass_duv_v3.ckpt \
  --stats-json checkpoints/bass_duv_v3.stats.json \
  --out reports/bass_duv_v3_metrics.json
```

### 3.5 Magenta 連携コマンド (Humanize 適用後)

```bash
# 1) Humanize 済み MIDI を Magenta GrooVAE へ供給
magenta-groovae-generate \
  --config=groovae_2bar_additive \
  --checkpoint=models/groovae_additive.ckpt \
  --input_midi=outputs/humanized_loop.mid \
  --output_dir=magenta_out/fills --num_outputs=8

# 2) 生成結果を slot にバインドする補助スクリプト
PYTHONPATH=. .venv311/bin/python scripts/magenta_slot_merge.py \
  --fills-dir magenta_out/fills \
  --slots parquet/bars_with_slots.parquet \
  --out plans/magenta_fills.json

# 3) Magenta + DurationHumanizeAI を一括適用
PYTHONPATH=. .venv311/bin/python scripts/apply_humanize_and_magenta.py \
  --plan plans/drums_plan.json \
  --humanize-midi outputs/humanized_loop.mid \
  --magenta-fills plans/magenta_fills.json \
  --out plans/drums_plan_with_magenta.json
```

> Magenta CLI (`magenta-groovae-generate`) は `pip install magenta` 済みを想定。既存環境の GPU/Mac 対応状況に応じて config/ckpt を差し替える。

---

## 4. Magenta 機能 (Groove/Drumify/Continue/Interpolate/Generate) の現状ポジショニング

| 機能 | 状況 | 推奨タイミング |
| --- | --- | --- |
| Groove | DUV + Humanize + plan_humanize でまず実現。Magenta 加工はその後に薄く乗せる。 |
| Drumify | RhythmAI + slot 情報で既に骨格あり。Magenta は fills/riffs の補強程度。 |
| Continue | RhythmAI / Stage3 の語彙が揃った後 (Phase 3)。通常 1 小節パターン → 8 小節へ拡張。 |
| Interpolate | 良質な A/B パターンが揃ってから (Phase 3 後半)。サビ前ブリッジ等に使用。 |
| Generate | Groove / Drumify / Continue / Interpolate が安定後 (Phase 4)。アイデア出し用途に限定。 |

---

## 5. 次のアクション (2025-11 版)

1. **DurationHumanizeAI**
   - `duration_humanize_ai.py` プロトタイプ作成。
   - `plan_humanize.yaml` の再定義 + JSON plan hook 実装。
   - DUV 再学習 (`bass_duv_v3.ckpt`) → `scripts/predict_duv.py` 包装。
2. **Stage3**
   - LoRA 付き `ml/stage3_generator.py` を `--n-layer 8 --n-head 8 --n-embd 512 --max-length 1792` で再走。
   - `quick_eval` による出力チェック + caption 欠如時の fallback 仕様を整理。
3. **Rhythm & Continue 準備**
   - `rhythm_vocab.yaml` v2 を作成 (genre×feel×density 軸)。
   - Continue モジュール設計書を `docs/continue_module.md` として追加。
4. **Magenta 準備**
   - Magenta CLI 動作確認 (`groovae_2bar_additive`)。
   - `scripts/magenta_slot_merge.py` の叩き台を用意し slot 接続方法を固める。
5. **ドキュメント & QA**
   - 本ロードマップを README から参照するリンクを追加。
   - Phase1 KPIs を `ci_quality_gate.sh` に組み込み。

これらを順番に進めることで、人間味 (DurationHumanizeAI) を基礎に据えつつ、Stage3/ Magenta / RhythmAI を統合した連続的なワークフローへ移行できる。必要に応じて本ドキュメントに進捗を追記して更新していくこと。
