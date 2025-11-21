# AI Components Inventory (2025-11-19)

Phase 0 のガードレール整備では、既存 AI 群の所在と即時可動状態を把握することが重要になる。以下では Composer4 リポジトリ内で継続利用されている主要 AI モジュールを「用途」「エントリーポイント」「現在の状態」で一覧化し、再利用や改修の足掛かりを提供する。

## Summary Table

| Component | Role / Scope | Key Entry Points | Status (2025-11) |
| --- | --- | --- | --- |
| Stage3 GPT-2 Generator | Phrase-to-plan モデル。LoRA で微調整し、Stage3 条件 CSV/Parquet を入力に生成する。 | `ml/stage3_generator.py`, `outputs/stage3/conditions_for_training.csv` | ✅ 小型モデルで動作確認済み。LoRA (peft) 有効、条件データ再生成済み。 |
| OtobonAI (Emotion / GuideTone) | Section 単位で emotion ベクトルや guide tone を算出し、各楽器プランに注入する。 | `emotion_ai_v2.py`, `guide_tone_ai_v2.py`, `plan_*_v2.py` | ⚠️ API は利用可だが Phase 2 統合途中。section ループへの接続を強化予定。 |
| DUV Humanize Stack | Velocity / Duration / Timing の補正モデル。DurationHumanizeAI の核となる。 | `scripts/train_phrase.py`, `scripts/predict_duv.py`, checkpoints under `checkpoints/` | ✅ 既存 ckpt (例: `bass_duv_v2.ckpt`) 利用可。再学習テンプレを docs/real_song_roadmap_v2.md に記載済み。 |
| DurationHumanizeAI (MVP) | DUV 出力と Humanize profile + EmotionAI を統合し JSON plan に `humanize` フィールドを付与。 | `duration_humanize_ai.py` (新規), `plan_humanize.yaml`, `json2midi.py` hook | 🚧 設計済み (docs/real_song_roadmap_v2.md)。実装は Phase 1/3 の最優先タスク。 |
| RhythmAI + Pattern Matcher | Rhythm vocab をもとにトップ K マッチングを行い、Drums/Continue へフィードする。 | `scripts/pattern_matcher.py`, `output/rhythm_ai/rhythm_patterns.pickle`, `rhythm_vocab.yaml` | ⚠️ 既存辞書あり。Phase 3 で vocab 更新と RhythmAI API 刷新予定。 |
| Suno Arrangement AI | Drum/Bass/Guitar/Piano/Strings を Stage2 Groove ベースで再構築する E2E ランナー。 | `scripts/e2e_suno_arrangement.sh`, `scripts/instrument_midi_to_plan_real.py`, `scripts/recommend_drums.py` | ✅ Artifact-only モードで Phase A 産物を再利用。Magenta Groove, Emotion/Harmony AI, KPI Gate 連携済み。 |
| Magenta Decorators | GrooVAE/Continue/Interpolate/Generate による groove/fill 補強層。 | `ops/magenta_groove.py`, `scripts/magenta_slot_merge.py`, CLI `magenta-groovae-generate` | ⚠️ GrooVAE は安定。その他機能は Humanize/Groove 安定後に段階導入。 |
| KPI / CI Gates | KPI Gate、Quality Gate、CI Verify を束ねる検証スクリプト群。 | `scripts/kpi_gate_enhanced.py`, `scripts/quality_gate_checker.py`, `scripts/ci_quality_gate.sh`, `ops/ci_verify_music_package.py` | ✅ 既存スクリプト稼働。Phase 0 で composer4 仕様 (tempo_map, bars_with_slots, plan5種) チェックを拡張中。 |

## Component Notes

### Stage3 GPT-2 Generator
- **目的**: XMIDI 条件をもとに各種テクニック (articulation, register, voicing 等) を付与したイベント列を出力。
- **依存**: `outputs/stage3/conditions.parquet`, `conditions_for_training.csv`, technique metadata (`technique_metadata_v2.csv` 他)。
- **現状**: `.venv311` に `torch`, `transformers`, `accelerate`, `peft` を導入済み。LoRA なしのスモーク確認済みで、LoRA 付き中規模モデルの再訓練が Phase 1.2 のタスク。

### OtobonAI (Emotion / GuideTone)
- **目的**: Section ごとのバランス (valence, arousal) とガイドトーンを決め、Instrument プラン作成時の tension/voice-leading に反映。
- **代表ファイル**: `emotion_ai_v2.py`, `guide_tone_ai_v2.py`, `otobon_ai/` サブパッケージ。
- **ToDo**: `instrument_midi_to_plan_real.py` の section loop で Emotion/GuideTone を一貫適用する hook 整備。

### DUV Humanize Stack & DurationHumanizeAI
- **目的**: DUV で得た `{delta_start, duration_scale, velocity}` を JSON plan に保存し、`json2midi.py` で反映。
- **チェックポイント**: `checkpoints/bass_duv_v2.ckpt` など。`docs/real_song_roadmap_v2.md` に再学習テンプレを記載。
- **MVP**: 新規 `duration_humanize_ai.py` に `HumanizeRequest` dataclass を定義し、`scripts/predict_duv.py` をラップして CSV→MIDI→Plan merge まで自動化予定。

### RhythmAI / Pattern Matcher
- **目的**: `matches_rhythm.json` を生成し、Drums/Continue/Interpolate の初期条件として利用。
- **現状**: ルールベース matcher + `output/rhythm_ai/rhythm_patterns.pickle` を再利用中。Phase 3 で辞書 `rhythm_vocab.yaml` の refresh と推論 API 化を実施。

### Suno Arrangement Runner
- **役割**: Phase A で得られた解析結果を用い、Drums → Instruments → Plan validation → MIDI → KPI Gate まで自動化。
- **強化点**: Artifact-only モード (WAV 参照なし) がデフォルト。Magenta GrooVAE、EmotionAI、HarmonyAI、KPI Auto-Repair、Groove Polish が組み込まれている。

### Magenta Decorators
- **役割**: Drum groove / fill, Continue, Interpolate, Generate のアイディア補強レイヤ。
- **注意**: Phase 4 の導入順に従い、まず GrooVAE + slot merge、続いて Continue / Interpolate、最後に Generate を限定導入する。

### KPI / CI Gates
- **役割**: Instrument 別 Quality Gate、KPI Gate、`ci_verify_music_package.py` による構造検証を一括実行。
- **Phase 0 作業**: `scripts/ci_quality_gate.sh` を composer4 仕様へ更新し、`tempo_map.json`, `bars_with_slots.parquet`, 5種の plan ファイルを必須化。

---
このドキュメントは Phase 0 の「リポジトリ同期 & ガードレール整備」における参照用インデックスであり、各 AI のオーナーや再訓練コマンドは `docs/real_song_roadmap_v2.md` および関連設計ドキュメントにリンクされている。必要に応じてコンポーネント追加時にこの表も更新すること。