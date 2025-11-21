# Otobpn Repository AI System

このメモは composer4 レポジトリ内で進めている AI アレンジ制作の流れ・成果物・今後の展望を 1 つの資料に整理したものです。既存ドキュメントの参照先を明記し、各 AI モジュールの接続状況や優先タスクを俯瞰できるようにしています。

---

## 1. 目的と適用範囲
- **楽曲ターゲット**: J-POP / ポップス (主に Suno stems 由来)。
- **最終アウトプット**: `arrangement_plan.json` → `song_xxx_integrated.mid` → DAW での音色差し替え。
- **統合すべきレイヤ**: HarmonyAI / RhythmAI / OtobonAI (Emotion & GuideTone) / CREPE & OaF features / DurationHumanizeAI / Magenta 装飾。
- **標準パイプライン**: Phase A (解析) → Phase B (plan 生成) → Phase C (アレンジ統合 & MIDI書き出し)。詳細は `docs/real_song_roadmap_v2.md` を参照。

---

## 2. パイプライン現況スナップショット
| ステップ | 主ファイル/データ | 現状 | 補足 |
| --- | --- | --- | --- |
| Phase A | `make_song_package_from_sources.sh` / `tempo_map.json` / `bars_with_slots.parquet` | Song_004 は再生成済み。slot coverage 0.9 以上を維持。 | `docs/real_song_roadmap_v2.md` §2 |
| Phase B | `generate_*_plan_v2.py` (各楽器) | Strings/Piano で Emotion/GuideTone hooks が未完全。 | `docs/OTOBON_AI_PHASE2_SUMMARY.md` |
| Phase C | `arrangement_orchestrator.py` → `json2midi.py` | 5 トラック / 2,192 events の統合 MIDI 出力が安定。 | `scripts/make_song_package_phase_c.sh` |
| QA/Gate | `kpi_gate_enhanced.py` / `quality_gate_fill_riff.py` | drums 密度・fill slot 使用率は網羅済み。長音比率は今後追加予定。 | `docs/DRUMS_PRODUCTION_READY.md` |

---

## 3. 主要 AI モジュールと接続状態
### 3.1 OtobonAI ファミリー
- **LyricAnchorIndex** (`otobonAI/lyric_index.py`): `lyric_anchors.json` を bar 単位で索引。phrase_role/stress を返す。
- **EmotionAI v2** (`otobonAI/emotion_ai_v2.py`): Rulebook + lyric context から `density_scale`, `velocity_scale`, `energy`, `tension` を提供。
- **GuideToneAI v2** (`otobonAI/guide_tone_ai_v2.py`): `notes_per_bar`, `preferred_degrees`, `phrase_shape` を決定し、strings/piano のカウンターメロ指針になる。
- **Rulebook** (`configs/otobonAI/rulebook.yaml`): SoundQuest/うちやま由来の和声・ガイドトーン・リズムルールを保持。LYRIC系や Emotion 系ルールも Phase 2 で追加済み。
- **統合状況**: Strings/Piano の V2 ジェネレータに context を渡す hooks は `docs/OTOBON_AI_PHASE2_SUMMARY.md` の TODO を適用すれば完成。Guitar/Bass は Emotion パラメータのみを参照予定。

### 3.2 Rhythm/Drums Stack
- **ルール/ライブラリ**: `data/drum_patterns.yml`, `data/rhythm_library.yml`, `data/riff_library.yaml` に Kick/Snare/Hihat パターン、ピアノ/ベース/ギター用の rhythm 語彙、ジャンル別リフが定義済み。
- **RhythmAI (Drums)**: `docs/DRUMS_PRODUCTION_READY.md` に Phase 25 実装記録。XGBoost + Logistic Regression により pattern 推薦、5 KPI ゲートと Auto-Recovery を備える。
- **今後の rhythm_vocab**: 既存 YAML 群を統合するインデックス層として `rhythm_vocab.yaml` を追加予定。内容自体は新規というより「drum_patterns/rhythm_library/riff_library を 1 つの schema で参照する manifest」として設計するのが推奨。
- **外部モデル**: Magenta 由来 fill/arpeggio 生成 (`docs/MAGENTA_INTEGRATION_PATCHES.md`) は Phase 4 で slots に限定して接続予定。

### 3.3 Harmony & Guide Systems
- **Chordmap**: `manual_chordmap_locked.json` を基準に HarmonyAI (将来) が候補提示、最終決定は手動 + rule-based (`docs/CHORDMAP_WORKFLOW.md`)。
- **Counter Melody**: `docs/COUNTER_MELODY_SPEC.md` で strings をカウンターメロディ扱いする方針を定義。GuideToneAI から得た `phrase_shape` を strings で優先、一方 piano は CREPE由来メロラインと chordpad のハイブリッドにする役割分担が想定されている。

### 3.4 Humanize & Duration Layer
- **Humanize Config**: `plan_humanize.yaml` が作成予定。`docs/HUMANIZE_ADVANCED_FEATURES.md` / `docs/EMOTION_HUMANIZE_USAGE.md` に Phase 3.5 のガイドあり。
- **Engine**: `HumanizeConfig` / `apply_*_humanize` (in `scripts/v2_common.py`) を再接続することで、strings の長音や piano の短音をセクション特性に応じて変調可能。

### 3.5 External Feature Extraction
- **CREPE / Onsets-and-Frames**: `analysis/crepe_f0.parquet` などにボーカル F0/Onset を保存。piano 短音ラインはここからトレースしている。
- **Stems**: `stems_wav/` と `old/stem_midi/` を参照し、RhythmAI/Magenta で追加素材を抽出可能。

---

## 4. これまでの主な成果
| フェーズ | 実績 | 詳細ドキュメント |
| --- | --- | --- |
| Phase 22–24 | Humanize, QA, Slot自動挿入、Emotion連携を順次導入 | `docs/PHASE_22_24_23_IMPLEMENTATION.md`, `docs/HUMANIZE_ROLE_SECTION.md` |
| Phase 25 | Drums RhythmAI + KPI Gates + Safe-Kit + Canary ロールアウト | `docs/DRUMS_PRODUCTION_READY.md` |
| OtobonAI Phase 2 | Rulebook統合・LyricAnchorIndex・EmotionAI v2・GuideToneAI v2 完成 | `docs/OTOBON_AI_PHASE2_SUMMARY.md` |
| Suno Stem Integration | Vocal/Piano stems から melody hints / lyric anchors を抽出 | `docs/SUNO_STEM_ARRANGEMENT.md`, `docs/SUNO_STRUCTURE_EXTRACTOR_OUTPUT.md` |
| Real Song Roadmap v2 | 全 AI レイヤ統合のマイルストーン策定 | `docs/real_song_roadmap_v2.md` |

---

## 5. 主要アセットと配置場所
- `analysis/`: CREPE/OaF/lyric anchor など song-specific features。
- `configs/otobonAI/rulebook.yaml`: 感情・ガイドトーン・リズムルールの一元的カタログ。
- `data/*.yml`: drums/piano/bass/guitar の rhythm & riff ライブラリ。RhythmAI/RulebookEngine から参照。
- `docs/`: 各 Phase レポートや AI 実装ガイド。特に Real Song Roadmap, OtobonAI summary, Drums production ready が基幹。
- `scripts/`: Phase C / humanize / orchestration ロジック (`json2midi.py`, `v2_common.py`, `arrangement_orchestrator.py`)。

---

## 6. 課題とチャンス
1. **Strings vs Piano の役割衝突**: Strings をカウンターメロに固定し、`source="melody_hint"` の短音は piano 側に集約するフィルタが必要。GuideToneAI の `phrase_shape` を strings で優先し、piano は CREPE トレース + chordpad で補完する。
2. **Emotion/GuideTone Hook 未適用**: V2 ジェネレータへ context を渡す実装を戻し、`density_scale` と `notes_per_bar` を bar 別に反映する。
3. **Humanize Config 不在**: `plan_humanize.yaml` を生成し、strings に `phrase_end_extend`, piano に `duration_scale_mean < 1` を適用して DAW 上のコントラストを確実にする。
4. **Rhythm vocab の統合**: 既存 YAML 群が分散しているため、`rhythm_vocab.yaml` を manifest として作り、RhythmAI / Rulebook / EmotionAI が同じ ID を参照できるようにする。
5. **QA 拡張**: 長音率や melody_hint 比率を KPI に加え、「strings 長音消失」などの退行を自動検出。

---

## 7. 今後のロードマップ (抜粋)
1. **Phase 1 (再整備)**
   - Tempo/Bars/Slots の再リラン (`make_song_phase_a.sh`)。
   - `plan_humanize.yaml` 作成と Humanize hook 再接続。
2. **Phase 2 (OtobonAI 完全統合)**
   - Strings/Piano の V2 ジェネレータに EmotionAI / GuideToneAI / LyricAnchor context を反映。
   - Rulebook に lyric 系 strings/piano ルールを追加し、phrase_start/end の動機を固定。
3. **Phase 3 (RhythmAI & DurationHumanizeAI)**
   - `rhythm_vocab.yaml` で drums/bass/guitar/piano パターンを統合管理し、RhythmAI 推薦と連携。
   - DurationHumanizeAI を `json2midi.py` 前に通し、セクション別マイクロタイミングを自動調整。
4. **Phase 4 (Magenta 装飾)**
   - Fill/riff slots へ限定的に Magenta 生成を注入し、多様性を確保。
5. **Phase 5 (QA & ワンボタン化)**
   - `kpi_gate_enhanced.py` に長音率等を追加。
   - `e2e_suno_arrangement.sh` で Phase A→C を一括実行。

---

## 8. 参考ドキュメント
- `docs/real_song_roadmap_v2.md`
- `docs/OTOBON_AI_PHASE2_SUMMARY.md`
- `docs/DRUMS_PRODUCTION_READY.md`
- `docs/COUNTER_MELODY_SPEC.md`
- `docs/HUMANIZE_ADVANCED_FEATURES.md`
- `docs/SUNO_STEM_ARRANGEMENT.md`

上記以外にも docs ディレクトリには Phase ごとの詳細レポートが揃っているため、必要に応じて該当 Phase 名で検索してください。
