# Changelog

## [Unreleased]
### Added
- **Phase 29-32 Usability Enhancement**: Practical optimization for production workflow (2025-10-19)
  - **Phase 29 Vocal-Aware Ducking**: Automatic velocity/duration reduction during vocal-dense moments
    - emotion_curve-driven Velocity reduction (max 3dB equivalent)
    - Duration shortening (max 20ms) for vocal clarity
    - Applied to Piano/Guitar/Strings (NO-OP default)
  - **Phase 32 Export Markers**: MIDI meta markers for DAW/VOCALOID/SynthV integration
    - Section markers (INTRO/VERSE/CHORUS) for quick navigation
    - Optional lyric markers for phoneme alignment
    - Embedded in part.comment for exporter processing
  - Phase 30/31 proposals documented (Cross-Instrument Balance, Voice-Leading Guard)
  - Implementation report: `PHASE_29_32_IMPLEMENTATION.md`
- **Phase 25-28 Post-Processing**: Advanced final-stage optimization (2025-10-19)
  - **Phase 25 Sparsify**: Note over-density reduction with endpoint preservation and min-gap control
  - **Phase 26 Hybrid Harmony**: Audio chordmap × creative chordmap blending with tension injection
  - **Phase 27 Style Adaptation**: Activity-driven preset interpolation (simple↔moderate↔complex↔intense)
  - **Phase 28 Export Postprocess**: Quantization, track splitting (RH/LH, Clean/FX), unified naming
  - Drums-specific HH over-density prevention (min_gap_ms: 18-30ms)
  - All instruments support Phase 25/27/28; Piano/Guitar/Strings add Phase 26
  - YAML presets updated for all styles (simple/moderate/complex/intense)
  - Comprehensive test suite `test_phase_25_28.py` (6/6 passing)
  - Regression test suite `test_phase_25_28_regression.py` (10/11 passing)
  - Full implementation report: `PHASE_25_28_IMPLEMENTATION.md`
  - Final validation report: `PHASE_25_28_FINAL_VALIDATION.md`
- **Phase 24/28 Validation Tests**: Control integrity and export meta verification (2025-10-19)
  - `test_export_split_and_controls.py`: RH/LH split meta & RPN/PB/CC11 integrity (6/6 passing)
  - `test_export_split_internal_threshold_optional.py`: Future internal split detection (1 skip)
  - RPN emission guard (max 1 per track), PB ±8191 range, CC11 0-127 range validation
- **Phase 4.3 External Benchmark Polish**: Schema versioning (1.1), fileset hash, threshold flags, provenance propagation to history, and PNG threshold visualization for long-term operational excellence (100% backward compatible)
- Lightweight module stubs for tests in `tests/_stubs.py`
- Instrument filtering for duration CSV via `--instrument` flag
- Phrase training visualizations (PR curve & confusion matrix), tag-wise metrics,
  deterministic and scheduler flags, weighted sampling, and optional DUV embeddings
- Headless-safe `--viz` plots, robust tag-aware evaluation, CSV tag filters and run
  metadata (git commit, env, sampler stats)
- `--strict-tags` option, transformer hyper-parameters/seed flags, CSV bucket
  emission, and sampler weight logging
- Standardize `duration_bucket`/`velocity_bucket` columns (legacy names warn),
  track CSV filtering stats, and record visualization usage in run metadata
- Reproducibility flags for device selection, deterministic execution, and
  strict tag validation against `tag_vocab.json`
- Split `--dur-decode`/`--vel-decode` options, separate velocity/duration modes
  for sampling
- `--best-metric` to select best checkpoint by macro F1 or tag/instrument F1
 - Visualization filenames `run-<timestamp>-epoch-<n>-*.png` and paths recorded in run metadata
 - Sparkle converter: CLI flags for section LFO, stable guard, vocal adapt, style injection, damping spec, and enriched debug reports
 - Fixed CSV column order with always-present `velocity_bucket` and
  `duration_bucket` (missing filled with -1)
- Temperature schedule for sampling via `--temperature-start/--temperature-end`
- Duration clamping via `--dur-max-beats` and recorded temperature schedule metadata
- Optional pitch-loss label smoothing (`--pitch-smoothing`) and per-loss CSV metrics
- `tools.corpus_to_phrase_csv --hash-split` for order-independent splits and new
  `guitar_low`/`guitar_lead` pitch presets
- `tools.corpus_to_phrase_csv --dry-run` smoke testing
- Strict tag workflow documentation
- Section presets and vocal-aware guidance for Sparkle converter
- Debug markdown output for per-bar tracing
- Harmony-aware phrase weighting, section pool weight overrides, smart style fills with gaps, and vocal ducking control
- Resumable Slakh2100 downloader staging via `--download-to`, download retry options, and refreshed MIDI extraction docs
### Fixed
- Harmonize DUV bucket column names and apply transformer nhead/layer/dropout
  flags while avoiding invalid LSTM kwargs
- Pitch targets now use raw MIDI values, avoiding head size mismatch
- PrettyMIDI tempo initialization fallback for older versions
### Changed
- ⚠️ Breaking change note removed – parameters are now optional
- Unified to **numba>=0.60.0** across requirements
- `scripts/train_duv_improved.py` gains optional AMP defaults, cosine warmup,
  EMA, and length-aware bucketing with expanded CLI flags
- Stage2 extractor aligns 5軸スコア名称 (Groove Harmony/Drum Cohesion) と
  正規化値を `axes_raw` として出力し、再処理キュー診断を強化
- `loop_summary` と `stage2_summary` に `git_commit` / `data_digest` /
  `score_axes` メタデータを追加し、契約ベースの品質検証を自動化

## [0.6.1] - 2025-07-25
### Fixed
- handle missing `pad_mask` in `decode_batch`
- error when no loops found in `_load_loops`

## [0.1.0] - 2025-07-21
### Added
- Initial dataset builder and CLI
- Unit tests and CI configuration

## [3.0.0] - 2025-07-15

### Added
- Modular plugin architecture
- Percussion sampler and groove utilities
- Style and auxiliary tag conditioning
- WebSocket bridge for realtime generation
- フェーズ0: 基盤機能とCLI整理
- フェーズ2: PercGenerator 試作
- フェーズ3: Style/Auxタグ対応
- フェーズ4: WebSocket ブリッジ
- フェーズ5: GrooveSamplerロードマップ完遂

### Changed
- Unified generator APIs and configuration loading
- Updated documentation and examples
- フェーズ1: ジェネレーターAPI統合
- フェーズ4: Hydra設定への移行

### Fixed
- Assorted stability fixes across generators and tests
- フェーズ移行時の互換性バグを修正
## [1.0.0] - 2025-07-22

### Added
- Breath Control module v1.0 with keep / attenuate / remove modes.
- ONNX inference option & energy_percentile configurability.

### Fixed
- Pop artefacts on micro breath segments.
