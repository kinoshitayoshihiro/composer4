#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# Piano A/B Train & Eval Pipeline
# A=template / B=transformer の完全自動実行
#
# Usage:
#   bash scripts/run_piano_ab_train_eval.sh [MIDI_DIR]

set -euo pipefail

MIDI_DIR=${1:-"output/piano_cleaned"}
SPLITS="data/piano_splits"
MODEL_OUT="models/piano_transformer_v1/best"
GEN_A="output/pianogen_A"
GEN_B="output/pianogen_B"
REPORT_DIR="output/reports"

echo "========================================="
echo "Piano A/B Training & Evaluation Pipeline"
echo "========================================="
echo "MIDI Source: $MIDI_DIR"
echo "Engine A: template"
echo "Engine B: transformer (to be trained)"
echo ""

# 1) Data Preparation
echo "[1/5] Preparing training data..."
python scripts/piano_train_prepare.py \
  --midi-dir "$MIDI_DIR" \
  --out-dir "$SPLITS" \
  --seed 1234 \
  --val-ratio 0.05 \
  --test-ratio 0.05 \
  --max-bars 64 \
  --min-length 32

# 2) Training
echo "[2/5] Training Transformer model..."
python scripts/piano_train.py \
  --splits-dir "$SPLITS" \
  --config-yaml configs/piano_transformer.yaml \
  --out-dir models/piano_transformer_v1

# 3) A/B Generation
echo "[3/5] Generating A/B samples..."
python scripts/gen_ab_stratified.py \
  --instrument piano \
  --styles block,arpeggio \
  --densities mid \
  --tempos 110,130 \
  --length-bars 8 \
  --n-per-stratum 2 \
  --A-humanize true \
  --B-humanize true \
  --A-engine template \
  --B-engine transformer \
  --B-model-dir "$MODEL_OUT" \
  --out-root output

# 4) Evaluation
echo "[4/5] Evaluating samples..."
mkdir -p "$REPORT_DIR"
python scripts/eval_drum_batch_stratified.py \
  --instrument piano \
  --dir-A "$GEN_A" \
  --dir-B "$GEN_B" \
  --out-json "$REPORT_DIR/ab_piano.json" \
  --out-csv  "$REPORT_DIR/ab_piano_files.csv"

# 5) Report with acceptance gate
echo "[5/5] Generating report..."
python scripts/ab_report_rich.py \
  --ab-json "$REPORT_DIR/ab_piano.json" \
  --out-md  "$REPORT_DIR/ab_piano_report.md" \
  --plot-dir "$REPORT_DIR/plots_piano" \
  --instrument piano \
  --strict-exit || true

echo ""
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo "Results:"
echo "  - Report: $REPORT_DIR/ab_piano_report.md"
echo "  - Model: $MODEL_OUT"
echo "  - Model Card: $MODEL_OUT/model_card.json"
echo ""
echo "Next steps:"
echo "  1. Review $REPORT_DIR/ab_piano_report.md for A/B differences"
echo "  2. If B (transformer) improves metrics, promote to production"
echo "  3. Update Nightly CI to use new model"
