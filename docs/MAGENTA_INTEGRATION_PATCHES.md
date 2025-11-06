# Magenta統合 推奨パッチ集

Phase A（Magenta統合）完了後に適用推奨の小パッチ集です。

## 1. stale再利用防止（実装済み）

**実装箇所**: `scripts/e2e_suno_arrangement.sh` Line 419-431

```bash
# 🧹 Purge old drum artifacts (if --force-regenerate-drums)
if [[ "$FORCE_REGENERATE_DRUMS" == "true" ]]; then
    echo "🧹 Force regenerate: purging cached drums artifacts"
    rm -f "$SONG_DIR/drums_plan.json" \
          "$SONG_DIR/drums_plan.log" \
          "$SONG_DIR/drums_seed.mid" \
          "$SONG_DIR/drums_grooved.mid" \
          "$SONG_DIR/drums_plan_seed.json"
    # 過去のMagenta出力も削除（latest linkを破棄）
    rm -rf "${WORKSPACE_ROOT}/data/Magenta_Studio/outputs/$(basename "$SONG_DIR")/latest"
fi
```

**使い方**:
```bash
bash scripts/e2e_suno_arrangement.sh \
  song_packages/suno_project/song_001 \
  --drums-mode magenta \
  --force-regenerate-drums  # 古いキャッシュを強制削除
```

---

## 2. KPI Gate Magenta専用緩和

**実装箇所**: 新規スクリプト `scripts/kpi_gate_magenta_relaxed.py` または既存の `kpi_gate_enhanced.py` に追加

**コンセプト**:
```python
# kpi_gate_enhanced.py 内で drums_source を検出
def validate_drums_kpi(midi_path: Path, meta: dict, config: dict) -> dict:
    drums_source = meta.get("drums_source", "rule")
    
    # Magenta時は閾値を緩和
    if drums_source == "magenta":
        BACKBEAT_THRESH = config.get("drums", {}).get("kpi", {}).get("backbeat_thresh", 0.75) * 0.9
        MAX_SNARE_BOOST = 16  # ルール時は+24、Magenta時は+16
    else:
        BACKBEAT_THRESH = config.get("drums", {}).get("kpi", {}).get("backbeat_thresh", 0.75)
        MAX_SNARE_BOOST = 24
    
    # 検証ロジック...
    if backbeat_acc < BACKBEAT_THRESH:
        # Auto-Repair時のBoost量を調整
        snare_boost = min(target_delta, MAX_SNARE_BOOST)
```

**設定ファイル**: `configs/gate_prod.yaml` に追加
```yaml
drums:
  kpi:
    backbeat_thresh: 0.75
    magenta_relaxation_factor: 0.9  # Magenta時は0.75 * 0.9 = 0.675
    max_snare_boost_rule: 24
    max_snare_boost_magenta: 16
```

---

## 3. Activity間引き上限のセクション連動

**実装箇所**: `scripts/instrument_midi_to_plan_real.py` 内の activity 間引きロジック

**現状**: 全セクション一律 `skip_prob_cap = 0.85`

**改善**:
```python
# セクション情報を bars.parquet から取得
section = bars_df.loc[bar_idx, "section"]  # "intro", "verse", "chorus", etc.

# セクション別の間引き上限
if section in ("intro", "outro", "bridge"):
    skip_prob_cap = 0.92  # 穏やかなセクションは92%まで間引き可
else:
    skip_prob_cap = 0.85  # それ以外（verse/chorus等）は85%
    
skip_prob = min(activity_to_skip_prob(activity), skip_prob_cap)
```

**効果**: intro/outro/bridgeでの過密演奏を防止、自然な抑揚向上

---

## 4. CI中間ファイル存在チェック（実装済み）

**実装箇所**: `ops/ci_verify_music_package.py` Line 88-132

```python
def check_magenta_intermediates(song_dir: Path, drums_mode: str) -> CheckResult:
    """Magenta中間ファイル存在チェック（drums_mode=magentaの時のみ）"""
    if drums_mode != "magenta":
        return CheckResult(
            name="Magenta intermediate files",
            status="pass",
            details="SKIP: drums_mode != magenta",
        )
    
    required = ["drums_seed.mid", "drums_grooved.mid", "drums_plan.json"]
    missing = [f for f in required if not (song_dir / f).exists()]
    
    if missing:
        return CheckResult(
            name="Magenta intermediate files",
            status="fail",
            details=f"❌ Missing Magenta files: {missing}",
        )
    
    # grooved.mid のノート数確認
    pm = pretty_midi.PrettyMIDI(str(song_dir / "drums_grooved.mid"))
    note_count = sum(len(instr.notes) for instr in pm.instruments)
    if note_count == 0:
        return CheckResult(
            name="Magenta intermediate files",
            status="fail",
            details="❌ drums_grooved.mid has 0 notes",
        )
    
    return CheckResult(
        name="Magenta intermediate files",
        status="pass",
        details=f"✅ All Magenta files present, grooved.mid has {note_count} notes",
    )
```

**使い方**:
```bash
python3 ops/ci_verify_music_package.py \
  --song-dir song_packages/suno_project/song_001 \
  --midi song_packages/suno_project/song_001/full_arrangement.mid \
  --bars song_packages/suno_project/song_001/bars.parquet \
  --drums-mode magenta \
  --tempo-bpm 74.67
```

---

## 5. MAGENTA専用venv分離（推奨設定）

**問題**: Magenta/note-seq/protobufの依存が他のライブラリと衝突（Bus error）

**解決策**: 専用venv作成

```bash
# 専用venv作成
python3.11 -m venv .venv_magenta
source .venv_magenta/bin/activate

# Magenta依存インストール（ピン推奨）
pip install numpy==1.23.5 \
            protobuf==3.20.* \
            note-seq==0.0.5 \
            magenta==2.1.4 \
            pretty-midi \
            tensorflow==2.11.0

deactivate
```

**E2E設定**: `scripts/e2e_suno_arrangement.sh` Line 22
```bash
# Magenta専用Python（別venv、依存衝突回避）
MAGENTA_PY="${MAGENTA_PY:-${WORKSPACE_ROOT}/.venv_magenta/bin/python}"
```

**実行時**:
```bash
# 環境変数で明示的に指定
MAGENTA_PY=/path/to/.venv_magenta/bin/python bash scripts/e2e_suno_arrangement.sh ...
```

---

## 6. CREPE/OaF 常時ON設定（NO-OP安全）

**現状**: Phase Cで `--enable-crepe` / `--enable-oaf` 手動指定

**推奨**: デフォルトON（ファイル無ければNO-OP）

**実装箇所**: `scripts/e2e_suno_arrangement.sh` Line 94-96

**変更**:
```bash
# 既存（手動ON）
ENABLE_CREPE=false
ENABLE_OAF=false

# 推奨（デフォルトON、NO-OP安全設計済み）
ENABLE_CREPE=true
ENABLE_OAF=true
```

**効果**: CREPE/OaF導入後は常に実行、ファイル未存在時は自動スキップ（NO-OP）

---

## 7. ログ保存の徹底

**実装箇所**: `scripts/e2e_suno_arrangement.sh` Magenta呼び出し箇所

**現状**: `magenta_groove.log` のみ保存

**推奨**: 全中間ステップのログ保存
```bash
# Step 2.1: Rule-based seed生成
"$PYTHON_BIN" scripts/recommend_drums.py \
    --song-package "$SONG_PKG" \
    --output "$DRUMS_REC_SEED" \
    --no-ml \
    --topk "$TOPK" \
    $STEMS_ARG 2>&1 | tee "$MAG_OUT_DIR/drums_seed_generation.log"

# Step 2.2: Seed→plan→MIDI
"$PYTHON_BIN" scripts/adapt_drums_to_plan.py \
    --recommendations "$DRUMS_REC_SEED" \
    --out "$SONG_DIR/drums_plan_seed.json" \
    --tempo-bpm "$TEMPO_BPM" 2>&1 | tee "$MAG_OUT_DIR/adapt_drums_seed.log"

# Step 2.3: GrooVAE humanize
"$MAGENTA_PY" "${WORKSPACE_ROOT}/ops/magenta_groove.py" groove \
    -i "$SONG_DIR/drums_seed.mid" \
    -o "$SONG_DIR/drums_grooved.mid" \
    --temp 0.7 \
    2>&1 | tee "$MAG_OUT_DIR/magenta_groove.log"

# Step 2.4: Grooved MIDI→plan
"$PYTHON_BIN" scripts/adapt_drums_to_plan.py \
    --recommendations "$DRUMS_REC_SEED" \
    --grooved-mid "$SONG_DIR/drums_grooved.mid" \
    --out "$SONG_DIR/drums_plan.json" \
    --tempo-bpm "$TEMPO_BPM" \
    --bars "$SONG_DIR/bars.parquet" \
    $STEMS_ARG 2>&1 | tee "$MAG_OUT_DIR/adapt_drums_grooved.log"
```

**効果**: デバッグ時に全ステップの詳細を追跡可能

---

## 適用優先順位

1. **即効性**: stale再利用防止（実装済み）、CI中間ファイルチェック（実装済み）
2. **安定化**: MAGENTA専用venv分離（Bus error対策）
3. **品質向上**: KPI緩和、Activity間引きセクション連動
4. **運用改善**: CREPE/OaF常時ON、ログ保存徹底

---

## 次のステップ（Phase B）

- **軽学習**: GMD MIDI-onlyで学習ループ動作確認（PyTorchインストール後）
- **本学習**: E-GMD WAV+MIDI導入、アタック/マイクロタイミング改善
- **5万曲追加**: クリーン度A/B絞り込み、教師多様性向上

詳細は `docs/PHASE_B_ROADMAP.md` 参照。
