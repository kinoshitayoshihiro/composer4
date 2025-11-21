# Continue Module — RhythmAI × Stage3 条件でモチーフ延伸

## 目的
1 小節モチーフを RhythmAI のボキャブラリと Stage3 条件（intensity/density/swing）に基づき 8 小節以上へ拡張する。Phase 3 の「Continue」要件に対応。

## 入出力
- **入力モチーフ**: `motif.json`（`events` 配列、`time_ql`/`duration_ql`/`velocity` 等）。
- **Stage3 条件**: `outputs/stage3/conditions.parquet` または CSV。
- **RhythmAI vocab**: `rhythm_vocab.yaml` + `groove_vocab.parquet`（任意）。
- **出力**: `{ "events": [...], "metadata": {...} }` 形式の JSON。`metadata.stage3_conditions` 内で各複製バーに採用した Stage3 サンプル（loop_id / density / intensity / swing）を確認できる。

### Stage3 → Heuristics マッピング
- **density**: `backbeat_strength`（欠損時は `n_downbeats`）を 0.5–2.0 にスケールし、コピー回数・ノート間引きに反映。
- **intensity**: `arousal` を -1〜+1 から 0.4〜1.8 にスケールし、velocity バイアスへ変換。
- **swing**: `swing_pct` (0–100) を 0〜0.6 に正規化し、16 分オフビートへ位相シフト。
- **loop 選択**: `--stage3-loop-id` を指定すると該当 XMIDI ループのみから条件をサンプル。未指定時は DataFrame からランダム抽出。

## 主要クラス
| クラス | 役割 |
| --- | --- |
| `ContinueModule` | RhythmAI / Stage3 条件を束ね、モチーフを複製＋変奏 |
| `Stage3Condition` | Parquet 1 行を密度/強度/スウィング情報に変換 |

### `ContinueModule.extend`
1. Stage3 条件から対象バーの `density/intensity/swing` を取得。
2. RhythmAI から `rhythm_pattern_id` を選択（manifest が無い場合は NO-OP）。
3. モチーフを `density_scale` 回複製し、`velocity_bias` `swing_ratio` を反映。
4. イベントへ `rhythm_pattern_id` メタを付与し `events` として返却。

## CLI
```bash
python scripts/continue_module.py \
  --motif sandbox/motif.json \
  --out sandbox/motif_continue.json \
  --source-bars 1 \
  --target-bars 8 \
  --instrument piano \
  --section chorus \
  --beats-per-bar 4 \
  --stage3-conditions outputs/stage3/conditions.parquet \
  --stage3-loop-id XMIDI_angry_classical_0631TTPB \
  --rhythm-manifest data/rhythm_vocab.yaml \
  --groove-vocab data/groove_vocab.parquet
```

## 統合ステップ
1. `generate_*_plan_v2.py` でスロット化後、1 小節モチーフを `motif.json` として書き出し。
2. 上記 CLI の出力を Plan へ差し戻し、`arrangement_orchestrator.py` 経由で取り込む。
3. `metadata.rhythm_pattern_ids` と `metadata.stage3_conditions` を `json2midi.py` → KPI ゲートで可視化。

## Guitar Plan V2 への組み込み
Stage2/Stage3 のギターリフは `scripts/generate_guitar_plan_v2.py` から直接 Continue を呼び出せるようになりました。設定は以下の 2 レイヤで制御します。

### 1. Section Policy でのトリガー
セクション定義に `guitar_continue` を追加すると、そのセクションの riff_slot が連続したまとまりについて Continue が自動的に走ります。

```yaml
sections:
  Chorus:
    guitar: 0.8
    density: 0.7
    guitar_continue:
      target_bars: 8   # 任意。未指定なら instrument default
      source_bars: 1
```

### 2. Instrument 設定
`policy.instruments.guitar.continue` にデフォルト値と Stage3/Continue リソースを記述します。

```yaml
instruments:
  guitar:
    continue:
      enabled: true
      sections:
        chorus:
          target_bars: 8
        bridge: {}
      source_bars: 1
      target_bars: 8
      stage3_conditions: outputs/stage3/conditions.parquet
      groove_vocab: data/groove_vocab.parquet
      rhythm_manifest: data/rhythm_vocab.yaml
      stage3_loop_id: XMIDI_demo_001   # 任意
      require_riff_slot: true         # false にすると slot 無しでも実行
```

### CLI オーバーライド
`generate_guitar_plan_v2.py` には下記オプションを追加しています。リモート環境や Continue の AB テスト時に使用してください。

| オプション | 説明 |
| --- | --- |
| `--continue-enable / --continue-disable` | ポリシー設定を無視して ON/OFF |
| `--continue-sections chorus,bridge` | 対象セクションを強制指定 |
| `--continue-stage3 path.csv` | Stage3 条件ファイルのフルパス |
| `--continue-loop-id XMIDI_xxx` | Stage3 ループを固定 |
| `--continue-groove / --continue-manifest` | RhythmAI vocab の差し替え |
| `--continue-motif motif.json` | 既存 Plan とは別のモチーフ JSON を使用 |
| `--continue-source-bars / --continue-target-bars` | モチーフ・生成長を数値指定 |
| `--continue-allow-non-slot` | riff_slot=0 のバーにも適用 |

### モチーフ出典について
Continue は MIDI/イベント列を前提としており、生 WAV を直接解析してモチーフ化する機能は提供していません。Suno 原曲 WAV をモチーフに使いたい場合は、先に社内の F0+Onsets パイプラインや DAW で MIDI 化し、`continue_module.py` 互換の JSON (`[{"time_ql":..., "duration_ql":..., "velocity":...}]`) に変換した上で `--continue-motif` で指定してください。

## 今後の拡張
- Stage3 条件の係数テーブル化（セクション別重み）。
- EmotionAI 連動で `velocity_bias` をセクションごとに可変にする。
- Continue 生成後に DurationHumanizeAI（DUV）を適用し、Magenta Interpolate への接続を行う。