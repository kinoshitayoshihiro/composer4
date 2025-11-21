# song_004 完全版実行ガイド

## 🎯 概要

`make_song_package_from_sources.sh`（完全版）は、**STEP 1-22の一気通貫フロー**を実現します。

### フロー全体像

```
Phase A (STEP 1-15): 自動段階
├─ tempo_map + bars + sections + chordmap（自動）
├─ stems_features + lyric_anchors
└─ CREPE統合（F0 → plan → MIDI → 監査 → Songpackage初回）

↓ 手動編集: manual_chordmap.json作成（任意）

Phase B (STEP 16-22): LOCK→plan→MIDI→監査
├─ STEP 16: LOCK（manual + auto merge + QA）
├─ STEP 17: music21正規化（ローマ数字・機能分析）
├─ STEP 18: 楽器別view（Pad/Guitar/Piano/Strings/Bass）
├─ STEP 19: 各楽器plan生成（LOCKED参照）
├─ STEP 20: 統合MIDI（可変テンポ）
├─ STEP 21: 最終監査（LOCKED参照）
└─ STEP 22: Songpackage最終版（3 variant）
```

---

## 🚀 基本コマンド

### 1. 全フロー一括実行（推奨）

```bash
bash RUN_SONG_004.sh full
```

**動作:**
- Phase A（STEP 1-15）→ Phase B（STEP 16-22）を連続実行
- `manual_chordmap.json`不在時は`auto chordmap.json`をそのままLOCK

**生成物:**
- `analysis/chordmap_locked.json` - 和声事実（唯一）
- `analysis/chordmap_m21.json` - music21正規化
- `analysis/chordmap_view_{role}.json` - 楽器別view（5種）
- `plans/{role}_plan.json` - 各楽器plan（最大9種）
- `midi/song_004_integrated.mid` - 統合MIDI
- `analysis/harmony_audit_final.json` - 最終監査
- `song_package_{variant}_final.yaml` - Songpackage最終版（3種）

---

### 2. Phase A のみ実行（自動段階）

```bash
bash RUN_SONG_004.sh phaseA
```

**動作:**
- STEP 1-15のみ実行（CREPE統合まで）
- `analysis/chordmap.json`（自動生成）を確認可能
- 手動編集後、Phase B単独実行可能

**次のステップ:**
1. `analysis/chordmap.json`を確認
2. 感情・歌詞・ボーカル実音に合わせて修正
3. `analysis/manual_chordmap.json`として保存
4. `bash RUN_SONG_004.sh phaseB`実行

---

### 3. Phase B のみ実行（LOCK→plan→MIDI）

```bash
bash RUN_SONG_004.sh phaseB
```

**前提条件:**
- Phase A実行済み（`analysis/chordmap.json`等が存在）
- `manual_chordmap.json`作成済み（任意）

**動作:**
- STEP 16-22を実行
- `manual_chordmap.json`不在時は警告表示 → 確認後にauto chordmapをLOCK

---

## 📁 ディレクトリ構造（実行後）

```
data/suno_ai/suno_themesong/song_004/
├── stem_wav/                    # 入力（Stems WAV）
│   ├── oreno_001_(Bass).wav
│   ├── oreno_001_(Drums).wav
│   ├── oreno_001_(Guitar).wav
│   ├── oreno_001_(Keyboard).wav
│   ├── oreno_001_(Strings).wav
│   ├── oreno_001_(Synth).wav
│   └── oreno_001_(Vocals).wav
│
├── stem_midi/                   # 入力（Stems MIDI、任意）
│   └── ...
│
├── analysis/                    # 分析結果（STEP 1-18）
│   ├── bars.parquet             # 完全版bars（23列以上）
│   ├── tempo_map.json           # 可変テンポマップ
│   ├── sections.json            # セクション情報
│   ├── lyric_anchors.json       # 歌詞アンカー
│   ├── chordmap.json            # 自動chordmap（Phase A）
│   ├── manual_chordmap.json     # 手動chordmap（ユーザー作成）
│   ├── chordmap_locked.json     # LOCK版chordmap（STEP 16）
│   ├── chordmap_qa.csv          # QAレポート（STEP 16）
│   ├── chordmap_m21.json        # music21正規化（STEP 17）
│   ├── chordmap_view_pad.json   # Pad view（STEP 18）
│   ├── chordmap_view_guitar.json
│   ├── chordmap_view_piano.json
│   ├── chordmap_view_strings.json
│   ├── chordmap_view_bass.json
│   ├── voicings_guide_{role}.csv # 楽器別ボイシングガイド
│   ├── stems_features.parquet   # Stems特徴量
│   ├── drum_accent_plan.json    # ドラムアクセント指針
│   ├── bassline_plan.csv        # ベースライン指針
│   ├── voicings_guide.csv       # 総合ボイシングガイド
│   ├── style_presets.yaml       # 3 variant定義
│   ├── harmony_audit_report.json # 初回監査（STEP 14）
│   └── harmony_audit_final.json  # 最終監査（STEP 21）
│
├── plans/                       # 各楽器plan（STEP 11, 19）
│   ├── bass_plan.json           # Bass plan（STEP 19、view_bass参照）
│   ├── guitar_plan.json         # Guitar plan（STEP 19、view_guitar参照）
│   ├── guitar_plan_optimized_micro.json # CREPE enhanced（STEP 11）
│   ├── piano_plan.json          # Piano plan（STEP 19、view_piano参照）
│   ├── piano_plan_hybrid.json   # CREPE enhanced（STEP 11）
│   ├── strings_plan.json        # Strings plan（STEP 19、view_strings参照）
│   ├── strings_countermelody_plan_vl.json # CREPE enhanced（STEP 11）
│   ├── pad_plan.json            # Pad plan（STEP 19、view_pad参照）
│   └── drums_plan.json          # Drums plan（STEP 19、drum_accent参照）
│
├── features/                    # 特徴量（STEP 10）
│   └── vocal_f0.parquet         # CREPE連続F0
│
├── midi/                        # MIDI出力（STEP 13, 20）
│   ├── piano_plan_hybrid.mid    # Piano個別MIDI（CREPE）
│   ├── strings_countermelody_plan_vl.mid # Strings個別MIDI（CREPE）
│   ├── guitar_plan_optimized_micro.mid   # Guitar個別MIDI（CREPE）
│   ├── song_004_hybrid_crepe.mid # CREPE統合MIDI（STEP 13）
│   └── song_004_integrated.mid  # 全楽器統合MIDI（STEP 20）
│
└── song_package_{variant}_final.yaml # Songpackage最終版（STEP 22）
    ├── song_package_soft_final.yaml
    ├── song_package_standard_final.yaml
    └── song_package_bright_final.yaml
```

---

## 🔑 重要ファイル説明

### `manual_chordmap.json`（ユーザー作成）

**形式:**
```json
{
  "events": [
    {
      "time_ql": 0.0,
      "bar": 0,
      "root": "C",
      "quality": "major"
    },
    {
      "time_ql": 4.0,
      "bar": 1,
      "root": "Am",
      "quality": "minor"
    }
  ]
}
```

**編集方針:**
- `time_ql`は16分音符グリッド（0.25刻み）推奨
- `root`はmusic21準拠（C, D♭, F#等）
- `quality`は既知定義のみ使用（major/minor/7/m7/maj7等）
- 感情・歌詞・ボーカル実音に合わせて修正

**QAチェック項目（STEP 16で自動検出）:**
- 時間重複（同一time_qlに複数イベント）
- 未知quality（定義外の記法）
- グリッド逸脱（16分音符グリッドから外れる）

---

### `chordmap_locked.json`（自動生成、唯一の和声事実）

**生成タイミング:** STEP 16

**内容:**
- `manual_chordmap.json`（存在時）+ `chordmap.json`（auto）のマージ
- 同一time_qlは`manual`が優先
- QAチェック結果を`meta`に記録

**重要性:**
- **以降全てのplan生成器がこのファイルを参照**
- root/quality/time_qlは不変（楽器別viewは注釈のみ）

---

### `chordmap_view_{role}.json`（楽器別view、STEP 18生成）

**役割:** 楽器ごとの演奏制約・推奨事項を付与

**追加情報:**
- `tensions_allowed`: 使用可能テンション（例: ["9", "13"]）
- `avoid_tensions`: 回避テンション（例: ["#11", "b9"]）
- `omit_third`: 3rd省略許容（Pad常時ON区間でtrue）
- `prefer_inversion`: 推奨転回形（root/1st/2nd）
- `register_low/high`: 音域制限（MIDI番号）
- `density_scale`: 密度スケール（セクション別、0.0-1.0+）
- `label`: セクションラベル（intro/verse/chorus等）

**使用例（Bass）:**
```json
{
  "time_ql": 4.0,
  "bar": 1,
  "root": "Am",
  "quality": "minor",
  "tensions_allowed": [],
  "avoid_tensions": ["b9", "#9", "#11", "b13", "9", "11", "13"],
  "omit_third": false,
  "prefer_inversion": "root",
  "register_low": 40,
  "register_high": 55,
  "density_scale": 0.9,
  "label": "verse"
}
```

---

## 🎵 plan生成器の実装要件（STEP 19）

各楽器のplan生成器は、以下を満たす必要があります:

### 必須引数

```bash
python scripts/{role}/generate_{role}_plan.py \
  --chordmap analysis/chordmap_locked.json \
  --view analysis/chordmap_view_{role}.json \
  --sections analysis/sections.json \
  --tempo-map analysis/tempo_map.json \
  --out plans/{role}_plan.json
```

### 必須実装

1. **LOCKED参照:** `chordmap_locked.json`のみを和声事実として使用
2. **view適用:** `chordmap_view_{role}.json`の制約を尊重
   - `tensions_allowed/avoid_tensions`でテンション選択
   - `register_low/high`で音域制限
   - `density_scale`でイベント密度調整
   - `omit_third`でPad衝突回避
3. **固定BPM禁止:** `tempo_map.json`のみをテンポ源とする

### 例（Bass plan生成器）

```python
#!/usr/bin/env python3
import json
from pathlib import Path

def generate_bass_plan(chordmap_locked, view_bass, sections, tempo_map, output):
    # LOCKED読み込み
    locked = json.loads(Path(chordmap_locked).read_text())
    view = json.loads(Path(view_bass).read_text())
    
    events = []
    for i, chord in enumerate(locked["events"]):
        # view取得
        v = view["events"][i]
        
        # root音取得
        root_midi = chord_to_midi(chord["root"]) + 36  # E2基準
        
        # 音域制限
        if root_midi < v["register_low"]:
            root_midi += 12
        elif root_midi > v["register_high"]:
            root_midi -= 12
        
        # density適用
        if random.random() > v["density_scale"]:
            continue  # スキップ
        
        # イベント生成
        events.append({
            "time_ql": chord["time_ql"],
            "pitch_midi": root_midi,
            "duration_ql": 4.0,
            "velocity": 80
        })
    
    # 出力
    Path(output).write_text(json.dumps({"events": events}, indent=2))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chordmap", required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--sections", required=True)
    ap.add_argument("--tempo-map", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    
    generate_bass_plan(a.chordmap, a.view, a.sections, a.tempo_map, a.out)
```

---

## 🐛 トラブルシューティング

### 1. `manual_chordmap.json`不在時の挙動

**症状:** Phase B実行時に警告表示

**対処:**
- 警告を確認し、`auto chordmap.json`をLOCKとして使用（Enter押下）
- または、`manual_chordmap.json`作成後に再実行

---

### 2. QAエラー（時間重複・未知quality等）

**症状:** `chordmap_qa.csv`にERROR行が記録される

**対処:**
```bash
# QAレポート確認
cat data/suno_ai/suno_themesong/song_004/analysis/chordmap_qa.csv

# エラー内容に応じてmanual_chordmap.json修正
# 例: 時間重複 → time_qlを0.25ずらす
# 例: 未知quality → 既知定義に変更（major/minor/7/m7等）

# 再実行
bash RUN_SONG_004.sh phaseB
```

---

### 3. plan生成器が見つからない

**症状:** `⚠️  {role} plan generator not found, skipping`

**対処:**
- 該当楽器のplan生成器が未実装の場合はスキップ（正常動作）
- CREPE enhanced plan（Strings/Guitar/Piano）は優先使用される
- Drumsのみ必須（`scripts/drums/generate_drums_plan.py`が必要）

---

## 📊 次のステップ（推奨）

### 1. 統合MIDI確認

```bash
open data/suno_ai/suno_themesong/song_004/midi/song_004_integrated.mid
```

**確認項目:**
- 可変テンポ動作（tempo_map.json通りか）
- 各楽器の音域（viewの`register_low/high`通りか）
- Pad常時ON区間での3rd省略（衝突回避されているか）

---

### 2. 監査レポート確認

```bash
cat data/suno_ai/suno_themesong/song_004/analysis/harmony_audit_final.json | jq
```

**確認項目:**
- 和声逸脱（chordmap_locked.jsonと一致しているか）
- テンション違反（viewの制約を守っているか）

---

### 3. Songpackage確認

```bash
# Standard variant確認
cat data/suno_ai/suno_themesong/song_004/song_package_standard_final.yaml

# 3 variant比較
diff -u song_package_soft_final.yaml song_package_standard_final.yaml
```

**確認項目:**
- CREPE統計（`harmony.crepe_ext`）
- paths.midi.integrated（統合MIDI参照）
- variant差分（velocity/density/instrumentation/effects）

---

## 🎯 まとめ

**一気通貫フロー:**
```bash
bash RUN_SONG_004.sh full
```

**段階的フロー（推奨）:**
```bash
# 1. Phase A（自動段階）
bash RUN_SONG_004.sh phaseA

# 2. manual_chordmap.json作成（感情・歌詞基準で修正）

# 3. Phase B（LOCK→plan→MIDI→監査）
bash RUN_SONG_004.sh phaseB
```

**全22ステップ（1-22）が自動実行され、以下が生成されます:**
- chordmap_locked.json（唯一の和声事実）
- 楽器別view（5種） + voicings_guide（5種）
- 各楽器plan（最大9種）
- 統合MIDI（可変テンポ）
- 最終監査レポート
- Songpackage最終版（3 variant）
