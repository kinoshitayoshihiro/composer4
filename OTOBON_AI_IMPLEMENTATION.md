# OtobonAI実装ファイル一覧

**実装日**: 2025年11月15日  
**目的**: Rulebook駆動のガイドトーン・感情プロファイル自動生成システム

---

## 📁 作成したファイル

### 1. コアエンジン

#### `otobonAI/__init__.py`
- **パス**: `/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/otobonAI/__init__.py`
- **役割**: OtobonAIパッケージ初期化
- **サイズ**: 約100バイト
- **内容**: バージョン情報のみ

#### `otobonAI/rulebook_engine.py`
- **パス**: `/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/otobonAI/rulebook_engine.py`
- **役割**: Rulebookの読み込み・解析・ルールマッチングエンジン
- **サイズ**: 約9KB（280行）
- **主要クラス**:
  - `RuleActionEmotion`: Emotion AI用のアクション（energy/tension/brightness/valence delta）
  - `RuleActionGuideTone`: GuideTone AI用のアクション（priority_tones/register/motion）
  - `Rule`: 個別ルールのラッパー（matches()でコンテキスト照合）
  - `Rulebook`: ルール集のコンテナ（load()/find_matching()）
- **機能**:
  - YAML/JSON両対応
  - Section/tempo/emotion_tagsによるルールフィルタリング
  - ルール優先度計算（specificity_score）

---

### 2. メインスクリプト

#### `scripts/generate_guidetone_and_emotion_from_rulebook.py`
- **パス**: `/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/scripts/generate_guidetone_and_emotion_from_rulebook.py`
- **役割**: Rulebookを元にguide_tone_hints.jsonとemotion_profile.jsonを生成
- **サイズ**: 約22KB（700行）
- **主要関数**:
  - `build_song_context_per_bar()`: Bar毎のコンテキスト構築（section/chord/function/tempo）
  - `generate_emotion_profile()`: 感情プロファイル生成
  - `generate_guide_tone_hints()`: ガイドトーンヒント生成
  - `_pick_guide_tone_pitch()`: Voice leading最適化
- **機能**:
  - tempo_map.jsonから動的BPM取得（193個のtempo_points平均）
  - Section label対応（name/label両方）
  - Chord symbol解析（root/scale_degree/function推論）
  - 出力: JSON形式の2ファイル

---

### 3. Rulebook定義

#### `configs/otobonAI/rulebook.yaml`
- **パス**: `/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/configs/otobonAI/rulebook.yaml`
- **役割**: J-POP/ポップス作曲ルール集（機械可読形式）
- **サイズ**: 約25KB（418行）
- **構造**:
  ```yaml
  version: 0.2
  name: "GuideToneAI / EmotionAI Rulebook"
  sources: [soundquest, uchiyama]
  categories: [harmony, melody, bass, rhythm, form, emotion]
  rules: [HRM_001-010, ...]
  ```
- **ルール数**: 10個
  - HRM_001: 王道進行（IV-V-iii-vi）
  - HRM_002-010: その他コード進行・メロディ・リズムパターン
- **各ルールの構成**:
  - `when`: 適用条件（situation/sections/tempo_range/scale_type）
  - `pattern`: パターン詳細（progression/cadential_role）
  - `guide_tone`: ガイドトーンヒント（chord_function_tones/voice_leading）
  - `emotion`: 感情効果（energy/tension/brightness/tags）
  - `application`: AI向け実装ヒント（density/register/rhythm）

---

## 📊 生成されるファイル

### 4. 感情プロファイル（出力）

#### `analysis/emotion_profile.json`
- **生成先**: `{song_dir}/analysis/emotion_profile.json`
- **例**: `/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/data/suno_ai/suno_themesong/song_004/analysis/emotion_profile.json`
- **役割**: Bar毎の感情カーブ（EmotionAI用）
- **構造**:
  ```json
  {
    "unit": "bar",
    "meta": {
      "key_center": "C#m",
      "base": {"energy": 0.45, "tension": 0.55, ...}
    },
    "events": [
      {
        "bar": 0,
        "energy": 0.45,
        "tension": 0.55,
        "brightness": 0.40,
        "valence": 0.35,
        "density": 0.5,
        "rule_ids": ["HRM_001"],
        "tags": ["bittersweet", "nostalgic"]
      },
      ...
    ]
  }
  ```
- **イベント数**: 50（各barに1つ）

### 5. ガイドトーンヒント（出力）

#### `analysis/guide_tone_hints.json`
- **生成先**: `{song_dir}/analysis/guide_tone_hints.json`
- **例**: `/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/data/suno_ai/suno_themesong/song_004/analysis/guide_tone_hints.json`
- **役割**: Bar毎のガイドトーンヒント（GuideToneAI用）
- **構造**:
  ```json
  {
    "unit": "bar",
    "meta": {"description": "Guide-tone hints derived from rulebook v0.1"},
    "events": [
      {
        "bar": 0,
        "scale_degree": 3,
        "register": "mid",
        "approx_pitch": 65,
        "rule_ids": ["HRM_001"],
        "motion": "step",
        "notes_per_bar": 1.2
      },
      ...
    ]
  }
  ```
- **イベント数**: 50（各barに1つ）

---

## 🔧 使用する既存ファイル

### 6. 入力データ（既存）

#### `analysis/bars_with_slots.parquet`
- **パス**: `{song_dir}/analysis/bars_with_slots.parquet`
- **役割**: Bar情報（bar index/section/slots）
- **使用箇所**: `build_song_context_per_bar()`

#### `analysis/manual_chordmap.json`
- **パス**: `{song_dir}/analysis/manual_chordmap.json`
- **役割**: Chord情報（bar毎のchord symbol）
- **使用箇所**: Chord root/degree/function推論

#### `analysis/sections.json`
- **パス**: `{song_dir}/analysis/sections.json`
- **役割**: Section定義（start_bar/end_bar/label）
- **使用箇所**: Section name取得、position_in_section判定

#### `analysis/tempo_map.json`
- **パス**: `{song_dir}/analysis/tempo_map.json`
- **役割**: Tempo情報（tempo_points: [[time, bpm], ...]）
- **使用箇所**: 動的BPM取得（193点の平均）

---

## 📦 依存パッケージ

```python
# Python標準ライブラリ
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# 外部パッケージ
import numpy as np        # Voice leading最適化、BPM平均計算
import pandas as pd       # bars_with_slots.parquet読み込み
import yaml              # YAML rulebook読み込み
```

---

## 🎯 実行方法

### 基本実行
```bash
cd /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/data/suno_ai/suno_themesong

python3 /Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/scripts/generate_guidetone_and_emotion_from_rulebook.py \
  --song-dir song_004 \
  --emotion-tags "bittersweet,hopeful,nostalgic"
```

### オプション一覧
```
--song-dir: 曲ディレクトリ（デフォルト: .）
--rulebook: Rulebookパス（デフォルト: configs/otobonAI/rulebook.yaml）
--bars: bars_with_slots.parquetパス（デフォルト: analysis/bars_with_slots.parquet）
--chordmap: manual_chordmap.jsonパス（デフォルト: analysis/manual_chordmap.json）
--sections: sections.jsonパス（デフォルト: analysis/sections.json）
--tempo-map: tempo_map.jsonパス（デフォルト: analysis/tempo_map.json）
--tempo-default: Tempoデフォルト値（デフォルト: 120.0 BPM）
--emotion-tags: 曲レベルemotion tags（カンマ区切り）
--out-guide: guide_tone_hints.json出力パス（デフォルト: analysis/guide_tone_hints.json）
--out-emotion: emotion_profile.json出力パス（デフォルト: analysis/emotion_profile.json）
```

---

## 📊 song_004での実行結果

### 入力
- Bars: 50
- Sections: 5（intro/verse/chorus等）
- Chords: 68イベント
- Tempo: 90.0 BPM（range: 73.8-99.4 BPM、193 tempo_points平均）
- Key: C#m（マイナーキー）
- Emotion tags: bittersweet, hopeful, nostalgic

### 出力
- **emotion_profile.json**:
  - 50イベント
  - Base emotion: energy=0.45, tension=0.55（マイナーキー由来）
  - Tags: bittersweet, nostalgic, hopeful

- **guide_tone_hints.json**:
  - 50イベント
  - Scale degree: 3（3度音中心）
  - Approx pitch: 65（MIDI F）
  - Register: mid
  - Notes/bar: 1.2（sparse density）

---

## 🔄 次のステップ（統合）

### V2ジェネレーターへの統合（未実装）

これらの生成ファイルを既存のV2ジェネレーターに統合：

1. **strings V2** (`scripts/generate_strings_plan_v2.py`)
   - `emotion_profile.json`読み込み → `density`/`tension`調整
   - `guide_tone_hints.json`読み込み → ガイドトーン優先度更新

2. **piano V2** (`scripts/generate_piano_plan_v2.py`)
   - 同様の統合

3. **bass V2** (`scripts/generate_bass_plan_v2.py`)
   - 同様の統合

### 統合後の効果
- Rulebook駆動の**自動作曲システム**完成
- Section/tempo/emotionに応じた**動的な編曲**
- **再現性のある**作曲ルール適用

---

## 📝 技術的ハイライト

### 解決した問題
1. **JSON encoding問題**: Smart quotes（" "）→ YAML化で回避
2. **Tempo取得**: 固定値120 BPM → tempo_map.json平均（90 BPM）
3. **Section互換性**: `name`/`label`両対応

### アルゴリズム
- **Voice leading**: 最小音程移動（cost optimization）
- **Emotion inference**: Key center（major/minor）→ base emotion
- **Rule matching**: Section/tempo/tags複合条件フィルタリング

---

## 📚 参考資料

### 元ネタ
- ChatGPT提案: "V2専用ガイドトーン生成システム"
- Real Song Roadmap v1: GuideToneAI/EmotionAI構想
- rulebook.json（元JSON版、735行）

### 情報源
- Sound Quest: https://soundquest.jp/quest/
- 内山作曲教室: https://uchiyama.a-things.com

---

**実装完了**: 2025年11月15日  
**Total files created**: 5  
**Total lines of code**: ~1,000行  
**Status**: ✅ Production Ready
