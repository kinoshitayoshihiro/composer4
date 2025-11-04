# Stem Harmony Analysis 実装レポート

**実装日**: 2025-01-XX  
**ファイル**: `analysis/stem_harmony.py`  
**目的**: Suno stems → mix_context / audio_chordmap / guides 変換

---

## 概要

本ファイルは **実コード雛形**（未設定=NO-OPを前提）として実装されています。

### 設計方針

1. **依存最小化**: 標準ライブラリ + numpy/pydub/music21/pretty_midi のみ
2. **NO-OP安全**: 例外時は空リスト/デフォルト値を返す
3. **戻り値の形は最終形**: 将来的なアルゴリズム差し替えが容易
4. **段階的実装**: Phase 13-18が明確に分離

### フェーズ構成

| Phase | 機能 | 実装状況 |
|-------|------|---------|
| Phase 13 | ビートグリッド生成 | ✅ 簡易一定テンポ版 |
| Phase 14 | 活動マスク推定 | ✅ RMSベース |
| Phase 15 | コード候補推定 | ✅ スケルトン（key_hint → I/V/IV） |
| Phase 16 | Stem投票集約 | ✅ 活動重み付き投票 |
| Phase 17 | アクセント格子抽出 | ✅ 簡易（1拍目=kick等） |
| Phase 18 | ガイドMIDI書き出し | ✅ テンポ/マーカー/コード |

---

## 実装詳細

### Phase 13: ビートグリッド生成

**関数**: `make_beat_grid(stems, default_bpm, time_sig)`

**現状の実装**:
- 一定テンポの安全フォールバック
- drums → bass → 最初のstem の順で優先
- オーディオ長からQL計算

**将来の改善**:
- オンセット検出によるテンポトラッキング
- 可変テンポ対応

**戻り値**:
```python
{
    "bpm": 120.0,
    "time_sig": [4, 4],
    "ql_per_bar": 4.0,
    "beats": [0.0, 1.0, 2.0, ...],  # 各拍のQL
    "bars": [0.0, 4.0, 8.0, ...],   # 各小節のQL
    "duration_ql": 360.0,
    "sec_per_q": 0.5
}
```

### Phase 14: 活動マスク推定

**関数**: `estimate_activity(wav_path, beat_grid)`

**現状の実装**:
- 小節ごとのRMS計算
- 95パーセンタイル正規化 → 0..1

**戻り値**:
```python
[(bar_index, activity_0_1), ...]
例: [(0, 0.8), (1, 0.9), (2, 0.7)]
```

**NO-OP安全**:
- ファイル読み込み失敗 → 空リスト `[]`

### Phase 15: コード候補推定

**関数**: `estimate_chords_per_stem(wav_path, beat_grid, role, key_hint, top_n)`

**現状の実装**（スケルトン）:
- key_hint あり → I/V/IV をダイアトニック候補
- key_hint なし → C/G/F を候補
- 役割別スコア重み（bassはIを好む等）

**将来の改善**:
- 拍同期クロマベクトル抽出
- HMM/Viterbi によるコード推定

**戻り値**:
```python
{
    (bar, beat): [
        {"chord": "C:maj", "score": 0.70},
        {"chord": "G:maj", "score": 0.20}
    ],
    ...
}
```

### Phase 16: Stem投票集約

**関数**: `aggregate_stem_chords(stem_votes, activity, key_hint, sections, cfg)`

**現状の実装**:
- 活動マスク × 役割重み で投票
- 最大票のコードを採用
- 穴埋め（前回コード or key_hintのI）

**重みデフォルト**:
```python
{
    "bass": 0.35,
    "guitar": 0.35,
    "piano": 0.2,
    "strings": 0.1
}
```

**戻り値**:
```python
{
    "key": "C:maj",
    "confidence_key": 0.8,
    "items": [
        {"bar": 0, "beat": 1, "chord": "C:maj", "confidence": 0.7},
        {"bar": 0, "beat": 2, "chord": "G:maj", "confidence": 0.6},
        ...
    ]
}
```

### Phase 17: アクセント格子抽出

**関数**: `extract_accent_grid(stems, beat_grid)`

**現状の実装**（簡易プレースホルダ）:
- kick: 各小節の1拍目
- snare: 2拍目 & 4拍目（4/4想定）
- hihat: 全拍（1QLごと）

**将来の改善**:
- オンセット検出による実際のkick/snare位置抽出

**戻り値**:
```python
{
    "kick": [0.0, 4.0, 8.0, ...],
    "snare": [1.0, 3.0, 5.0, 7.0, ...],
    "hihat": [0.0, 1.0, 2.0, 3.0, ...],
    "strum_ud": []
}
```

### Phase 18: ガイドMIDI書き出し

**関数**: `export_guides_to_midi(out_path, beat_grid, sections, audio_chordmap)`

**実装内容**:
- テンポ設定（pretty_midi.PrettyMIDI初期化）
- セクションマーカー追加
- ブロックコード（triad + 低Velルート）

**出力例**:
- Track 0: "Guide Chords"
  - Root/Third/Fifth (velocity=40)
  - Root-1oct (velocity=25)
- Markers: "INTRO", "VERSE", "CHORUS"

**NO-OP安全**:
- 例外時は何も書き出さず（CI安全）

---

## ユーティリティ関数

### BeatGrid データクラス

```python
@dataclass
class BeatGrid:
    bpm: float
    time_sig: Tuple[int, int]
    ql_per_bar: float
    beats: List[float]
    bars: List[float]
    duration_ql: float
    
    def to_dict(self) -> Dict[str, Any]: ...
```

### Role推定

**関数**: `guess_role_from_path(path)`

**実装**:
```python
ROLE_ALIASES = {
    "vocals": "vocals",
    "drums": "drums",
    "bass": "bass",
    "guitar": "guitar",
    "piano": "piano",
    "strings": "strings",
    ...
}
```

**例**:
- `"path/to/drums.wav"` → `"drums"`
- `"vocals_main.mp3"` → `"vocals"`
- `"unknown.wav"` → `"other"`

---

## テスト結果

### テストスクリプト
`scripts/test_stem_harmony.py`

### 実行結果
```
✅ PASS: Role Guessing
✅ PASS: Beat Grid (Phase 13)
✅ PASS: Activity Mask (Phase 14)
✅ PASS: Chord Estimation (Phase 15)
✅ PASS: Chord Aggregation (Phase 16)
✅ PASS: Accent Grid (Phase 17)
✅ PASS: MIDI Export (Phase 18)

Total: 7/7 tests passed 🎉
```

### 検証内容

**Test 1: Role Guessing**
- ファイル名から役割を推定
- 7パターンすべて正確に識別

**Test 2: Beat Grid (Phase 13)**
- BPM=120, 4/4拍子で360QL（3分）のグリッド生成
- 361 beats, 91 bars 生成確認

**Test 3: Activity Mask (Phase 14)**
- 存在しないファイルで空リスト返却（NO-OP安全）

**Test 4: Chord Estimation (Phase 15)**
- Key hint "C:maj" → I/V/IV 候補生成
- 361拍分のコード候補生成確認

**Test 5: Chord Aggregation (Phase 16)**
- Bass + Guitar の投票を活動マスク重み付きで集約
- audio_chordmap 形式の出力確認

**Test 6: Accent Grid (Phase 17)**
- Kick: 91箇所（各小節1拍目）
- Snare: 180箇所（2&4拍目）
- Hihat: 361箇所（全拍）

**Test 7: MIDI Export (Phase 18)**
- 9,101 bytes のMIDIファイル生成確認
- テンポ/マーカー/コードが含まれる

---

## 使用例

### 基本フロー

```python
from analysis.stem_harmony import (
    make_beat_grid,
    estimate_activity,
    estimate_chords_per_stem,
    aggregate_stem_chords,
    extract_accent_grid,
    export_guides_to_midi,
)

# 1. Stems辞書
stems = {
    "drums": "path/to/drums.wav",
    "bass": "path/to/bass.wav",
    "guitar": "path/to/guitar.wav",
}

# 2. ビートグリッド生成
beat_grid = make_beat_grid(stems, default_bpm=120.0, time_sig=(4, 4))

# 3. 各stemの活動マスク
activity = {}
for role, path in stems.items():
    activity[role] = estimate_activity(path, beat_grid)

# 4. 各stemのコード候補
stem_votes = {}
for role, path in stems.items():
    stem_votes[role] = estimate_chords_per_stem(
        path, beat_grid, role, key_hint="C:maj"
    )

# 5. 投票集約 → audio_chordmap
sections = [
    {"bar": 0, "label": "Intro"},
    {"bar": 4, "label": "Verse"},
]
cfg = {"weights": {"bass": 0.35, "guitar": 0.35, "piano": 0.2}}

audio_chordmap = aggregate_stem_chords(
    stem_votes, activity, key_hint="C:maj", sections=sections, cfg=cfg
)

# 6. アクセント格子
accents = extract_accent_grid(stems, beat_grid)

# 7. ガイドMIDI書き出し
export_guides_to_midi(
    "output/guide.mid", beat_grid, sections, audio_chordmap
)
```

### mix_context への統合

```python
mix_context = {
    # ... existing data ...
    "beat_grid": beat_grid,
    "stem_activity": activity,
    "audio_chordmap": audio_chordmap,
    "kick_onsets_ql": accents["kick"],
    "snare_onsets_ql": accents["snare"],
    "hihat_onsets_ql": accents["hihat"],
}
```

---

## 将来の改善計画

### Priority ★★★★★ - テンポトラッキング
- **現状**: 一定テンポ（default_bpm）
- **改善**: librosa による onset detection + tempo tracking
- **効果**: 可変テンポ・ルバート対応

### Priority ★★★★ - コード推定精度向上
- **現状**: key_hint → I/V/IV のスケルトン
- **改善**: 拍同期クロマベクトル + HMM/Viterbi
- **効果**: 実際のオーディオからコード検出

### Priority ★★★ - アクセント検出
- **現状**: 拍位置ベースのプレースホルダ
- **改善**: librosa.onset_detect によるkick/snare実測
- **効果**: 実際のドラムパターン反映

### Priority ★★ - 活動マスク高度化
- **現状**: RMS正規化
- **改善**: スペクトル重心/ZCR等の特徴量追加
- **効果**: より精密な activity 推定

---

## 依存関係

### 必須
- `numpy`: 配列操作・正規化
- `pydub`: オーディオファイル読み込み・RMS計算
- `pretty_midi`: MIDIファイル書き出し

### オプショナル
- `music21`: コード解析（フォールバックあり）

### 将来追加予定
- `librosa`: オンセット検出・テンポトラッキング・クロマ抽出

---

## まとめ

### 実装の特徴

1. ✅ **動く骨格**: 全Phase 13-18が実装済み
2. ✅ **NO-OP安全**: 例外時も安全に空値返却
3. ✅ **戻り値形式確定**: 将来のアルゴリズム差し替えが容易
4. ✅ **依存最小**: 標準+既存依存のみ
5. ✅ **テスト完備**: 7/7 tests passed

### 現状の制約

- ⚠️ 一定テンポのみ（可変テンポ未対応）
- ⚠️ コード推定はスケルトン（key_hint依存）
- ⚠️ アクセント検出は簡易版（拍位置ベース）

### 推奨される使用方法

**現在（v1.0）**:
- ユーザーが BPM / key / time_sig を指定
- 安全フォールバックとして動作
- ガイドMIDI生成による耳チェック

**将来（v2.0）**:
- librosa統合によるオーディオ解析
- 自動テンポ/キー/コード検出
- 高精度なアクセント抽出

---

**Status**: ✅ **実装完了・テスト通過**  
**Next**: Suno stem統合ワークフロー構築 → 実オーディオでの検証
