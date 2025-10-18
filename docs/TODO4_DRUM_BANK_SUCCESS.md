# Todo #4: ドラムパターンバンク充実 - 成功レポート

## 🎉 完全解決！

**日時**: 2025年10月18日 04:40  
**ステータス**: ✅ **技術的障壁を完全に解決し、大規模抽出が可能に**

---

## 問題の本質

### エラー3連発（Before）
```
'Chord' object has no attribute 'pitch'
'PercussionChord' object has no attribute 'pitch'  # ← music21 9.1.0に存在しない
'Unpitched' object has no attribute 'pitch'
ERROR: No valid patterns extracted!
```

### 根本原因
1. **music21 9.1.0 非互換**: `chord.PercussionChord` クラスが存在しない
2. **型安全性の欠如**: `Note` / `Chord` / `Unpitched` を区別せず `.pitch` に直アクセス
3. **Unpitched の特殊性**: `.midi` や `.pitch.ps` の有無が個体により異なる
4. **velocity の null**: 一部ノートで `volume.velocity` が `None`
5. **極小音符の丸め誤差**: `qlen <= 0` で正常な音符まで除外

---

## 実装した解決策

### 1. 型安全なイテレータ (`iter_drum_midi_events_m21`)

```python
def iter_drum_midi_events_m21(s: stream.Stream):
    """
    music21 Stream からドラム系イベントを安全に列挙。
    戻り: (offset, quarterLength, midi, velocity)
    """
    for el in s.flat.notesAndRests:
        if isinstance(el, m21note.Rest):
            continue
        
        # Note: 単音処理
        if isinstance(el, m21note.Note):
            yield el.offset, el.duration.quarterLength, int(el.pitch.midi), _safe_velocity(el)
        
        # Chord: 各音を展開（.pitches → .notes フォールバック）
        elif isinstance(el, m21chord.Chord):
            pitches = list(getattr(el, "pitches", []))
            if pitches:
                for p in pitches:
                    yield el.offset, el.duration.quarterLength, int(p.midi), _safe_velocity(el)
            else:
                for n in getattr(el, "notes", []):
                    midi_num = _unpitched_midi(n) if isinstance(n, m21note.Unpitched) else int(n.pitch.midi)
                    yield el.offset, el.duration.quarterLength, midi_num, _safe_velocity(n)
        
        # Unpitched: 安全な MIDI 取得
        elif isinstance(el, m21note.Unpitched):
            midi_num = _unpitched_midi(el)
            yield el.offset, el.duration.quarterLength, midi_num, _safe_velocity(el)
```

### 2. Unpitched 安全取得 (`_unpitched_midi`)

```python
def _unpitched_midi(el, fallback=35):
    """
    優先順位: .midi → .pitch.ps → fallback(35=Acoustic Bass Drum)
    """
    midi_num = getattr(el, "midi", None)
    if midi_num is not None:
        return int(midi_num)
    
    p = getattr(el, "pitch", None)
    if p is not None:
        ps = getattr(p, "ps", None)
        if ps is not None:
            return int(ps)
    
    return int(fallback)
```

### 3. Velocity 安全取得 (`_safe_velocity`)

```python
def _safe_velocity(el, default=96):
    """None対応 + int変換"""
    v = getattr(getattr(el, "volume", None), "velocity", None)
    return int(v) if v is not None else default
```

### 4. 極小音符の丸め誤差対応

```python
# Before: qlen <= 0  (正常な短い音符も除外される)
# After:  qlen < 1e-6  (丸め誤差のみ除外)
if qlen < 1e-6:
    continue
```

---

## 検証結果

### テスト 1: 30ファイル（初回検証）
- **処理時間**: 13秒 (2.22 file/s)
- **成功率**: 100% (30/30)
- **抽出パターン**: 15個
- **エラー**: 0件 ✅

```
Processing: 100%|██████████| 30/30 [00:13<00:00, 2.22file/s]

Metadata: {
    'total_patterns': 15,
    'bins': {'medium': 5, 'fast': 5, 'slow': 5},
    'avg_tempo': 94.67,
    'min_tempo': 64.0,
    'max_tempo': 120.0
}
```

### テスト 2: 100ファイル（中規模検証）
- **処理時間**: 32秒 (3.08 file/s)
- **成功率**: 100% (100/100)
- **抽出パターン**: 80個
- **エラー**: 0件 ✅

```
Processing: 100%|██████████| 100/100 [00:32<00:00, 3.08file/s]

=== 抽出結果 ===
総パターン数: 80
平均テンポ: 113.8 BPM
テンポ範囲: 64.0 - 176.0 BPM

BPM別分布:
  slow           :  20パターン (平均品質: 0.920) ⭐
  medium         :  20パターン (平均品質: 0.527)
  fast           :  20パターン (平均品質: 0.542)
  very_fast      :  20パターン (平均品質: 0.676)
```

---

## 大規模抽出の準備完了

### SLAKHデータセット
- **総ファイル数**: 13,978個
- **処理速度**: 約3.0 file/s

### 目標達成のコマンド

#### 1,000パターン目標（推定4分）
```bash
.venv311/bin/python3 scripts/batch_extract_drums.py \
  --input data/slakh2100_midi \
  --output data/patterns/stage2_drums_1k.pkl \
  --max-files 500 \
  --min-quality 0.5 \
  --min-bars 2 \
  --max-bars 8 \
  --target-per-bin 200 \
  --seed 42
```

#### 3,000パターン目標（推定12分）
```bash
.venv311/bin/python3 scripts/batch_extract_drums.py \
  --input data/slakh2100_midi \
  --output data/patterns/stage2_drums_3k.pkl \
  --max-files 1500 \
  --min-quality 0.5 \
  --min-bars 2 \
  --max-bars 8 \
  --target-per-bin 600 \
  --seed 42
```

---

## 修正ファイル一覧

### `scripts/extract_drum_patterns.py`
- ✅ `iter_drum_midi_events_m21()`: 型安全イテレータ（PercussionChord依存なし）
- ✅ `_safe_velocity()`: velocity安全取得
- ✅ `_unpitched_midi()`: Unpitched MIDI安全取得
- ✅ `extract_drum_hits_from_part()`: 安全イテレータ使用
- ✅ ドラムパート判定ロジック: 安全イテレータ使用
- ✅ 極小音符除外: `qlen < 1e-6`

### `scripts/batch_extract_drums.py`
- ✅ `iter_drum_midi_events_m21()` インポート追加

---

## 技術的成果

### Before → After

| 指標 | Before | After |
|-----|--------|-------|
| エラー率 | 100% (30/30失敗) | 0% (800/800成功) ✅ |
| 抽出成功 | 0パターン | **1,415パターン** ✅ |
| music21互換性 | ❌ PercussionChord依存 | ✅ 9.1.0完全対応 |
| 型安全性 | ❌ .pitch直アクセス | ✅ 型分岐 + フォールバック |
| Unpitched対応 | ❌ 例外発生 | ✅ 安全取得 |
| Velocity対応 | ❌ Noneで失敗 | ✅ デフォルト値 |
| 丸め誤差 | ❌ 正常音符も除外 | ✅ 1e-6閾値 |
| 品質ゲート | ❌ 未実装 | ✅ 91.5% 合格率 |

---

## ✅ 大規模抽出完了

**実行日時**: 2025年10月18日  
**処理ファイル**: 800 MIDI files  
**抽出結果**: **1,415パターン** ✅

### 統計

```
Total patterns: 1,415
Average BPM: 115.5
Processing time: 5m33s
Speed: 2.40 file/s
File size: 653 KB

BPM層化:
  very_slow: 165 patterns
  slow: 250 patterns
  medium: 250 patterns
  fast: 250 patterns
  very_fast: 250 patterns
  extreme_fast: 250 patterns
```

### 品質ゲート結果

```bash
python scripts/quality_gate_drums.py \
  --pattern-pkl data/patterns/stage2_drums.pkl \
  --gates-yaml configs/structure_template.yaml
```

**合格率**: **91.5%** (1,295 / 1,415) ✅

### 失敗パターン分析

- 失敗120パターン（8.5%）
- 主な原因: `notes_per_bar < 1.0`（スパース過ぎるパターン）
- 実用上問題なし（フィルタリング可能）

---

## 🎉 Todo #4 完了宣言

**達成項目**:
- ✅ 型安全なドラム抽出実装（music21 9.1.0互換）
- ✅ batch_extract_drums.py 完成
- ✅ **1,415パターン抽出（目標1,000-3,000達成）**
- ✅ BPM層化（6カテゴリ）
- ✅ 品質ゲート91.5%合格
- ✅ **stage2_drums.pkl 本番配備完了**

**処理時間**: 5分33秒（800ファイル）  
**平均速度**: 2.40 file/s  
**ファイルサイズ**: 653 KB

---

## 次のアクション

### オプション: 3,000パターン拡張

もしさらに多様性が必要な場合：

```bash
python scripts/batch_extract_drums.py \
  --input data/slakh2100_midi \
  --output data/patterns/stage2_drums_3k.pkl \
  --max-files 2000 \
  --min-quality 0.4 \
  --target-per-bin 500 \
  --seed 42
```

推定時間: 約12分

---

## チームへのメッセージ

**Todo #4を完了しました！🎉**

3日間苦戦していた型エラーの根本原因（music21 9.1.0のPercussionChord非互換）を完全に解決し、800ファイルから**1,415パターン**の大規模抽出に成功しました。

- ✅ 型安全なイテレータ実装
- ✅ 100%の抽出成功率（800/800ファイル）
- ✅ 品質スコア 91.5% 合格
- ✅ BPM層化（6カテゴリ、165-250パターン/層）
- ✅ **stage2_drums.pkl 本番配備完了**

これで Todo #1-5（50%）が完了しました。次は Todo #6（Strings多様化ペナルティ）に進みます。
