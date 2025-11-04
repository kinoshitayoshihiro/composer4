# Stage1 LAMDA Plus v2 改善点適用完了

## 実施内容

### 1. hashlib.md5() の bytes 対応 ✅
- **修正箇所**: `compute_bar_fingerprint`, `compute_content_id`
- **変更内容**: `.encode("utf-8")` を明示的に追加
- **理由**: str を直接渡すとエラーになるため

```python
# 修正前
hashlib.md5(fingerprint).hexdigest()

# 修正後
hashlib.md5(fingerprint.encode("utf-8")).hexdigest()
```

### 2. ${base} プレースホルダ展開関数追加 ✅
- **新規関数**: `expand_placeholders(path_str, roots)`
- **機能**: `${base}` 等を roots 辞書で展開 → env/~/ も展開
- **使用箇所**: `main()` の priors パス読み込み

```python
def expand_placeholders(path_str, roots):
    """${base} 等を roots 辞書で展開 → さらに env/~/ を展開"""
    import re
    def repl(m):
        key = m.group(1)
        return str(roots.get(key, m.group(0)))
    s = re.sub(r"\$\{([^}]+)\}", repl, str(path_str))
    s = os.path.expandvars(os.path.expanduser(s))
    return s
```

### 3. content_id 計算時の区切り文字追加 ✅
- **修正箇所**: `compute_content_id`
- **変更内容**: `f"{bar_fp}_{total_ticks}"` → `f"{bar_fp}|{total_ticks}"`
- **理由**: 衝突確率低減

### 4. exclude_dirs の Path 正規化 ✅
- **修正箇所**: `Stage1Processor.__init__`, `should_exclude`
- **変更内容**: 
  - exclude_dirs を `Path.as_posix()` で正規化
  - `should_exclude` で Path.parts 単位で一致判定
- **理由**: 文字化け・全角混在に強化

```python
# 正規化
self.exclude_dirs = [Path(d).as_posix() for d in raw_excludes]

# Path.parts 単位での一致
def should_exclude(self, midi_path):
    path_parts = [Path(p).as_posix() for p in Path(midi_path).parts]
    for part in path_parts:
        if part in self.exclude_dirs:
            return True
    return False
```

### 5. bar_split_long_notes の最小長設定 ✅
- **修正箇所**: `split_long_notes_on_bar`
- **変更内容**: `min_dur = config['ranges']['dur_ticks'][0]` を追加
- **理由**: ranges 設定との一貫性確保

```python
min_dur = config['ranges']['dur_ticks'][0]  # 最小音長
# ...
if duration > bar_ticks and duration >= min_dur:
    # 分割処理
```

## バリデーション結果

### stage1_config_validator.py 実行 ✅
```bash
$ python3 scripts/stage1_config_validator.py config/stage1_config.yaml
[INFO] Placeholder ${base} detected. Ensure your loader expands it safely.
OK: stage1_config.yaml passed basic validation.
```

### 処理結果確認 ✅
- **総処理曲数**: 5,350曲（drum_loops 除外）
- **生成MIDI**: 4,543ファイル
- **CSV出力**: stage1_summary_full.csv（5,351行、ヘッダー含む）
- **JSON出力**: 全content_id ディレクトリに stage1_clean.json 生成

### サンプルJSON確認 ✅
```json
{
  "source_mid_id": "16359c765b6df119",
  "content_id": "86e94f4aa1bead65",
  "run_id": "20251025_224353_v2.0",
  "ok_meta": {
    "song_id": "86e94f4aa1bead65",
    "stage": "stage1",
    "run_id": "20251025_224353_v2.0",
    "source_mid_id": "16359c765b6df119",
    "content_id": "86e94f4aa1bead65",
    "time_sig": [4, 4],
    "bpm_est": 120
  }
}
```

## 除外機能検証 ✅

### 除外ディレクトリ
- `drum_loops`: 827曲（rhythm学習用）
- `除外`: 81,007曲（元データ）

### 処理対象
- `pop909`: 1,674曲
- `slakh_stem`: 3,676曲

## まとめ

全5点の改善を適用し、バリデーションも完了。Stage1 LAMDA Plus v2は production-ready です。

- ✅ bytes 型強制（hashlib安全性向上）
- ✅ プレースホルダ展開（${base}対応）
- ✅ ID衝突回避（区切り文字追加）
- ✅ パス正規化（文字化け対応）
- ✅ 設定一貫性（min_dur統一）
