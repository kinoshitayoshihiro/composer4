# Production Hardening Phase 2: 実運用レベルへの最終詰め

**作成日**: 2025年10月18日  
**目標**: "使える初版" → "商用運用レベル" への昇格

---

## 📊 現状評価（Chat レビュー結果）

### ✅ 強み（Keep）

- **再現性・トレーサビリティ**: seed一本化、provenance.json、datasets.lock完備
- **宣言的アレンジ**: structure.yamlでセクション別奏法オーバーライド可能
- **Drums Stage2初版**: GM準拠、フォールバック、5/5テスト合格

### ⚠️ 改善点（Tweak）

1. **ドラムパターンバンクの薄さ**: 実データ1,000〜3,000件必要
2. **ハイハット開閉整合**: Open/Closedの排他制御不足
3. **ダイナミクス上限**: ベロシティ123上限ガード未実装
4. **多様性ペナルティ**: 楽器別最適化未対応
5. **境界テスト不足**: テンポ変化・奇数小節・3/4拍子

### 🆕 追加実装（Add）

- **Drums品質ゲート**: kick_onbeat_ratio_min等
- **Suno抽出信頼度ログ**: セクション置信度、拍同期ロス
- **60秒CIパイプライン**: YAML→MIDI→WAV→ゲート検証

---

## 🎯 優先度付きアクションプラン

### 🔴 最優先（リリース品質直結）

#### Todo #2: オーディオ出力の堅牢化
- [ ] **SF2固定・ハッシュ記録** (2h)
  - `soundfonts.lock` 作成（SF2ファイルのSHA256）
  - レンダリング時にハッシュ検証
  - バージョン不一致時に警告
  
- [ ] **-1.0dBFS正規化の強制** (1h)
  - `dawdreamer_batch.py`で正規化必須化
  - ピーク検出とクリッピング率計算
  - `clipping_rate < 0.1%` をゲート条件に
  
- [ ] **並列リカバリ機能** (3h)
  - 失敗MIDIのID記録（`failed_renders.jsonl`）
  - `--resume`フラグで失敗分のみ再実行
  - タイムアウト設定（デフォルト60秒/ファイル）

#### Todo #3: ドラムパターンバンク充実（最重要）
- [ ] **SLAKH/LAMDA抽出強化** (4h)
  - BPM帯で層化（60-90/90-120/120-150/150-180）
  - 拍子別分類（4/4/3/4/6/8）
  - 最低1,000件、目標3,000件
  
- [ ] **品質フィルタ強化** (2h)
  - On-beat率 ≥ 0.6（キックが拍頭に来る）
  - ゴースト比率 ≤ 0.3（弱打が多すぎない）
  - 密度範囲 4〜32 notes/bar

### 🟡 次点（多様性・品質向上）

#### Todo #4: Stage2推薦の多様化
- [ ] **楽器別diversity_penalty最適化** (2h)
  - Strings: 0.3（高め、同質化防止）
  - Bass: 0.15（低め、安定性重視）
  - Guitar: 0.2（バランス）
  - Drums: 0.25（中高め）
  
- [ ] **Strings特徴量再スコア** (3h)
  - 持続時間（legato指標）
  - 速度分散（tremolo指標）
  - 跳躍率（pizzicato指標）
  - 既存類似度と統合（0.7×類似度 + 0.3×特徴適合度）

#### Todo #5: Drums品質ゲート追加
- [ ] **quality_gates.drumsをYAMLに追加** (1h)
  ```yaml
  quality_gates:
    drums:
      kick_onbeat_ratio_min: 0.6
      ghost_note_ratio_max: 0.3
      notes_per_bar_range: [4, 32]
      crash_per_section_max: 4
      hihat_open_closed_gap_min_ms: 50
  ```
  
- [ ] **ゲート検証を arrange_from_yaml.py に統合** (2h)

#### Todo #6: ハイハット・クラッシュの整合性
- [ ] **Open/Closed排他制御** (2h)
  - 同一タイミング±50msで重複検出
  - Openを優先（Closedをミュート）
  
- [ ] **クラッシュチョーク実装** (1h)
  - クラッシュ後200ms以内の他シンバルをチョーク
  - チョーク長を80msに制限

### 🟢 継続改善（中長期）

#### Todo #7: 60秒CIパイプライン
- [ ] **最小E2Eテスト作成** (3h)
  - 固定YAML（4セクション、16小節、120BPM）
  - 固定SF2（FluidR3_GM.sf2）
  - 目標実行時間: 60秒以内
  - 検証項目:
    - 拍数一致
    - テンポ±1BPM
    - クリッピング < 0.1%
    - 全楽器MIDI/WAV生成成功
  
- [ ] **datasets.lock検証をCI統合** (1h)
  - `python scripts/verify_datasets.py`
  - ハッシュ不一致でfail

#### Todo #8: Suno抽出の信頼度ログ
- [ ] **信号品質インジケータ追加** (3h)
  - セクション置信度（オンセット検出の確信度）
  - 拍同期ロス（テンポ推定の分散）
  - キー一致率（コード推定の一貫性）
  
- [ ] **低信頼セクションの保守的設定** (2h)
  - 置信度 < 0.7 → diversity_penalty +0.1
  - 置信度 < 0.5 → faithfulness +0.2（原曲忠実度を上げる）

#### Todo #9: 境界条件テスト追加
- [ ] **テンポ変化テスト** (1h)
  - tempo_map付きYAML
  - 120→140BPMへの加速
  
- [ ] **奇数小節テスト** (1h)
  - 5小節Intro、7小節Bridge等
  
- [ ] **3/4拍子テスト** (1h)
  - ワルツ、6/8拍子対応確認

---

## 📅 実装スケジュール（3日間）

### Day 1（今日）: 🔴最優先
- [ ] Todo #2: オーディオ堅牢化（6h）
- [ ] Todo #3: ドラムパターン抽出開始（4h）

### Day 2: 🔴最優先 + 🟡次点
- [ ] Todo #3: ドラムパターン検証・統合（2h）
- [ ] Todo #4: 推薦多様化（5h）
- [ ] Todo #5: Drums品質ゲート（3h）

### Day 3: 🟡次点 + 🟢継続
- [ ] Todo #6: ハイハット整合性（3h）
- [ ] Todo #7: 60秒CI（4h）
- [ ] Todo #8〜#9: 余裕があれば着手

---

## 🎯 成功指標

### 定量目標

- **ドラムパターン**: 1,000件以上（目標3,000件）
- **クリッピング率**: 全出力で < 0.1%
- **CI実行時間**: 60秒以内
- **パターン多様性**: 同一奏法の連続 ≤ 3回
- **品質ゲート合格率**: ≥ 95%

### 定性目標

- **再現性**: 同一seed→完全同一出力
- **堅牢性**: 10曲連続レンダリングでfail 0件
- **多様性**: A/B比較で明確な差異を体感
- **運用性**: エラー時の復旧手順が明確

---

## 📝 実装メモ

### ベロシティ上限ガード実装例

```python
# generator/base_generator.py
MAX_VELOCITY = 123  # -1dBFS余裕確保

def _normalize_velocity(self, vel: int) -> int:
    """ベロシティを安全範囲に正規化"""
    return min(vel, MAX_VELOCITY)
```

### SF2ハッシュ検証例

```python
# scripts/render/soundfont_manager.py
import hashlib

def verify_soundfont(sf2_path: Path, expected_hash: str) -> bool:
    """SF2ファイルのSHA256検証"""
    sha256 = hashlib.sha256()
    with open(sf2_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if actual != expected_hash:
        print(f"⚠️  SF2 hash mismatch: {actual[:8]}... != {expected_hash[:8]}...")
        return False
    return True
```

### 並列リカバリ例

```bash
# 初回実行（3/10失敗）
python scripts/dawdreamer_batch.py --midi-dir out/midi --out-audio out/wav
# → failed_renders.jsonl に失敗ID記録

# リトライ（失敗分のみ）
python scripts/dawdreamer_batch.py --midi-dir out/midi --out-audio out/wav --resume
# → failed_renders.jsonl から読み込み、3件のみ再実行
```

---

## 🚀 次のアクション

1. **Todo #2開始**: SF2ハッシュ管理 → 正規化強制 → リカバリ機能
2. **Todo #3並行**: ドラムパターン抽出スクリプト修正・実行
3. **進捗確認**: 各Todo完了時にこのドキュメント更新

**目標**: Phase 2完了後に「商用運用可能」と宣言できる状態へ！
