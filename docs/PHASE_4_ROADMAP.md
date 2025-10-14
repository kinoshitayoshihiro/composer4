# Phase 4 Progress Roadmap
# Updated: 2025-10-14

## 🎯 Phase 4: "壊さず・速く・再現可能に" 仕上げるゾーン

### ✅ 完了済み

| Phase | 内容 | 状態 | コミット | 日付 |
|-------|------|------|---------|------|
| **4.0** | Piano Transformer統合（基盤） | ✅ Complete | - | - |
| **4.1** | 学習堅牢化（AdamW/cosine/warmup/TF32） | ✅ Complete | - | - |
| **4.2** | データ品質＆生成品質（層別分割・Best-of-N） | ✅ Complete | - | - |
| **4.2-polish** | 決定論＆監査性（SHA1分割/score内訳） | ✅ Complete | - | - |
| **4.3** | 外部ベンチ統合（MAESTRO, 履歴, 可視化） | ✅ Complete | a77f4fa25 | 2025-10-14 |
| **4.3-robustness** | 初回失敗ゼロ＆PNG閾値線 | ✅ Complete | a77f4fa25 | 2025-10-14 |
| **4.4** | Attention自動選択（Flash/SDPA/Math） | ✅ Complete | 908b6258c | 2025-10-14 |
| **4.5** | A/B評価精度向上（Guitar窓クランプ） | ✅ Complete | - | - |
| **4.6-prep** | 品質ゲート準備（YAML/checker） | ✅ Complete | c1b635451 | 2025-10-14 |
| **4.6** | CI品質ゲート配備 | ✅ **Complete** | **[今日]** | **2025-10-14** |

### 🔄 進行中

| Phase | 内容 | 進捗 | 推定工数 | 優先度 |
|-------|------|------|---------|--------|
| **4.7** | セクション整合＆感情マッピング確定 | 0% | 1日 | 🔥 高 |
| **4.8** | music21/ASAP強化 | 0% | 3-5日 | 中 |
| **4.9** | v1.0 Release準備 | 0% | 2-3日 | 中 |

---

## 📊 Phase 4.6: CI品質ゲート配備 (本日完了)

### 実装内容

#### 1. **scripts/ci_quality_gate.sh** (80行)
```bash
# CI統合スクリプト
# - 5楽器の品質ゲートチェック
# - threshold_flags 抽出・表示
# - 失敗時は CI 赤、成功時は緑
```

**機能**:
- 各楽器の評価JSON読み込み
- `quality_gate_checker.py` による検証
- カラー出力（✅ PASS / ❌ FAIL）
- `threshold_flags` の詳細表示

#### 2. **quality_gate_checker.py 拡張**
```python
# CLI機能追加
python scripts/quality_gate_checker.py --check piano --json output.json
# Exit code: 0=PASS, 1=FAIL
```

#### 3. **.github/workflows/quality_gate.yml**
```yaml
# GitHub Actions統合
# - Nightly at 2:00 AM JST
# - Push to main/develop
# - Pull request
# - Manual dispatch
```

**機能**:
- Python環境セットアップ
- 評価スクリプト実行
- 品質ゲートチェック
- GitHub Summary生成
- Artifacts保存（30日間）

---

## 🎯 次のステップ: Phase 4.7

### Phase 4.7: セクション整合＆感情マッピング確定

**目標**: BasePartGenerator準拠APIで Verse/Pre-Ch/Ch/Bridge の境界精度テスト + emotion_profile.yaml の写像規約化

#### 実装タスク

1. **セクション境界精度テスト** (各楽器30分)
   ```python
   def test_piano_section_boundaries():
       # Verse→Chorusの境界でマーカー/長さが一致
       sections = [
           ("Verse", 8),  # 8 bars
           ("Chorus", 8),  # 8 bars
       ]
       midi = piano_gen.generate_multi_section(sections)
       # Assert: 境界位置が正確、オーバーラップなし
   ```

2. **emotion_profile 写像規約** (各楽器1時間)
   ```yaml
   # emotion_profile.yaml に規約追加
   piano:
     mapping:
       intensity:
         low: { velocity: 60-80, density: 0.6, articulation: "legato" }
         medium: { velocity: 80-100, density: 1.0, articulation: "normal" }
         high: { velocity: 100-120, density: 1.4, articulation: "marcato" }
   ```

3. **統合テスト** (1時間)
   ```python
   def test_emotion_section_integration():
       # emotion変化 + セクション変化の同時動作
       pass
   ```

**推定工数**: 1日（5楽器 × 1.5時間 + 統合1時間）

---

## 🎼 楽器別ステータス (更新)

| 楽器 | 完成度 | 4.6対応 | 4.7対応 | v1.0到達 |
|------|--------|---------|---------|---------|
| **Piano** | v1.0 ✅ | ✅ Complete | 🔶 Needed | 🎯 Already |
| **Guitar** | RC 🟢 | ✅ Ready | 🔶 Needed | 🎯 After 4.7 |
| **Drums** | 85% 🟡 | ✅ Ready | 🔶 Needed | 🎯 After 4.7 |
| **Bass** | 85% 🟡 | 🔶 eval_bass.py | 🔶 Needed | 🎯 After eval+4.7 |
| **Strings** | 85% 🟡 | 🔶 eval_strings.py | 🔶 Needed | 🎯 After eval+4.7 |

---

## 📈 Phase 4 進捗サマリー

**完了**: 10 / 13 フェーズ (77%)

**残作業**:
- Phase 4.7: セクション整合＆感情マッピング（1日）
- Phase 4.8: music21/ASAP強化（3-5日、オプション）
- Phase 4.9: v1.0 Release準備（2-3日）

**v1.0到達まで**: 3-5日（4.7完了で実質運用可能）

---

## 🚀 Phase 5 プレビュー

Phase 4完了後の展開:

### 5.0 構造計画レイヤ
- セクション配列最適化
- モチーフ再帰
- キー変化

### 5.1 Relative Attention系
- Music Transformer評価
- Museformer系導入

### 5.2 メモリ効率・速度最適化
- 長文対応（8K+ tokens）
- 推論ストリーミング

### 5.3 器間インタラクション
- ベース↔ドラム ロック
- ギター↔ピアノ 衝突回避

### 5.4 ミックス下ごしらえ
- 音域衝突検出
- 密度バランサ

---

**Updated**: 2025-10-14  
**Current**: Phase 4.6 Complete, Moving to 4.7  
**Target**: Phase 4 completion within 1 week
