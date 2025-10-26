# LAMDA クリーニングシステム - 改善実装 (2025-10-18)

## 🎯 改善目的

**最小変更で運用堅牢性を向上**
- 表記ゆれ解消
- 誤カウント防止
- 自動化強化

---

## ✅ 実装した改善（3点）

### 1. 表記ゆれの一本化（LAMDA / LAMIDI）

#### 問題
- ドキュメント: `run_lamda_full.sh` / `monitor_lamda.sh`
- 実ファイル: 混在していた

#### 解決策
**薄ラッパーによる透過的リダイレクト**

```bash
# scripts/monitor_lamidi.sh → scripts/monitor_lamda.sh へリダイレクト
#!/bin/bash
exec "$(dirname "$0")/monitor_lamda.sh" "$@"
```

**効果:**
- ✅ どちらの名前で呼んでも動作
- ✅ 既存スクリプトを変更せず（リスク最小）
- ✅ ドキュメントと実装が一致

---

### 2. モニタースクリプトの改善

#### 改善A: 誤カウント防止

**Before:**
```bash
CLEANED=$(find "${CLEAN_DIR}" -name "*.mid" 2>/dev/null | wc -l)
```

**After:**
```bash
CLEANED=$(find "${CLEAN_DIR}" -type f -name "*.mid" 2>/dev/null | wc -l)
```

**効果:**
- ✅ `-type f` でファイルのみカウント
- ✅ ディレクトリ名や壊れたリンクを除外
- ✅ 正確な進捗表示

#### 改善B: EXPECTED_TOTAL自動読込

**Before:**
```bash
EXPECTED_TOTAL=10000  # 手動設定
```

**After:**
```bash
# 環境変数 or ファイルから自動読込
: "${EXPECTED_TOTAL:=}"
if [ -z "${EXPECTED_TOTAL}" ] && [ -f "${BASE_DIR}/data/lamda_expected_total.txt" ]; then
  EXPECTED_TOTAL="$(cat "${BASE_DIR}/data/lamda_expected_total.txt" 2>/dev/null || echo "")"
fi
# デフォルト値
: "${EXPECTED_TOTAL:=404714}"
```

**効果:**
- ✅ `check_lamidi_dataset.sh`実行後に自動反映
- ✅ 手動編集不要
- ✅ 環境変数でも上書き可能

---

### 3. 楽器名マッピング機能

#### 問題
- `run_dataset_full.sh`: `LAMDA_PIANO`等のデータセット名でフィルタ
- ドキュメント: `piano`, `strings`等の楽器名を案内

#### 解決策
**ラッパー側で楽器名→データセット名変換**

```bash
map_inst() {
  case "$1" in
    piano)   echo "LAMDA_PIANO"   ;;
    strings) echo "LAMDA_STRINGS" ;;
    guitar)  echo "LAMDA_GUITAR"  ;;
    bass)    echo "LAMDA_BASS"    ;;
    drums)   echo "LAMDA_DRUMS"   ;;
    LAMDA_*) echo "$1" ;;  # 既にLAMDA_*なら素通し
    *)       echo "$1" ;;
  esac
}

# 引数をマッピング
for arg in "$@"; do
  MAPPED=$(map_inst "$arg")
  if [[ "$MAPPED" =~ ^LAMDA_ ]]; then
    INSTRUMENTS+=("$MAPPED")
  fi
done
```

**効果:**
- ✅ `./scripts/run_lamda_full.sh piano` → `LAMDA_PIANO`に自動変換
- ✅ `./scripts/run_lamda_full.sh LAMDA_PIANO`も動作（後方互換）
- ✅ ドキュメント通りのコマンドで実行可能

---

## 🔄 連携フロー

### Before（手動・エラー誘発）
```
1. check_lamidi_dataset.sh 実行
   → 「EXPECTED_TOTALをXXXに更新してください」と表示
2. ユーザーがmonitor_lamidi.shを手動編集
   → タイポ・更新忘れのリスク
3. monitor_lamidi.sh 実行
   → 誤った進捗表示の可能性
```

### After（自動・堅牢）
```
1. check_lamidi_dataset.sh 実行
   → 総ファイル数をdata/lamda_expected_total.txtに自動保存
2. monitor_lamda.sh 実行
   → ファイルから自動読込
   → 正確な進捗表示（-type fで誤カウント防止）
```

---

## 📊 ファイル構成

### メインスクリプト
```
scripts/
├── run_lamda_full.sh        # メイン実行（楽器名マッピング対応）
├── monitor_lamda.sh         # 本体モニター（改善版）
├── monitor_lamidi.sh        # 薄ラッパー → monitor_lamda.sh
└── check_lamidi_dataset.sh  # 総数自動保存対応
```

### 自動生成ファイル
```
data/
└── lamda_expected_total.txt  # check実行時に自動生成
```

---

## 🚀 使い方（改善後）

### 1. データセット確認
```bash
./scripts/check_lamidi_dataset.sh
```
**自動で行われること:**
- MIDIファイル総数カウント
- `data/lamda_expected_total.txt`に保存
- 次のコマンド提案

### 2. 実行
```bash
# 楽器名で指定（推奨）
./scripts/run_lamda_full.sh piano
./scripts/run_lamda_full.sh guitar bass

# データセット名でも可（後方互換）
./scripts/run_lamda_full.sh LAMDA_PIANO LAMDA_GUITAR
```

### 3. 進捗監視
```bash
# どちらでも動作（薄ラッパーで統一）
./scripts/monitor_lamda.sh
./scripts/monitor_lamidi.sh  # → monitor_lamda.shへリダイレクト
```
**自動で行われること:**
- `data/lamda_expected_total.txt`から総数読込
- `-type f`で正確なファイル数カウント
- 各楽器の進捗を一覧表示

---

## 🎯 品質ゲート強化（次ステップ提案）

### Guitar: アルペジオ偏重の是正

**Stage2スコアリングに合成指標を導入:**

```python
def guitar_quality_score(pattern):
    """
    ギターパターンの品質スコア
    
    指標:
    - simultaneity_index: 同時発音の密集度（低いほどストラム的）
    - triad_plus_rate: 3和音以上の割合
    - arp_density: アルペジオ密度（高いほどアルペジオ的）
    """
    w1 = 0.4  # ストラム重視
    w2 = 0.3  # コード構造重視
    w3 = 0.3  # アルペジオ抑制
    
    score = (
        w1 * (1.0 - normalize(pattern.simultaneity_index)) +  # 低いほど高得点
        w2 * pattern.triad_plus_rate +
        w3 * (1.0 - pattern.arp_density)  # 低いほど高得点
    )
    
    return score

# 上位パーセンタイルのみ採用
threshold = np.percentile(scores, 70)  # 上位30%
filtered_patterns = [p for p, s in zip(patterns, scores) if s >= threshold]
```

**実装場所:**
- `scripts/lamda_stage2_extractor.py` にギター専用スコアリング追加
- `scripts/extract_guitar_patterns.py` で品質ゲート適用

---

## 🔧 トラブルシューティング

### Q: `data/lamda_expected_total.txt`が存在しない
**A:** `check_lamidi_dataset.sh`を実行すれば自動生成されます。

### Q: 手動で総数を設定したい
**A:** 環境変数で上書き可能：
```bash
EXPECTED_TOTAL=500000 ./scripts/monitor_lamda.sh
```

### Q: 旧コマンド（`monitor_lamidi.sh`）は使える？
**A:** はい。薄ラッパーで自動的に新版へリダイレクトされます。

---

## 📈 改善効果

### 堅牢性向上
- ✅ 表記ゆれによるヒューマンエラー防止
- ✅ 誤カウント防止（-type f）
- ✅ 手動編集不要（自動化）

### 運用改善
- ✅ コマンドがシンプル（楽器名で指定可能）
- ✅ 進捗表示が正確
- ✅ ドキュメントと実装が一致

### 保守性向上
- ✅ 最小変更（既存スクリプト無改造）
- ✅ 後方互換（旧コマンドも動作）
- ✅ 拡張容易（map_inst関数で楽器追加可能）

---

## 🎯 次の実装推奨

### 短期（即座に効果）
1. ✅ **完了**: 薄ラッパー・自動読込・楽器名マッピング
2. ⏭️ **Guitar品質スコア**: simultaneity_index + triad_plus_rate - arp_density
3. ⏭️ **BPM層化パラメトリック化**: `config/pattern_quality.yaml`で調整可能に

### 中期（運用安定化）
4. ⏭️ **進捗ダッシュボード**: 全楽器の処理速度・予想完了時刻を表示
5. ⏭️ **自動リカバリ**: エラー時の自動リトライ機能
6. ⏭️ **品質レポート**: クリーニング後の統計サマリー自動生成

---

## 📚 関連ドキュメント

- `docs/LAMDA_QUICKSTART.md` - クイックスタートガイド
- `docs/LAMDA_EXECUTION_GUIDE.md` - 詳細実行手順
- `docs/LAMIDI_CLEANING_GUIDE.md` - クリーニング詳細
- `scripts/cleaners/guitar.py` - Guitar品質ゲート実装

---

**実装完了日**: 2025年10月18日  
**最小変更の原則**: 既存動作に影響せず、堅牢性のみ向上
