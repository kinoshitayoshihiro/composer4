# Guitar/Bass/Strings Stage2 Metrics Test Results

テスト日: 2025年10月17日  
テスト件数: 各楽器100ファイル  
データセット: SLAKH2100 (Stage1 clean)

## テスト結果サマリー

| 楽器 | 平均スコア | 中央値 | 合格率 | 閾値 | 総合評価 |
|------|-----------|--------|--------|------|----------|
| **Guitar** | 43.6% | 47.0% | 68% | 40.0 | ✅ 良好 |
| **Bass** | 76.7% | 76.6% | 100% | 40.0 | 🏆 優秀 |
| **Strings** | 50.9% | 55.6% | 70% | 45.0 | ✅ 良好 |

## 詳細メトリクス

### Guitar (アルペジオ・コード・ストラム評価)

**全体統計:**
- テスト件数: 100ファイル
- 加重平均スコア: 0.436 (43.6%)
- 中央値スコア: 0.470 (47.0%)
- 合格: 68ファイル (68%)
- 不合格: 32ファイル (32%)

**メトリクス別スコア:**

| メトリクス | 平均 | 標準偏差 | 最小 | 中央値 | 最大 |
|-----------|------|---------|------|--------|------|
| **arpeggio_quality** (アルペジオ品質) | 0.461 | 0.091 | 0.283 | 0.440 | 0.756 |
| **chord_coherence** (コード協和度) | 0.478 | 0.271 | 0.000 | 0.559 | 1.000 |
| **strumming_pattern** (ストラム規則性) | 0.325 | 0.289 | 0.000 | 0.419 | 0.850 |

**TOP 5:**
1. Track00410_S01.mid: 71.19%
2. Track00986_S02.mid: 69.29%
3. Track00192_S06.mid: 68.03%
4. Track01465_S04.mid: 65.02%
5. Track00385_S06.mid: 64.17%

**評価:**
- ✅ アルペジオパターン検出が適切に機能
- ✅ コード協和度評価が幅広い範囲で分布
- ⚠️ ストラムパターンは25%が0点(検出なし)
- ✅ 閾値40.0は適切(68%が合格)

---

### Bass (ルート音・グルーヴ・音域評価)

**全体統計:**
- テスト件数: 100ファイル
- 加重平均スコア: 0.767 (76.7%)
- 中央値スコア: 0.766 (76.6%)
- 合格: 100ファイル (100%) 🎉
- 不合格: 0ファイル (0%)

**メトリクス別スコア:**

| メトリクス | 平均 | 標準偏差 | 最小 | 中央値 | 最大 |
|-----------|------|---------|------|--------|------|
| **root_accuracy** (ルート音正確性) | 0.841 | 0.073 | 0.716 | 0.820 | 1.000 |
| **groove_quality** (グルーヴ品質) | 0.642 | 0.079 | 0.437 | 0.624 | 0.839 |
| **pitch_range_fit** (音域適合性) | 0.868 | 0.149 | 0.187 | 0.911 | 0.959 |

**TOP 5:**
1. Track01465_S05.mid: 86.62%
2. Track01275_S05.mid: 86.26%
3. Track00986_S01.mid: 84.50%
4. Track01217_S02.mid: 84.38%
5. Track01217_S01.mid: 84.38%

**評価:**
- 🏆 全ファイル合格(100%)の優秀な結果
- ✅ ルート音検出が極めて高精度(84.1%)
- ✅ 音域適合性が高い(86.8%)
- ✅ グルーヴ評価も安定(64.2%)
- ✅ 閾値40.0は十分緩い(全て合格)

---

### Strings (ボウイング・ハーモニー・レガート評価)

**全体統計:**
- テスト件数: 100ファイル
- 加重平均スコア: 0.509 (50.9%)
- 中央値スコア: 0.556 (55.6%)
- 合格: 70ファイル (70%)
- 不合格: 30ファイル (30%)

**メトリクス別スコア:**

| メトリクス | 平均 | 標準偏差 | 最小 | 中央値 | 最大 |
|-----------|------|---------|------|--------|------|
| **bowing_expression** (ボウイング表現) | 0.479 | 0.193 | 0.150 | 0.510 | 0.770 |
| **harmony_quality** (ハーモニー品質) | 0.585 | 0.315 | 0.000 | 0.670 | 1.000 |
| **legato_quality** (レガート品質) | 0.419 | 0.159 | 0.039 | 0.407 | 0.972 |

**TOP 5:**
1. Track01076_S08.mid: 75.13%
2. Track01920_S00.mid: 74.85%
3. Track01551_S03.mid: 72.67%
4. Track01518_S08.mid: 72.10%
5. Track00754_S09.mid: 72.01%

**評価:**
- ✅ ハーモニー評価が高い(58.5%)
- ✅ ボウイング表現検出が機能(47.9%)
- ⚠️ レガート検出がやや低め(41.9%)
- ✅ 閾値45.0は適切(70%が合格)
- 💡 レガート検出の改善余地あり

---

## 結論と推奨事項

### ✅ 実装成功

全3楽器でメトリクス実装が成功し、適切な評価が可能であることを確認:

1. **Guitar**: アルペジオ・コード・ストラムの多様な演奏スタイルを評価
2. **Bass**: ルート音・グルーヴ・音域の基本要素を高精度に評価
3. **Strings**: ボウイング・ハーモニー・レガートの表現力を評価

### 📊 スコア分布の妥当性

- **Bass**: 最も高スコア(平均76.7%) → ルート音中心の単純な構造
- **Strings**: 中程度(平均50.9%) → 和音とレガートの複雑性
- **Guitar**: やや低め(平均43.6%) → ストラムパターン検出の課題

### 🎯 閾値設定の妥当性

現在の閾値は適切:
- Guitar: 40.0 → 68%合格
- Bass: 40.0 → 100%合格
- Strings: 45.0 → 70%合格

### 💡 今後の改善提案

1. **Guitarストラムパターン検出の強化**
   - 現在25%が検出なし(0点)
   - ストラム検出ロジックの改善

2. **Stringsレガート検出の精度向上**
   - duration情報が不足している場合の対応
   - オーバーラップ検出の改善

3. **閾値の微調整(オプション)**
   - 現状で十分だが、より厳格な選別が必要なら:
     - Guitar: 40.0 → 45.0 (合格率 55%程度)
     - Strings: 45.0 → 50.0 (合格率 55%程度)

### 🚀 次のステップ

1. ✅ **テスト完了**: 3楽器すべてでメトリクス動作確認済み
2. ⏭️ **ドキュメント更新**: MULTI_DATASET_RUNNER_GUIDE.mdに結果追記
3. ⏭️ **本番実行準備**: 全データ(Guitar 1,422件, Bass 584件, Strings 999件)での実行

---

## 技術的詳細

### メトリクス実装方式

**簡易実装 (現状):**
- `scripts/stage2_instrument_metrics.py`で楽器別メトリクスを実装
- MIDI解析による基本的なパターン検出
- YAMLベースの設定管理

**将来の完全統合 (TODO):**
- LAMDAコアエンジンへの統合
- Audio-adaptive weights対応
- Retry presets対応

### 設定ファイル構造

```yaml
score:
  axes:
    # 楽器固有のメトリクス軸
    arpeggio_quality: 2.0      # Guitar
    root_accuracy: 2.0         # Bass
    bowing_expression: 2.0     # Strings
  
  # 各軸の詳細設定
  arpeggio_quality:
    min_notes: 4
    weights:
      pattern_consistency: 0.40
      interval_regularity: 0.30
      timing_precision: 0.30
```

### テストスクリプト

```bash
# Guitar
python scripts/test_instrument_metrics.py \
  --instrument guitar \
  --input-dir output/slakh/clean/guitar \
  --config configs/lamda/guitar_stage2.yaml \
  --max-files 100 \
  --output-json output/test_results/guitar_test.json

# Bass
python scripts/test_instrument_metrics.py \
  --instrument bass \
  --input-dir output/slakh/clean/bass \
  --config configs/lamda/bass_stage2.yaml \
  --max-files 100 \
  --output-json output/test_results/bass_test.json

# Strings
python scripts/test_instrument_metrics.py \
  --instrument strings \
  --input-dir output/slakh/clean/strings \
  --config configs/lamda/strings_stage2.yaml \
  --max-files 100 \
  --output-json output/test_results/strings_test.json
```

---

**生成日時**: 2025年10月17日  
**テスト環境**: macOS, Python 3.11, SLAKH2100 clean data  
**スクリプト**: scripts/test_instrument_metrics.py v1.0
