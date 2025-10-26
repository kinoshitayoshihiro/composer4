# Markov Stage2 パッチ第二弾 - 設定ガイド

## 概要

`generator/drums_generator_stage2.py` の Markov ボイス切替システムに以下の4機能を追加：

1. **prob_bounds** - フロア/キャップ（確率の上下限）
2. **preempt** - セクション直前プリエンプト（距離比例）
3. **sticky** - 滞在バイアス（連続スイッチ抑制）
4. **seed** - 決定論的シード（再現性）

すべて **未設定時はNO-OP**（既存の挙動は変わりません）。

---

## 完全な設定例

```yaml
drums_params:
  ride:
    ride_markov:
      enable: true
      start: ride1
      cooldown_china_hits: 2

      # ========================================
      # 1) prob_bounds: 確率の上下限
      # ========================================
      prob_bounds:
        floor: { ride1: 0.02 }   # ride1が完全に消えないよう保険
        cap:   { china: 0.60 }   # chinaが出過ぎないよう上限
        
        # セクション別上書き
        per_section:
          verse:  
            floor: { ride1: 0.10 }
            cap:   { china: 0.20 }  # バースでは控えめ
          chorus: 
            floor: { ride2: 0.10 }
            cap:   { ride1: 0.70 }  # サビではride2を推奨

      # ========================================
      # 2) preempt: セクション直前プリエンプト
      # ========================================
      preempt:
        enable: true
        ahead_ms: 150         # セクション境界の150ms前から適用
        # または ahead_ql: 0.5  # クォーター長で指定も可能
        mode: add             # add: 加算, mul: 乗算
        
        # 距離比例の形状
        falloff: 
          mode: ease          # linear: 線形, ease: S字カーブ, exp: 指数
          window_ms: 150      # フォールオフウィンドウ
          # または window_ql: 0.5
        
        # セクション別の遷移ブースト
        sections:
          chorus: 
            add: { ride2: +0.25 }  # サビ前にride2を押す
            head: bell             # Bell時のみ適用（any: 全頭）
          bridge: 
            add: { china: +0.50 }  # ブリッジ前にchinaを押す

      # ========================================
      # 3) sticky: 滞在バイアス
      # ========================================
      sticky:
        enable: true
        min_hits: 1           # 最低ヒット数（この回数まで強制維持）
        self_boost: 0.10      # その後の自己ブースト量
        
        # Head別の細分化
        per_head:
          bell: 
            min_hits: 2       # Bell時は最低2打
            self_boost: 0.10
          bow:  
            min_hits: 1
            self_boost: 0.05
        
        # State別の細分化
        per_state:
          china: 
            min_hits: 2       # Chinaは最低2打（連打抑制）
            self_boost: 0.00  # その後は寄せない

      # ========================================
      # 4) seed: 決定論的シード
      # ========================================
      seed:
        base: 12345           # ベースシード値
        per_section: true     # セクション名でシード変動

      # ========================================
      # 既存機能: energy_blend
      # ========================================
      energy_blend:
        mode: mul             # mul: 乗算, add: 加算
        alpha: 0.5            # 効きの強さ
        bias: 0.0             # バイアス
        clamp: [0.6, 1.8]     # スケール係数の範囲（mul時）
        
        sensitivity:          # 状態ごとの感度
          ride1: -0.20        # E↑で抑える
          ride2: +0.10        # E↑で少し押す
          china: +0.45        # E↑で強く押す
        
        per_head:             # Head別上書き
          bell: { china: +0.20 }
        
        per_section:          # セクション別上書き
          chorus:
            alpha: 0.7
            clamp: [0.7, 2.0]
            sensitivity: { ride2: +0.20, china: +0.55 }
            per_head: { bow: { china: +0.10 } }
        
        smooth:               # エネルギー平滑化
          window_ms: 120
          mode: mean          # mean: 平均, max: 最大

      # ========================================
      # 既存機能: sections_blend
      # ========================================
      sections_blend:
        chorus: { ride2: +0.10 }
        bridge: { china: +0.15 }

      # ========================================
      # 状態遷移定義
      # ========================================
      states:
        ride1:
          bow:  { ride1: 0.70, ride2: 0.20, china: 0.10 }
          bell: { ride1: 0.60, ride2: 0.30, china: 0.10 }
        ride2:
          bow:  { ride1: 0.30, ride2: 0.55, china: 0.15 }
          bell: { ride1: 0.25, ride2: 0.50, china: 0.25 }
        china:
          bow:  { ride1: 0.60, ride2: 0.30, china: 0.10 }
          bell: { ride1: 0.50, ride2: 0.35, china: 0.15 }
```

---

## 処理順序

Markov遷移確率は以下の順序で変形されます：

```
1. 基底確率取得: states[current_state][head]
2. セクションブレンド: sections_blend
3. エネルギーブレンド: energy_blend
4. プリエンプト: preempt          ← NEW
5. スティッキー: sticky            ← NEW
6. Chinaクールダウン: cooldown_china_hits
7. フロア/キャップ: prob_bounds    ← NEW
8. サンプリング: seed対応           ← NEW
```

---

## 使用例

### 例1: 静かなバースから激しいサビへ

```yaml
prob_bounds:
  per_section:
    verse:  { cap: { china: 0.10 } }    # バースではChina抑制
    chorus: { floor: { china: 0.20 } }  # サビではChina保証

preempt:
  enable: true
  ahead_ms: 200
  sections:
    chorus: { add: { china: +0.60 } }   # サビ直前から激しく
```

### 例2: Bell主導のスムーズな展開

```yaml
sticky:
  per_head:
    bell: { min_hits: 3, self_boost: 0.15 }  # Bell時は滞在的
    bow:  { min_hits: 1, self_boost: 0.05 }  # Bow時は軽快

preempt:
  sections:
    chorus: { add: { ride2: +0.30 }, head: bell }  # Bell時のみ適用
```

### 例3: 完全再現可能なテイク

```yaml
seed:
  base: 42
  per_section: true  # セクション境界で決定論的に変化

# → 同じYAML+セクション構成で完全に同じボイス遷移
```

---

## NO-OP保証

| 機能 | 未設定時の挙動 |
|------|--------------|
| **prob_bounds** | `floor`/`cap` が空 → 何もしない |
| **preempt** | `enable: false` または `ahead_ms/ql: 0` → スキップ |
| **sticky** | `enable: false` → スキップ |
| **seed** | `base: null` → `self._rnd` を使用（従来通り） |

**すべての機能を削除すれば、完全に従来の挙動に戻ります。**

---

## テスト結果

```bash
$ python3 test_markov_stage2.py

============================================================
Markov Stage2 パッチ第二弾 動作確認
============================================================

=== Test 1: NO-OP (未設定) ===
✓ NO-OP設定で正常に初期化

=== Test 2: Deterministic Seed ===
✓ 決定論的シード: base=123, per_section=True

=== Test 3: Prob Bounds ===
✓ グローバル設定: Floor/Cap
✓ セクション別設定: verse, chorus

=== Test 4: Preempt (距離比例) ===
✓ プリエンプト設定: 200.0ms ahead, linear falloff

=== Test 5: Sticky (滞在バイアス) ===
✓ スティッキー設定: min_hits=2, per_head/per_state

=== Test 6: Energy Blend (既存) ===
✓ Energy Blend設定: mul mode, per_section, smooth

============================================================
✅ すべてのテストが成功しました！
============================================================
```

---

## トラブルシューティング

### Q: China が出過ぎる
```yaml
prob_bounds:
  cap: { china: 0.40 }  # 上限を設定
```

### Q: Ride1 が消える
```yaml
prob_bounds:
  floor: { ride1: 0.05 }  # 最低確率を保証
```

### Q: セクション境界がぎこちない
```yaml
preempt:
  enable: true
  ahead_ms: 200
  falloff: { mode: ease }  # S字カーブで滑らか
```

### Q: 高速でボイスが切り替わりすぎる
```yaml
sticky:
  enable: true
  min_hits: 2  # 最低2打は維持
```

### Q: 毎回違う結果になる
```yaml
seed:
  base: 42  # 固定シード
```

---

## 次のステップ

1. 実際の楽曲で各パラメータを試す
2. セクション構成に合わせてpreemptを調整
3. エネルギーカーブと組み合わせて動的に変化させる
4. テイクの再現性が必要な場合はseedを設定

**Happy Drumming! 🥁**
