# Markov Stage2 パッチ第三弾 ユーザーガイド

Ride Cymbal の Markov 遷移システム（ride1↔ride2↔china）に**6つの高度な機能**を追加しました。

---

## 📚 目次

1. [状態別クールダウン](#1-状態別クールダウンstate_cooldown)
2. [遷移コスト](#2-遷移コストtransition_costs)
3. [セクション終端ラッチ](#3-セクション終端ラッチlatch)
4. [確率の慣性](#4-確率の慣性prob_momentum)
5. [ローカルモード](#5-ローカルモードlocal_modes)
6. [音素連動ブレンド](#6-音素連動ブレンドphoneme_blend)
7. [統合例](#統合例すべての機能を組み合わせる)
8. [トラブルシューティング](#トラブルシューティング)

---

## 1. 状態別クールダウン（state_cooldown）

### 概要
各状態（ride1/ride2/china）の**最小間隔**を制御します。

### パラメータ
```yaml
drums_params:
  ride:
    ride_markov:
      enable: true
      state_cooldown:
        # グローバル設定
        ride2: { bars: 1 }              # 1小節以上あける
        china: { bars: 2, hits: 2 }     # 2小節以上 & 2ヒット以上
        
        # Head別上書き
        per_head:
          bell:
            china: { hits: 2 }           # Bell時は2ヒット制限
          bow:
            ride2: { bars: 0 }           # Bowは制限なし
        
        # セクション別上書き
        per_section:
          verse:
            china: { bars: 3 }           # Aメロはchina控えめ
          chorus:
            ride2: { bars: 0 }           # サビではride2自由
            china: { bars: 1, hits: 1 }  # chinaは少し緩める
```

### 使用例

#### 基本: Chinaの連発防止
```yaml
state_cooldown:
  china: { bars: 2 }   # 2小節以上あける
```

#### 応用: セクションごとにChina頻度を変える
```yaml
state_cooldown:
  china: { bars: 2, hits: 2 }   # 基本は控えめ
  per_section:
    chorus:
      china: { bars: 1, hits: 1 }  # サビだけ少し頻繁に
```

---

## 2. 遷移コスト（transition_costs）

### 概要
状態遷移に**コスト関数**を適用し、特定の遷移を抑制/促進します。

### パラメータ
```yaml
drums_params:
  ride:
    ride_markov:
      enable: true
      transition_costs:
        mode: exp                    # mul: 1-cost, exp: exp(-cost) 推奨
        energy_alpha: 0.2            # Eが高いほどコスト増減（>0で増、<0で減）
        bias: 0.0                    # 全体的なバイアス
        
        # 基本コスト行列（現在状態→次状態）
        base:
          ride1:
            ride2: 0.10              # ride1→ride2は軽いコスト
            china: 0.30              # ride1→chinaは重いコスト
          ride2:
            ride1: 0.08
            china: 0.22
          china:
            ride1: 0.05
            ride2: 0.10
        
        # Head別追加コスト
        per_head:
          bell:
            ride1: { china: 0.20 }   # Bellからchinaはさらに重く
        
        # ヘッド切替コスト（前ヒットとhead変わる時）
        head_switch:
          enable: true
          cost: 0.08                 # ペナルティ
        
        # プリエンプト連動（境界直前の処理）
        preempt:
          alpha: 0.5                 # 境界近いほど全体コスト低減
          head: any                  # bell/any
          prefer:
            chorus:
              ride2: -0.20           # コーラス直前はride2へ誘導（コスト低減）
              china: -0.05           # chinaも少し誘導
```

### モード比較

| mode | 計算式 | 特性 | おすすめ |
|------|--------|------|----------|
| mul  | `p * (1 - cost)` | 線形減衰 | シンプル |
| exp  | `p * exp(-cost)` | 滑らかな減衰 | ✅ 推奨 |

### 使用例

#### 基本: ride1→chinaを抑制
```yaml
transition_costs:
  mode: exp
  base:
    ride1: { china: 0.40 }   # 重いコスト
```

#### 応用: エネルギー連動
```yaml
transition_costs:
  mode: exp
  energy_alpha: 0.3    # 高エネルギー時にコストが増える
  base:
    ride1: { china: 0.30 }
```

#### 高度: プリエンプト＋セクション誘導
```yaml
transition_costs:
  mode: exp
  base:
    ride1: { china: 0.30 }
  preempt:
    alpha: 0.6       # 境界直前は全体的にコスト低減
    prefer:
      chorus:
        ride2: -0.25  # コーラス直前は特にride2へ誘導（負値でコスト低減）
```

---

## 3. セクション終端ラッチ（latch）

### 概要
セクション**終端数拍**と**開始グレース**で状態を固定/誘導します。

### パラメータ
```yaml
drums_params:
  ride:
    ride_markov:
      enable: true
      latch:
        enable: true
        beats: 2.0               # セクション終端2拍でラッチ
        mode: prefer             # hold | prefer | force
        prefer_boost: 0.40       # prefer時のブースト量
        sections: [chorus]       # 適用セクション（省略で全体）
        state: ride2             # mode:force/prefer時の目標状態
        grace_beats: 0.5         # セクション開始直後0.5拍は現状態保持
```

### モード比較

| mode | 動作 | 用途 |
|------|------|------|
| hold | 現状態を固定 | 安定感重視 |
| prefer | 目標状態をブースト | 柔軟性重視（推奨） |
| force | 目標状態へ強制 | 確実性重視 |

### 使用例

#### 基本: サビ終わりを安定化
```yaml
latch:
  enable: true
  beats: 2.0
  mode: hold           # 現状態を保持
  sections: [chorus]
```

#### 応用: 終端はride2へ誘導＋開始グレース
```yaml
latch:
  enable: true
  beats: 2.0
  mode: prefer
  prefer_boost: 0.5    # 強めに誘導
  state: ride2
  sections: [chorus]
  grace_beats: 1.0     # セクション開始1拍は現状態保持
```

---

## 4. 確率の慣性（prob_momentum）

### 概要
**前フレーム分布**とのブレンドで微小ゆらぎを抑制します。

### パラメータ
```yaml
drums_params:
  ride:
    ride_markov:
      enable: true
      prob_momentum:
        enable: true
        alpha: 0.35              # 0..1（大きいほど前フレーム寄り）
```

### 使用例

#### 基本: 微小ゆらぎ抑制
```yaml
prob_momentum:
  enable: true
  alpha: 0.3   # 前フレーム30%、現在70%
```

#### 強め: 滑らかな変化
```yaml
prob_momentum:
  enable: true
  alpha: 0.5   # 半々でブレンド
```

---

## 5. ローカルモード（local_modes）

### 概要
**セクション/小節レンジ別**に遷移行列を切り替えます。

### パラメータ
```yaml
drums_params:
  ride:
    ride_markov:
      enable: true
      local_modes:
        # セクション指定
        chorus:
          inherit: true          # true=既存とブレンド / false=完全置換
          mix: 0.7               # 既存:0.3 / ローカル:0.7
          bars: [[1, 3]]         # コーラス内2〜4小節目だけ適用（省略で全体）
          states:
            ride1:
              bow:  { ride1: 0.50, ride2: 0.40, china: 0.10 }
              bell: { ride1: 0.40, ride2: 0.45, china: 0.15 }
        
        # 小節レンジ直接指定
        by_range:
          - from: 8
            to: 16
            head: any            # any / bell / bow
            sections: [chorus]   # 省略可（全体）
            inherit: false       # 完全置換
            states:
              ride1:
                bow: { ride1: 0.30, ride2: 0.50, china: 0.20 }
```

### 使用例

#### 基本: サビはride2多め
```yaml
local_modes:
  chorus:
    inherit: true
    mix: 0.8       # ローカル寄り
    states:
      ride1:
        bow: { ride1: 0.40, ride2: 0.50, china: 0.10 }
```

#### 応用: 特定小節でChina強調
```yaml
local_modes:
  by_range:
    - from: 12
      to: 16
      inherit: false   # 完全置換
      states:
        ride1:
          bow: { ride1: 0.30, ride2: 0.30, china: 0.40 }  # China多め
```

---

## 6. 音素連動ブレンド（phoneme_blend）

### 概要
**子音クラス×エネルギー**で遷移を抑制/促進、**クラス別ウィンドウ**対応。

### パラメータ
```yaml
drums_params:
  ride:
    ride_markov:
      enable: true
      phoneme_blend:
        enable: true
        mode: add                # add / mul
        alpha: 0.5               # 効きの強さ
        energy_alpha: 0.5        # エネルギー連動の強さ
        head: any                # any / bell / bow
        sections: [verse, chorus]  # 適用セクション（省略で全体）
        window_ms: 120           # グローバル窓（クラス別で上書き可）
        falloff: ease            # linear / ease / exp
        
        classes:
          sibilant:
            states: { china: -0.40, ride2: -0.20, ride1: +0.10 }
            window_ms: 160       # ヒスはやや広めに監視
            falloff: exp         # 中心寄りを強調
            head: any
            sections: []         # 全体適用（省略可）
          
          plosive:
            states: { china: +0.15, ride2: +0.10 }
            window_ql: 0.25      # ごく短く（瞬間）
            falloff: linear
            sections: [chorus]   # サビ中のみ効かせる
          
          nasal:
            states: { ride1: +0.10, ride2: -0.05 }
            window_ms: 100
            head: bow            # Bowヒット時のみ
```

### モード比較

| mode | 計算式 | 特性 |
|------|--------|------|
| add  | `p + α*δ*(1+E_α*E)` | 素朴な加算 |
| mul  | `p * (1 + α*δ*(1+E_α*E))` | 倍率調整 |

### 使用例

#### 基本: ヒス音でChina抑制
```yaml
phoneme_blend:
  enable: true
  mode: add
  alpha: 0.5
  classes:
    sibilant: { china: -0.40 }   # ヒス音でchina確率を下げる
```

#### 応用: クラス別ウィンドウ
```yaml
phoneme_blend:
  enable: true
  mode: add
  alpha: 0.6
  window_ms: 120        # デフォルト窓
  classes:
    sibilant:
      states: { china: -0.50 }
      window_ms: 200    # ヒスは長めに監視
      falloff: exp      # 中心寄りを強調
    plosive:
      states: { china: +0.20 }
      window_ql: 0.2    # 破裂音は極短
      falloff: linear
```

---

## 統合例：すべての機能を組み合わせる

```yaml
drums_params:
  ride:
    ride_markov:
      enable: true
      start: ride1
      states:
        ride1:
          bow:  { ride1: 0.60, ride2: 0.30, china: 0.10 }
          bell: { ride1: 0.50, ride2: 0.40, china: 0.10 }
        ride2:
          bow:  { ride1: 0.30, ride2: 0.60, china: 0.10 }
          bell: { ride1: 0.25, ride2: 0.60, china: 0.15 }
        china:
          bow:  { ride1: 0.40, ride2: 0.45, china: 0.15 }
          bell: { ride1: 0.30, ride2: 0.50, china: 0.20 }
      
      # 1) 状態別クールダウン
      state_cooldown:
        ride2: { bars: 1 }
        china: { bars: 2, hits: 2 }
        per_section:
          verse: { china: { bars: 3 } }      # Aメロは控えめ
          chorus: { ride2: { bars: 0 } }     # サビは自由
      
      # 2) 遷移コスト
      transition_costs:
        mode: exp
        energy_alpha: 0.25
        base:
          ride1: { ride2: 0.10, china: 0.35 }
          ride2: { ride1: 0.08, china: 0.25 }
          china: { ride1: 0.05, ride2: 0.10 }
        head_switch: { enable: true, cost: 0.08 }
        preempt:
          alpha: 0.5
          prefer:
            chorus: { ride2: -0.25, china: -0.08 }
      
      # 3) ラッチ
      latch:
        enable: true
        beats: 2.0
        mode: prefer
        prefer_boost: 0.45
        sections: [chorus]
        state: ride2
        grace_beats: 0.5
      
      # 4) 慣性
      prob_momentum:
        enable: true
        alpha: 0.35
      
      # 5) ローカルモード
      local_modes:
        chorus:
          inherit: true
          mix: 0.75
          states:
            ride1:
              bow: { ride1: 0.50, ride2: 0.42, china: 0.08 }
        by_range:
          - from: 12
            to: 16
            inherit: false
            states:
              ride1:
                bow: { ride1: 0.35, ride2: 0.40, china: 0.25 }
      
      # 6) 音素連動
      phoneme_blend:
        enable: true
        mode: add
        alpha: 0.55
        energy_alpha: 0.45
        window_ms: 120
        falloff: ease
        classes:
          sibilant:
            states: { china: -0.45, ride2: -0.22, ride1: +0.12 }
            window_ms: 180
            falloff: exp
          plosive:
            states: { china: +0.18, ride2: +0.12 }
            window_ql: 0.22
```

---

## トラブルシューティング

### Q1: 設定が効いていない気がする
**A1**: まず `enable: true` を確認してください。各機能ごとに個別のenableフラグがあります。

```yaml
state_cooldown: { ... }           # これだけでは不十分
ride_markov: { enable: true }     # これも必要
```

### Q2: Chinaが全く出なくなった
**A2**: 複数の制約が重なっている可能性があります。以下を確認:

1. `state_cooldown.china` の `bars`/`hits` が厳しすぎないか
2. `transition_costs.base.*.china` のコストが高すぎないか
3. `phoneme_blend.classes.*.states.china` が強く負値になっていないか
4. `local_modes` でchinaが0に近くなっていないか

### Q3: 処理が重い
**A3**: 以下を試してください:

1. `phoneme_blend.window_ms` を小さくする（200→100等）
2. `phoneme_blend.classes` の数を減らす
3. 不要な機能を `enable: false` にする

### Q4: エラーが出る
**A4**: YAMLの構文エラーを確認してください。特に:

- インデント（スペース2個）
- コロン `:` の後ろにスペース
- リスト `[]` とマップ `{}` の使い分け

### Q5: 予期しない動作をする
**A5**: デバッグのコツ:

1. まず全機能を `enable: false` にして基本動作確認
2. 1つずつ機能を `enable: true` にして原因特定
3. `prob_momentum.alpha` を小さく（0.1等）して慣性を弱める

---

## さらなる情報

- **実装詳細**: `MARKOV_STAGE2_PATCH3_REPORT.md`
- **テストコード**: `test_markov_patch3.py`
- **パッチ第二弾**: `MARKOV_STAGE2_PATCH2_GUIDE.md`（prob_bounds/preempt/sticky/seed）

---

**最終更新**: 2025年10月18日  
**バージョン**: パッチ第三弾
