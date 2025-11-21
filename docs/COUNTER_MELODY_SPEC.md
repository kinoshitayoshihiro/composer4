# Counter-Melody Specification (B-2)

## 📋 目的

メインボーカルのメロディを支えつつ、和声的な厚みと横方向の流れを付与すること。

OtobonAI の **GuideToneAI / EmotionAI / LyricAnchorIndex** を利用し、

- **常時鳴るが、邪魔にならない**
- **サビで自然に高揚する**

という J-POP的なストリングス/ピアノの**対旋律**を自動生成すること。

---

## 🎯 適用範囲

### 対象楽器
- **strings_plan_v2**（第1バイオリン相当の上声）
- **piano_plan_v2**（主に右手パート）

### 対象セクション
- **必須**: chorus / pre_chorus / bridge
- **可変**: intro / verse / outro（policy設定 `countermelody_strength` により制御）

---

## 📥 入力（依存コンテキスト）

カウンターメロディ生成は、以下の情報を前提とする。

### 1. manual_chordmap.json / chordmap_locked.json
- **key_center**（例: `C#m`）
- **bar単位のsymbol**（テンション含む、例: `C#m7(9)`）

→ ガイドスケール・コードトーン・テンショントーンの決定

### 2. lyric_anchors.json → LyricAnchorIndex
各 bar ごとの:
- `has_vocal`: ボーカル有無
- `phrase_role`: `start` / `mid` / `end`
- `stress_level`: 強勢（0.0-1.0）

→ ボーカルと時間軸上での**衝突回避**と、フレーズの**山谷の位置決定**に使用

### 3. rulebook.yaml
domain=`guidetone` / `lyric` / `emotion` のルール群

セクション名・phrase_role・registerに応じた:
- `preferred_degrees`: 推奨スケール度数（例: `[3, 7, 9, 11]`）
- `phrase_shape`: `uphill` / `downhill` / `arch`
- `notes_per_bar`: 基準値（例: `{min: 3, max: 6}`）

→ スタイルの"お作法"を決める

### 4. EmotionAI（emotion_ai_v2.py）
bar ごとの:
- `energy`: 0-1（エネルギー）
- `tension`: 0-1（緊張感）

→ density / velocity / duration のスケーリングに使用

### 5. policy/<song_id>.yaml
楽器・セクション別の:
- `countermelody_strength`: 0-1（対旋律の主張度）
- `density_floor` / `density_ceil`: 音数範囲
- `register`: `low` / `mid` / `high`

→ どのセクションでどの程度対旋律を主張させるかを制御

---

## 📤 出力

各 bar について、対旋律の **note イベント列** を `*_plan.json` に追加する。

### 仕様
- **1 bar あたりの note 数**: `rulebook.notes_per_bar × emotion.energy` スケールの範囲内
- **付与メタデータ**:
  - `phrase_id`: 対旋律フレーズ単位のID（4-8 bars単位）
  - `role`: `counter_melody` / `padding` / `chordpad`

→ 後続の QA や Humanize で利用

---

## 🎨 設計原則

### 1. 「歌より一段上」原則
- **Strings**: ボーカルの最高音より **3-5度上** を基本レンジ
  - 原則として**ボーカルとユニゾンしない**
- **Piano**: ボーカルの上下をまたぐが、**完全ユニゾンは避ける**

### 2. Chordmap-First
音高は必ず:
- **コードトーン**（1, 3, 5, 7）
- **テンション**（9, 11, 13）

から選ぶ。**スケールだけで勝手に動かない**。

**非和声音**（passing / neighbor）は **前後がコードトーンの場合のみ許可**。

### 3. Lyric-Aware

#### phrase_role == "start" の bar
- ボーカルの頭を邪魔しないように:
  - **拍アタマを外す**
  - または同じリズムで **3rd/5th のハーモニー** として扱う

#### phrase_role == "end"
- Strings に **上昇または下降のフィル** を優先的に生成

### 4. Emotion-Aware

#### energy が高い bar
- `notes_per_bar` を増やす
- `velocity` を +α する（上限は policy 指定）

#### tension が高い bar
- **テンション度数（9, 11）比率** を引き上げる
- ただし連続しすぎる場合は **4小節単位でコードトーンに回帰**

### 5. No-Mute & Noise-Free

#### chorus セクション
- strings/piano ともに **常時何かしらの note を持つ**

#### lyric_anchors が無く（ボーカル不在）・energy が低い bar
- **sustain pad** に切り替え、動きすぎないよう制御

---

## 🔧 生成アルゴリズム概要

### 6.1 bar 単位のコンテキスト構築

各 bar について、以下をまとめて `context` を構成する:

```python
context = {
    "section_name": "chorus",
    "bar_index": 16,
    "phrase_role": "start",
    "has_vocal": True,
    "chord_symbol": "C#m7(9)",
    "key_center": "C#m",
    "emotion": {
        "energy": 0.75,
        "tension": 0.6
    },
    "lyric_anchor": {
        "stress_level": 0.8
    },
    "policy": {
        "countermelody_strength": 0.8
    }
}
```

この `context` を **RulebookEngine** にクエリし、  
**GuideTonePlan**（preferred degrees / phrase_shape / register / notes_per_bar 基準）を取得する。

---

### 6.2 音高（pitch）選択

#### Step 1: ガイドトーン列生成
- `chord_symbol` と `preferred_degrees` から、bar 中心の  
  **scale_degree → MIDI pitch** を 1-2 個決定

- `phrase_shape = "uphill"` の場合:
  - 4小節単位で徐々に上昇（例: 3rd → 5th → 7th → 9th）

- `phrase_shape = "downhill"` の場合:
  - 逆方向に配置

#### Step 2: 補完ノート生成
- `notes_per_bar` を満たすまで、  
  **同スケール内で stepwise（±2度）に移動**

- **非和声音**は **前後が chord_tone の場合のみ**、16分〜8分音符として挿入

#### Step 3: レンジ制御
- `register`（low/mid/high）と  
  `vocal_estimated_register` を比較し、

- **Strings は vocal より最低 +3度高い位置** にクランプ

- **Piano は mid を中心**に、必要に応じて上下 1オクターブに収める

---

### 6.3 リズム（タイミング）生成

#### 基本グリッド
- **1/8〜1/16**

#### has_vocal == true の拍
- **強拍の完全被りを避け**、裏拍寄りに配置
- `phrase_start` では **2拍目以降を優先**

#### fill_slot == true な bar
- **4拍連続の スケールラン or アルペジオ** を優先

#### riff_slot == true な bar
- 2〜4小節にわたって繰り返せる **短いモチーフ** を生成
- 次小節以降で**リズムだけ変える**

---

### 6.4 ベロシティ / デュレーション

#### velocity
```python
base_velocity = policy.base_velocity  # 例: 72
velocity *= 1 + (energy - 0.5) * gain  # gain は 0.3〜0.5
```

- `phrase_start` では頭 1音を **+5〜+10**

#### duration
- `tension` が高い場合は **短め（staccato 寄り）**
- `tension` が低い場合は **長め（legato 寄り）** にスケール

- **sustain pad** に切り替える場合は **1〜2音/bar の ロングトーン** に固定

---

## 🚫 衝突回避ルール（最低限）

### 1. ボーカルとのユニゾン回避
- ボーカルと**同一ピッチが 1小節中 50%以上**を超える場合、  
  対旋律側を **3度上/下にシフト**

### 2. ベースラインとの距離確保
- Strings の最低音が  
  **ベースラインの最高音から 完全4度以内に入らない**よう制限（モコり防止）

### 3. 大跳躍の制限
- 1小節内に **跳躍 > 完全5度 が2回以上** 発生した場合、  
  2つ目以降を **stepwise に修正**

---

## ✅ QA 指標（チェック用）

カウンターメロディ導入後、以下を自動チェックする:

### Chorus 区間
- `strings_active_rate` ≥ **0.9**
- `piano_countermelody_rate` ≥ **0.6**（残りは pad / chord）

### Bar ごと
- `countermelody_notes_per_bar` ∈ **[1, 8]**

### 和声衝突
- **非和声音比率** ≤ **40%**
- **avoid interval**（例: major3度での低域クラスター）発生 bar 率 ≤ **10%**

---

## 📊 実装例（generate_strings_plan_v2.py）

```python
# Bar loop内
context = {
    "bar_index": bar_idx,
    "section": section_label,
    "role": "strings",
    "phrase_role": lyric_info.get("phrase_role", "mid"),
    "has_vocal": lyric_info.get("has_anchor", False),
    "chord_symbol": chord.get("symbol", "C"),
    "energy": emotion_params.energy,
    "tension": emotion_params.tension
}

# GuideTonePlan取得
guide_plan = guidetone_ai.get_plan(context)

# Counter-Melody生成
if guide_plan.pattern == "arpeggio_up":
    notes = make_countermelody_arpeggio_up(chord, guide_plan, emotion_params)
elif guide_plan.pattern == "cadential_hold":
    notes = make_sustain_pattern(chord, guide_plan, emotion_params)
else:
    notes = make_countermelody_generic(chord, guide_plan, emotion_params)

# 衝突回避
if vocal_f0:
    notes = [avoid_vocal_collision(n, vocal_f0) for n in notes]

# イベント追加
for note in notes:
    events.append({
        "time_ql": note["time_ql"],
        "note": note["midi"],
        "velocity": note["velocity"],
        "duration_ql": note["duration_ql"],
        "is_tension": note["is_tension"],
        "phrase_id": current_phrase_id,
        "role": "counter_melody"
    })
```

---

## 🎯 次のステップ

### 1. Algorithm明文化（B-2）
- `docs/COUNTER_MELODY_ALGORITHM.md` 作成
- `scripts/countermelody_lib.py` 共通ライブラリ作成

### 2. Rulebook拡張（C-1）
- `configs/otobonAI/rulebook.yaml` にLYRIC_101-110追加

### 3. CREPE導入（D-1）
- `scripts/analyze_vocal_f0.py` 作成
- `analysis/vocal_f0.csv` 生成
- Rulebook VOCAL_001-003追加

---

**作成日**: 2025-11-15  
**ステータス**: Phase B-1完了、B-2準備  
**関連ドキュメント**: `PHASE_2.5_COUNTERMELODY_ROADMAP.md`
