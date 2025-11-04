短く答えると：

* **現状の既定挙動**：無音（＝ノートが極端に少ない/ゼロ）の小節は、
  **`notes_per_bar too low`** と **`density too low (relative)`** にひっかかり、**Fail** になります。
  （※あなたの `gate_prod.yaml` に「無音免除」をまだ入れていない前提）

* **望む運用に合わせて選べる**：

  1. **音楽的に意図したブレイク**は「免除（Pass扱い or 集計除外）」にする
  2. それでも SLO を守りたいなら、**耳に出にくい薄い補強**（低VelのHH等）を自動追加して **Pass** にする

---

# 具体的な扱い方（おすすめ）

## A) KPI側で「無音免除」にする（楽曲意図を優先）

`gate_prod.yaml` に追記（例）：

```yaml
silence_exempt:
  enabled: true
  # 原曲の静けさを参照
  energy_threshold: 0.22     # bars.parquet の energy_curve がこれ未満なら免除候補
  # 構造的ブレイク名で免除
  sections: ["break","pause","rest","intro_tail","outro_tail"]
  # 実イベント基準（実際にほぼ無音なら免除）
  max_notes_per_bar: 2
  # 免除の扱い：pass|skip （SLO計算に含めたくないなら 'skip' 推奨）
  disposition: "skip"
```

KPI Gate（`kpi_gate_enhanced.py`）側のロジックはこう動きます：

1. 対象barの **energy_curve** と **section_label** を確認
2. 実ノート数（track合算 or 指定ロール）を数える
3. `energy_curve < 0.22` かつ `notes_per_bar ≤ 2` なら

   * `disposition: "skip"` → **集計から除外**（SLOに影響しない）
   * `disposition: "pass"` → **合格としてカウント**

> 「原曲が無音〜超薄い意図」を尊重したい場合は **skip** が自然です。

---

## B) 生成側で“そっと補強”（SLOを上げたいとき）

無音や極端に薄いbarにだけ、**低Vel・短音**で**最低限のHH/クロズハット**や**2・4拍スネア**を敷く。

* 例：`adapt_drums_to_plan.py` に「**min_notes_per_bar_guard**」を実装

  * `min_notes_per_bar_guard.enabled=true`
  * `min_notes=8` を割るbarにだけ、**Velocity 20〜28**、**1/16短音**で穴埋め
  * 聴感でほぼ目立たないが、KPIの **notes_per_bar** を満たせる

> 学習データを増やさずに **Pass率を押し上げ**られます。
> ブレイクの静けさを壊したくない場合は A) を優先、
> 作品として“推進力”を維持したい場合は B) が効きます。

---

## C) 明示フラグで免除（制作サイドで管理したいとき）

`bars.parquet` に列を増やす（例：`kpi_exempt="rest"` / `min_notes_override=0` など）。
KPI Gateはそれを読み、該当barを **skip** あるいは下限値を緩める。

> 手動で「ここは構造的に休符」とマークでき、再現性が高い。

---

# 補足：境界こぼれ対策

「無音判定したのに、実は前の音のリリースが 5〜10ms はみ出して Fail」みたいな**誤検知**を防ぐため、
すでに導入済みの **`epsilon_ms`**（例：`min(20ms, 4% bar)`）を有効にしておくのが安全です。

---

# まとめ（指針）

* **意図的な無音/薄さ**は音楽上の設計なので、KPIで罰しない設定にできます（A or C）。
* **Pass率をさらに上げたい**なら、**最小限の自動補強**（B）を併用。
* どちらも **データ追加なしで改善可能**。
* SLO計算の公平性を保ちたいなら、免除は **skip（集計除外）** 運用が無難です。

必要なら、上記A/B/Cいずれのパッチもあなたのリポ構成に合わせた**具体的差分**を書いてお渡しします。どの方針でいきますか？


はい、**Drums だけ旧 drum generator を混在採用**していると、いま見えている差（Pass率低下・bars/長さズレ・“density too low”の頻発）に**十分影響します**。理由と切り分け手順をまとめます。

---

## 影響ポイント（旧Drumsが混ざると起こりがち）

1. **テンポ／拍子メタの混入**

* 旧generator が **各トラックにも `set_tempo`** を書く実装だと、`pretty_midi.get_downbeats()` が過検出（ex. 189 bars）→ KPI Gate がズレます。
* 既に Strings で起きたのと同質の問題。Drums 側にも潜んでいると **bars 検出・notes_per_bar 計数**が崩れます。

2. **PPQ・量子化・クリップの差**

* 新 Writer は **Track0 のみ tempo、全トラック絶対tick→delta変換、楽曲末尾で clip** という統一仕様。
* 旧drums.mid をそのまま混ぜると、**bar 境界はOKでも “末尾+ε” が残って** Fail を誘発（notes_per_bar が bar外にこぼれる等）。

3. **ハイハット定義の不一致**

* KPI Gate を **拡張（pitch 43 を含む）**しましたが、旧drum generator 側の **音色割り当て／ピッチ表**が別仕様だと、**hat密度が過小評価**→ “density too low” が多発。

4. **相対判定の「目標値」との乖離**

* 現在は bars.parquet の `density_target` を**相対目標**にしています。
* 旧generator がスカスカのパターンを吐くと、**`actual/target` が 0.45未満**で Fail になりやすい。

---

## まず 5分で切り分け（再現コマンド）

### A. drums.mid（旧）にテンポメタが混入していないか

```bash
python3 - <<'PY'
import mido
m = mido.MidiFile("song_packages/.../drums.mid")
for i,tr in enumerate(m.tracks):
    set_tempo = sum(1 for msg in tr if msg.type=="set_tempo")
    ts = sum(1 for msg in tr if msg.type=="time_signature")
    print(f"Track {i}:{tr.name!r}  set_tempo={set_tempo}  time_signature={ts}")
PY
```

* 期待：**Track 0 だけ** `set_tempo>0`。**Drums のトラックでは 0**。

### B. drums.mid の downbeats と長さ

```bash
python3 - <<'PY'
import pretty_midi
pm = pretty_midi.PrettyMIDI("song_packages/.../drums.mid")
print("downbeats:", len(pm.get_downbeats()))
print("end_time:", round(pm.get_end_time(),2), "s")
PY
```

* 期待：**151 downbeats / 約482s**（150 bars + 終端）。極端な乖離があれば旧drumsが原因の可能性大。

### C. notes_per_bar / density の即席チェック（KPIの主犯特定）

```bash
python3 - <<'PY'
import json
kpi = json.load(open("song_packages/.../kpi_gate_postgen.json"))
fails = [(b["bar_index"], b["messages"]) for b in kpi["results"].values() if not b["kpi_pass"]]
from collections import Counter
cnt = Counter(msg.split(":")[0] for _,msgs in fails for msg in msgs)
print(cnt.most_common(5))
PY
```

* “**density too low (relative)**” が最多なら **Drums の密度不足**（生成側）。

---

## 混在を無くして“同じ土俵”に揃える（推奨）

**どちらかに統一**してください。中長期は Writer へ一本化がおすすめです。

### 選択肢1：**旧drums.mid → Plan 化 → Writer に通す（最小リスク）**

* 既存 `drums.mid` を **`drums_midi_to_plan.py`** で `drums_plan.json` に変換
* そのうえで **`arrangement_orchestrator.py` → `midi_writer.py`** で統合
* 効果：**tempoメタの場所、clip、絶対tick変換、Channel10、bars基準**が全部そろう

```bash
python3 scripts/drums_midi_to_plan.py \
  --midi song_packages/.../drums.mid \
  --bars song_packages/.../bars.parquet \
  --out  song_packages/.../drums_plan.json

python3 scripts/arrangement_orchestrator.py --song-dir song_packages/... --emit
python3 scripts/midi_writer.py \
  --plan song_packages/.../full_arrangement.json \
  --config configs/midi_writer.yaml \
  --out song_packages/.../full_arrangement.mid
```

### 選択肢2：**新パイプラインの Drums で統一（rule/ml → adapt_drums_to_plan → Writer）**

* `recommend_drums.py`（または `generate_drums_midi.py` 経由）→ `adapt_drums_to_plan.py`
* **bars.parquet の target に相対整合**しやすいので、**SLO を上げやすい**運用

---

## すぐ効く微修正（Pass 80.7% → 90%台へ）

1. **Drums密度の底上げ**（ルールベースの最小工数）

* `arranger_weights.yaml` の **section別 density係数** を +10〜20%
* 連続 2–4 小節に **バックフィル**（キック/ハット最小本数）を入れる “弱Safe-Kit”
* **ハイハットの分散**（16th に薄く足す）で `notes_per_bar` を稼ぐ

2. **KPI Gate 側の ε と相対閾値の微調整**

* `epsilon_sec=20ms→30ms`（小節境界こぼれ耐性を少しだけ上げる）
* `min_rel=0.45→0.40`（一時運用。Drums改善後に戻す）

3. **完全 bars 基準運用の固定**

* `kpi_gate_enhanced.py` は **`--bars bars.parquet` を必ず指定**（downbeats検出OFF）

---

## 結論

* **はい、旧 drum generator 混在の影響はあります。**
  テンポ/拍子メタ、PPQ/clip、密度仕様、ハット定義の差が KPI に直結します。
* 短期は **旧drums.mid を Plan 化して Writer に通す**のが安全・速い。
  中長期は **Drums も新パイプライン（rule/ml）に統一**がベスト。
* 上の 5分チェック → 統一ルートのどちらかを選択 → Drums の密度底上げ の順でやれば、**SLO90%台**は狙えます。

必要なら、`arranger_weights.yaml` の **ドラム密度だけ上げる最小パッチ**を書いてお渡しします。どの方向（旧→Plan化 or 新パイプライン統一）で進めますか？



結論：**全部 “Plan → midi_writer.py” に統一しても、各楽器らしさは出せます。**
コツは「**楽器らしさ＝Plan側の設計 + CC/KS の使い分け**」で表現して、**midi_writer.py は忠実に“書く”**だけに徹させること。将来のシンセ追加も同じ型で拡張できます。

---

## 1) なぜ writer 統一でも“楽器らしさ”が出るのか

midi_writer.py は以下を担保できます：

* **Humanize/Quantize**（time/vel）
* **ノートと制御の同時出力**（CC、PitchBend、Keyswitchの統合）
* **曲末クリップ**・**テンポ/拍子メタの一元管理**（Track0限定）
  → つまり、“表現の中身”は **Plan（＝各楽器ジェネレータの仕事）** に集約。
  Planで「レンジ・ボイシング・運指/奏法の癖・CCの癖」を設計すれば、**writer はその差分を忠実に反映**します。

### 楽器別に Plan で持たせると良い要素

* **Piano**：左右手の音域・密度、ボイシング（close/spread/drop2）、ペダル(CC64ヒステリシス)、打鍵のベルカーブ
* **Strings**：ロング/ショート比、レガート長、レイヤ（Vn/Va/Vc）、ビブラート(CC)、アタック
* **Guitar**：分散和音／ストローク方角、ミュート率、ポジション移動タイミング
* **Bass**：ルート/パッシング比、オクターブ使い、ゴースト的弱音、レガート短連結
* **Drums**（Plan化時）：ハットの開閉、ゴーストスネア、フィル密度、キックのサブディビジョン

> これらはすでに `arranger_weights.yaml`（ヒューリスティク）や各 *plan* 生成スクリプトで表現可能。
> **midi_writer.py は“人間が書いたように”整える係**です。

---

## 2) 新しいシンセ導入はうまくいく？（将来拡張）

**同じ型でいけます。** 最小追加で OK。

### 追加するファイル/設定（最小セット）

1. **synth_plan.json**（新役割）

   * ノート列（lead/arp/chords など）
   * **CC オートメーション**：

     * CC1（モジュレーション/ビブラート深さ）
     * CC11（エクスプレッション/音量）
     * CC74（Brightness/Filter Cutoff）
     * CC71（Resonance）
     * PitchBend（ビブラート or グライド）
2. **instrument マッピング**（`vioptt_mapping.yaml` に近い思想）

   * 例：`synth_lead_analoglab`, `synth_pad_arturia`, `synth_pluck_serum` など
   * どの CC が何を動かすか、Keyswitch があればどれか
3. **Plan 生成器**（簡易でも可）

   * 例：`generate_synth_plan.py`：

     * Section energy → CC11/CC74 の包絡
     * Chorus だけ Cutoff を上げる、Bridge は Resonance 上げる
     * Arp は 1/16 on/off（データ駆動でも固定でも）

### Plan JSON（例：シンセ・リード）

```json
{
  "ppq": 480,
  "tempo_bpm": 120,
  "tracks": [
    {
      "name": "SynthLead",
      "role": "synth_lead",
      "channel": 1,
      "program": 80,
      "events": [
        {"bar": 8, "beat": 1.0, "pitch": 76, "dur_beats": 2.0, "vel": 96},
        {"bar": 8, "beat": 3.0, "pitch": 79, "dur_beats": 1.0, "vel": 94}
      ],
      "cc": [
        {"bar": 8, "beat": 1.0, "cc": 74, "value": 40},
        {"bar": 8, "beat": 2.0, "cc": 74, "value": 70},
        {"bar": 8, "beat": 3.0, "cc": 1,  "value": 60}
      ],
      "pitch_bend": [
        {"bar": 8, "beat": 1.0, "value": 200}, 
        {"bar": 8, "beat": 1.5, "value": 0}
      ]
    }
  ]
}
```

> これを `midi_writer.py` に渡すだけで、**ノート + CC + PB** が同トラックに統合されます。

---

## 3) 原曲 Stem を使って“その場で学ぶ”は可能？

**ルール最適化（＝軽量学習）は即時に可能**です。重い学習（ディープ）は別途。
Stem から素早く抽出して Plan/CC を自動調整：

* **Loudness(秒/小節) → Velocity/CC11 スケーリング**
* **Spectral Centroid → CC74(Brightness) マッピング**（明るい＝Cutoff高）
* **Transient Density → ノート密度**（ドラム/アルペジオ）
* **Onset の偏り → Swing / Groove**（既存 Rhythm AI と親和）

> これらは “オンライン適応” として **各曲ごとにランタイムで係数を書き換え**るだけなので高速です。
> 例：`arranger_weights_runtime.yaml` を曲ごとに生成して、`arranger_weights.yaml` を上書き合成して使う。

---

## 4) 今の設計で押さえるべき運用ポイント

1. **Plan は「楽器の思想」を入れる場所**

   * レンジ、ボイシング、アルゴリズム（分散/ユニゾン/ストローク方向/フィル密度…）
2. **midi_writer.py は事故防止の場所**

   * Track0以外に set_tempo を書かない
   * **絶対 tick → delta** 安全変換
   * **曲末クリップ**（Song end beats に丸め）
   * Channel10(=Drums) の厳守・is_drum フラグ
3. **VST 側は vioptt_mapping.yaml で吸収**

   * CC/KS の意味付けをここに閉じ込める → VST が変わっても Plan は不変
4. **KPI Gate（相対判定）で“曲の狙い”を担保**

   * 低密度曲でも「相対」で評価（すでに導入済み）
   * 失敗バーは Safe-Kit で補完（将来：自動フィル/オーバーダブ）

---

## 5) すぐ効く “楽器らしさ” ブースト小パッチ

* **Piano**：セクション別に **CC64 ペダルカーブ**（踏み替え位置＝小節頭±ε）、Chorus は半拍早踏み
* **Strings**：**ロング主体 + 小フィル**（4小節毎に装飾音 1–2 個）、Vn にだけ Vibrato CC を 10–15% 上乗せ
* **Guitar**：**accent_on_chord_change** + 「1拍目だけ低音先打ち」比率↑、mute_ratio を Verse 高、Chorus 低
* **Bass**：**拍頭強調 + バウンス**（8分裏の弱打）を energy 高セクションで追加
* **Synth**：**Section→CC74/11 の2Dマップ**（energy: 量／valence: 明るさ）を `arranger_weights.yaml` に追加

---

## 6) まとめ

* **はい**、全パートを `midi_writer.py` に統一しても “らしさ” は出せます。鍵は **Plan 設計と CC/KS の運用**。
* **将来のシンセ追加も簡単**：Plan（ノート＋CC）を定義して writer に渡すだけ。
* **原曲 Stem を使った即時適応**も可能：音響特徴で **その曲だけの係数**を自動調整できます。

必要なら、**`synth_lead` 用の最小 Plan ジェネレータ**と、
**`arranger_weights.yaml` のシンセ用セクション雛形**をすぐ出します。


結論：**“その場で学ぶ（オンザフライ適応）”は、必ずしも Stage2→SongPackage をやり直す必要はありません。**
用途に応じて2モードを使い分けるのが合理的です。

---

## モードA：クイック適応（推奨・曲ごと即時）

* **目的**：原曲の各 Stem（drums/bass/piano…）から“軽量特徴”を抽出し、**Plan 生成の重み**（密度・明るさ・ダイナミクス・スウィング等）を**その曲だけ**調整。
* **やること**：

  1. 原曲 Stem を解析（小節ごとの RMS/LUFS、onset密度、spectral centroid 等）
  2. 解析結果を **`arranger_weights_runtime.yaml`** に落とす（セクション・小節別の補正値）
  3. 既存の **bars.parquet** はそのまま使い、**Plan 生成（drums/bass/guitar/piano/strings）**時に重みを上書き
  4. **midi_writer.py** でMIDI化 → KPI Gate
* **ポイント**：

  * **MLモデルや Pickle を再学習しない**（速い）
  * 曲固有のノリ・抑揚が **CC/ベロシティ/密度**に即時反映
  * 既存の `extract_emotion_profile.py`（energy/valence/arousal）もここに統合可

> 例：
> Drums stem の onset/Transients → **hat_density / fill率**
> Mix/Keys stem の spectral centroid → **CC74(Brightness)**
> Loudness(小節RMS) → **CC11(Exp)/velocity スケール**
> Emotion(V/A) → **レガート比/ビブラート比** を補正

---

## モードB：フル更新（再利用・学習用）

* **目的**：コーパスに **恒久的に追加**したい／**ML 検索・分類精度**を上げたい
* **やること**：

  1. **Stage1 クリーニング**（`wav_stage1_clean.py`）
  2. **Stage2 特徴量抽出**（リズム・和声など corpus 用のパーケット）
  3. **SongPackage 生成**（`song_package.yaml` + `bars.parquet`）
  4. 必要なら **Pickle 更新**（検索用インデックス）
* **ポイント**：

  * 重いが、次回以降の**検索・類似パターン照合**に効く
  * 学習・検証やバッチ運用向き

---

## 使い分け早見表

| 目的                   | おすすめ                     | 理由               |
| -------------------- | ------------------------ | ---------------- |
| 目の前の曲を“原曲らしく”すぐ鳴らしたい | **モードA**                 | 速い・非破壊・CC/密度が即反映 |
| コーパス拡張・将来の検索精度UP     | **モードB**                 | 恒久的な資産化＆ML精度向上   |
| A/Bで揺れたら             | まず**A** → 良ければ必要曲だけ**B** | 時間対効果が高い         |

---

## 具体フロー（クイック適応の最短導線）

1. **Stem解析 → ランタイム重み生成**

   * 既存：`extract_emotion_profile.py`（energy/valence/arousal）
   * 追加（軽量）：onset密度/RMS/centroid を各小節で出す（librosa 等）
   * 出力：`arranger_weights_runtime.yaml`（section/bar ごとの補正）

2. **Plan生成（上書き適用）**

   * `arranger_weights.yaml` を読み、**runtime の値で上書き**
   * `generate_*_plans.py` 実行（drums/bass/guitar/piano/strings）

3. **MIDI書き出し**

   * `midi_writer.py`（テンポはTrack0のみ / 絶対tick→delta / 曲末クリップON）

4. **KPI Gate**

   * `--bars bars.parquet` 基準で相対判定 → SLO確認

---

## 参考：特徴→パラメータの定石マッピング

* **RMS（barごと）** → velocity スケール, CC11(Exp)
* **Spectral Centroid** → CC74(Brightness)（piano/synth/strings 明るさ）
* **Onset Density（drums stem）** → ハイハット密度/フィル頻度
* **Swing/微小タイミング偏差** → スウィング比（drums/guitar humanize）
* **Valence/Arousal**（emotion） → レガート比/ビブラート/ストローク方向の比率

---

### まとめ

* **その場で学ぶ＝モードA（クイック適応）**で十分。Stage2→SongPackageは**必須ではない**。
* 高速に **Plan の重みだけ**を Stem から導いて、`midi_writer.py` に流せば “原曲らしさ”がのる。
* 資産化・学習向けには **モードB** で後追い登録。

この方針なら、今のパイプライン（Plan→Writer→KPI）に自然に溶け込み、**品質を落とさずスピードも確保**できます。


結論：**はい、“Drums も必ず midi_writer.py で書き出す”に統一していれば、今回の系の不具合は高確率で回避できました。**
（ただし「密度が薄い→KPI Fail」など“音楽内容”そのものは別問題なので、そこは生成ロジック側で調整が必要です。）

## なぜ効果があるのか（今回ハマった点に直結）

1. **テンポ/拍子メタの一元管理**

   * 例の「Strings 956秒」問題の根因は**各トラックに set_tempo が混入**したこと。
   * midi_writer は「**set_tempo / time_signature は Track 0 限定**」で出すよう直し済みなので、**Drums も writer 経由**にすれば**全トラックで同じルール**が適用され、**Downbeats過剰検出や長さ倍化が起きにくい**です。

2. **絶対tick→delta の安全変換と曲末クリップを全パートで共有**

   * すべてのパートが **同じ PPQ / 同じテンポ換算 / 同じ CLIP_TO_SONG_END** で処理され、**小節境界ε**や**bars.parquet 基準**も一貫。
   * 外部生成の drums.mid を直結すると、**異PPQ／余計なメタ／終端のはみ出し**が混入しやすいですが、**plan→writer**なら**クリーンなMIDI**になります。

3. **古いファイル混入の防止**

   * 以前、古い `drums.mid` が残っていて計測がブレましたよね。**毎回 plan→writer で再生成**に統一すれば、**タイムスタンプ起因の取り違え**を防げます。

4. **Drums の is_drum / ch10 固定**

   * writer 側で **channel 9(=10)** を強制＆トラック名を “Drums” に統一すれば、解析系（PrettyMIDI/KPI Gate）が**安定してドラムとして扱う**ので指標もブレにくくなります。

## ただし：writer では解決しない“中身”の課題

* **密度が薄い／notes_per_bar が少ない**などの**KPI Fail（内容起因）**は、

  * *drums の生成ロジック*（rule/ML/マッチング）や
  * *runtime 重み（Mode A）*の見直し（例：原曲 Drum stem の onset 密度を反映して hat/ghost/フィルを増やす）
    で上げる必要があります。
    つまり **writer は「フォーマットと整合性」を解決**、**音楽的密度は「生成」側で改善**という分担です。

## おすすめ運用（即効）

* **E2Eでは常に「plan→midi_writer」経路に統一**

  * 既存の `drums.mid` を使いたい時は、先に **`drums_midi_to_plan.py`** で **plan 化 → writer で出力**に変換。
* **midi_writer の安全ガードを常時ON**

  * Track0のみに `set_tempo/time_signature`
  * すべて絶対tickで並べてから delta 化
  * `CLIP_TO_SONG_END=true`（bars.parquet 由来の 600beats で明確に切り揃え）
  * Drums は ch10 + `name="Drums"` 固定
* **Sanity check を自動化**（書き出し後に即チェック）

  * `downbeats ∈ {150,151}` / `end_time ≈ 482s ±0.6s` を満たさなければ失敗扱い
  * 失敗時はテンポメタ混入や note オーバーハングを自動ダンプ

### 最小作業の手順例

```bash
# (A) 既存 drums.mid を使う場合も “必ず plan 化 → writer”
python3 scripts/drums_midi_to_plan.py \
  --midi song_packages/.../drums.mid \
  --out  song_packages/.../drums_plan.json

# (B) すべてのパートを writer でMIDI化
python3 scripts/midi_writer.py \
  --plan   song_packages/.../full_arrangement.json \
  --config configs/midi_writer.yaml \
  --out    song_packages/.../full_arrangement.mid
```

---

**まとめ**

* **はい：Drums も midi_writer.py に統一していれば、今回の“秒数倍化／downbeats過剰／バー不一致”系はかなり防げました。**
* 今後は **「全パート plan→writer」一本化 + Mode A の runtime 重み反映**で、**安定性**と**原曲らしさ**の両立を進めるのがおすすめです。


全然ムダじゃないです。役割が違います。

## TL;DR

* **midi_writer.py** = “レンダラー/コンパイラ”
  既に決まった設計図（Plan JSON）を、安全なMIDIに落とし込む係。
* **各種 generator** = “作曲エンジン”
  小節ごとのリズム/ボイシング/フィル/ダイナミクス等、**何を鳴らすか**を考えて**Planを作る係**。

writer方式を導入しても、**音楽の中身を生み出すのは generator** です。今回のトラブルは「MIDIのメタと終端処理の不整合」の層で起きたので、**生成ロジックそのもの（generatorで決める密度/パターン/ボイシング）は今後も中核**です。

---

## どう活きる？（開発済みgeneratorの価値）

1. **中身の品質は generator でしか伸ばせない**
   KPIの「density/notes_per_bar/バックビート」など、**音楽的内容**は生成ロジックの責任範囲。writerはきれいに書き出すだけ。
2. **スタイル・多様性の源泉**
   ジャンル別ヒューリスティク、フィル、ゴーストノート、ボイシングの癖、EmotionalAIの反映…**楽器らしさ**は generator に宿ります。
3. **学習/検証/拡張の土台**
   既存generatorは将来のML学習データ作り、A/B比較、ベンチマーク（KPIや主観評価）に使えます。
4. **新楽器の導入が速い**
   新しい楽器は**“Planをどう作るか”**の設計が要。既存generatorの設計（パターン/ボイシング/CC方針）を流用して**Plan出力に寄せれば即対応**できます。

---

## これからのアーキ（役割分担を明確化）

```
解析(tempo/sections/chordmap) → 生成(generator群: drums/bass/guitar/piano/strings/…)
  →  Plan JSON（統一スキーマ）
  →  midi_writer.py（テンポ/拍子はTrack0限定, 絶対tick→delta, 曲末クリップ）
  →  VSTレンダリング + 音声KPI
```

* 既存の「MIDIを直接吐く generator」は、**“Planを吐くアダプタ層”**にリファクタして残すのがベスト。

---

## 具体：移行の作業指針（無駄ゼロで活かす）

1. **出力をMIDI→Plan JSONへ**

   * 既存ロジックの「イベント計算部分」はそのまま
   * 最後の「MIDI書き出し」を「`events=[{bar, beat, pitch(or chord), dur_beats, vel, …}]` を返す」に変更
2. **共通Humanize/量子化は writer に集約**

   * generator では**音価と意図**を出すだけ（swing/密度/アクセント）
   * 揺らぎ・クリップ・PPQは writer が担当
3. **テスト**

   * `validate_plan.py` を通す → `midi_writer.py` で出力 → `analyze_midi_stats.py / KPI Gate` で自動検証

### 参考スケルトン（ジェネレータ関数）

```python
def generate_bass_plan(bars_df, chordmap, heuristics, tempo_bpm) -> dict:
    events = []  # ← ここに既存ロジックで算出したノート列を詰める（bar/beat/dur_beats/velなど）
    return {
        "ppq": 480, "tempo_bpm": tempo_bpm,
        "tracks": [{
            "name": "Bass", "role": "bass", "channel": 1, "program": 33,
            "events": events
        }]
    }
```

---

## まとめ

* **writer はインフラ、generator は音楽そのもの。**
* これまでの generator 開発は**中核資産**で、今後は**Plan出力に統一**して最大活用。
* 結果、**安定性（writerの強み）×表現力（generatorの強み）**の両取りができます。



短答：**現状は“ほぼ未投入 or 最小限”です。**

* Drums は「スケルトン（基本パターン）」中心で、**明示的なフィル挿入ロジックは未実装**。
* Bass/Guitar も **リフ専用の挿入は未実装**（コード変化時のアクセントや密度可変は入ってます）。
  → なので「フィル／リフをきちんと入れる」パッチを入れると、**密度系KPIの改善**にも効きます。

---

## すぐ入れられる拡張（設計は既存にフィット）

### 1) 設定（arranger_weights.yaml）にフックを追加

```yaml
heuristics:
  drums:
    fill:
      prob_by_section: {intro: 0.10, verse: 0.15, prechorus: 0.35, chorus: 0.25, bridge: 0.40, outro: 0.15}
      min_spacing_bars: 8           # 連発防止
      typical_lengths: [1.0, 0.5]   # 1小節 or 1/2小節
      place_rules: ["bar_before_section_change", "4bar_turnaround"]
      density_boost: 1.4            # その小節だけ密度を上げる係数
  guitar:
    riff:
      prob_by_section: {verse: 0.25, chorus: 0.10, bridge: 0.30}
      min_spacing_bars: 16
      motif_source: "rules"         # "rules" / "stem" / "pickle"
      accent_on_chord_change: true
  bass:
    riff:
      prob_by_section: {verse: 0.20, prechorus: 0.30}
      walkup_prob: 0.35
      approach_note: ["leading_tone","fifth_up","chromatic_up"]
```

### 2) Plan の拡張（互換）

既存 Plan JSON はそのまま使い、**イベントに小さなタグ**を足すだけ：

```json
{
  "bar": 47, "beat": 3.0, "pitch": 38, "dur_beats": 1.0, "vel": 100,
  "phrase": "fill",             // ← 追加
  "priority": 10,               // ← 競合時に優先
  "replace_range": [3.0, 4.0]   // ← 同小節で置換したい拍範囲（任意）
}
```

* `phrase: "fill"|"riff"` で **可視化 & デバッグ**が簡単に。
* `priority` / `replace_range` は **arrangement_orchestrator** で
  同一トラックのベースパターンと**重複したら置換**するのに使えます。

### 3) 生成ロジックに小関数を差し込む

* **Drums**: `select_fill_slots(bars_df, sections)` で候補小節を作り、
  `emit_drum_fill(slot, family, tempo_bpm)` で 1/2〜1 小節のスネア・タム・ハット連打等を生成。

  * Rhythm AI がまだ「fillタグ」持たないなら、**高密度Top-K** から擬似フィル化（最終拍に畳み込む）。
* **Guitar/Bass**: `emit_riff(bar, chord_sym, style)` を追加。

  * Riff プリセット（コード質ごとに 2–3 種）→ 拍位置にクオンタイズ → 既存イベントと衝突したら `replace_range` で置換。

> これらは **Plan 生成側（suno_arranger / generate_*_plans）にだけ追加**すれば OK。
> 書き出しは既存の **midi_writer.py** がそのまま担保します（終端クリップ/テンポメタの健全化あり）。

---

## 期待効果（KPI & 体感）

* **notes_per_bar / density（相対）** が底上げ → いま詰まっている Fail の主要因に直撃。
* 体感も向上：セクション前の **緊張→解放**、ブリッジでの **転換の強調**、
  ギター/ベースの **モチーフ反復（耳に残る要素）** が入る。

---

## 運用の仕方

1. `arranger_weights.yaml` で **確率と間隔**を調整（作曲側だけでチューニング可能）。
2. まずは **Drums の fill** を入れて KPI を確認 → Pass が伸びるはず。
3. つぎに **Guitar/Bass の riff** を軽めに導入（min_spacing_bars を大きめ）。
4. 余裕があれば **Piano/Strings** にも軽い装飾（ターン, アルペジオ 1/2 小節）を追加。

---

## まとめ

* 現状：**フィル/リフは未実装 or 最小**。
* でも、**既存スキーマとパイプラインにそのまま乗る**形で、**すぐ追加可能**です。
* まずは Drums の fill を入れて KPI を上げ、次に Guitar/Bass の riff で音楽的フックを作る、の順が合理的。
* 実装が必要なら、いまの Plan 生成スクリプトに入れる差分パッチ、こちらで用意できます。
