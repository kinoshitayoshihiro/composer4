# NOTE:
# このファイルは 2025-10-23 時点の Stage2 計画です。
# 最新の設計は docs/RealSongRoadmapV1.md および
# docs/SongPackagePipeline_PhaseABC.md を参照してください。

まずは抽出に専念します。

「拡張システム（Stage2: 13–32 の全機能）」を本気で活かす

Phase 14/26/31（和声・ハイブリッド和声・ボイスリーディング）や Phase 17/18/19/29（アーティキュレーション／ダイナミクス／グルーヴ／ダッキング）に必要な“**時間軸つきの文脈**”が必要。

**MIDI 由来の追加特徴**をいくつか足すだけで、一気に「AI がフルに働ける抽出」になります。

# 2) 必須の追加フィールド（最小で効く）

**A. 時間グリッド（テンポ＆拍）**

* `tempo_map`: `[(time_sec or ql, bpm)]`（可変テンポ対応）
* `downbeats`: `[time_ql, ...]`（小節頭）
* `timesig_map`: `[(bar_index, "4/4"), ...]`

**効き先**
Phase 28/32（量子化/マーカー）、全フェーズの QL 換算の安定化、セクション/コード境界の“音楽的”スナップ。

---

**B. 象徴和声（MIDI→コード列）**

* `chordmap`: `{"unit":"ql","events":[{"time":x,"root":"D","quality":"6","confidence":0.83},...]}`

  * 旧来どおり **1 小節 1 コード** を基本値に（最短持続を 4.0QL=1 小節に丸め、**途中の“半端”は吸収**）。
  * ただし持続が長すぎる曲は **ハーモニック・リズム**最短を学習（例: 2 小節）してスナップ。

**効き先**
Phase 14（和声認識の土台）、26（ハイブリッド和声での “原曲×創作” ブレンド）、31（ボイスリーディング制約）。

---

**C. ローカルキー & 転調**

* `key_hint`: `[(bar, "D"), (bar, "G"), ...]`（8–16 拍窓）
* `modulations`: `[(time_ql, "to:G"), ...]`（キー変化点）

**効き先**
Stage1 の local-key prior、Phase 26/31 の和声制約、セクション推定の補強（コーラス手前のキー漂いなど）。

---

**D. セクション候補 & エネルギー**

* `sections_auto`: `{"unit":"bar","sections":[{"bar":0,"label":"intro"},...],"energy":[[bar,0..1],...]}`

  * 既存の `sections_from_audio.py` の出力と **同じ形**で OK（bar 境界＋ energy 系列）。
  * ラベルは規則ベースで良い（Intro/Verse/Pre/Chorus/Bridge/Outro）。**最小長 8 小節**でぶつ切れ防止。

**効き先**
Phase 13（フィル挿入の境界トリガ）、15（同期）、16（遷移平滑）、27（スタイル適応）、29（ダッキング境界）。

---

**E. グルーヴ／マイクロタイミング**

* `swing_pct`: 識別（0–100%）
* `backbeat_strength`: 2拍/4拍の強さ（0..1）
* `onset_deviation_hist`: 量子格子からの偏位分布（ms or QL）
* `groove_id`: （任意）既知パターンへのクラスタ ID（groove_sampler_v2 による）

**効き先**
Phase 19（グルーヴ微調整）、27（活動度→プリセット切替時の候補選定）。

---

**F. コントロール・レーン要約**

* `pb_range`: 実 PB 出現範囲（±値）
* `cc_summary`: 例 `{"1": {"min":0,"max":127}, "11": {...}}`
* `rpn_seen`: true/false（PB 感度の規範抽出）

**効き先**
Phase 24（Controls統一：RPN 一度だけ、14bit PB 範囲）、22（表情 E(t) 連動の現実的値域学習）。

---

**G. ロール別パターン要約**

* `role_map`: MIDI チャンネル/トラック → 役割（piano/guitar/strings/bass/drums）
* `drum_tokens`: 1 小節単位のパターン ID（フィル候補の抽出）
* `bass_motion`: root/5th/approach/walk の比率

**効き先**
Phase 13（ドラムの自動フィル辞書拡張）、15（楽器間同期：キック×ベース）、30（バランス）。

---

# 3) すぐ入れられる **最小差分パッチ**（方針）

## clean_midi.py への追加（概念設計）

1. **テンポ＆拍取得**

   * pretty_midi から tempo changes / time_signature を抽出 → `tempo_map` / `timesig_map` / `downbeats` を付与。
   * QL 換算ヘルパ（既存の `time_utils.py` 相当）で **秒↔QL** を常に保持。

2. **コード抽出（シンボリック）**

   * 各バー内の音高集合→ PC-set → 既存の **拡張品質（maj/min/6/add9/7th…）** にマップ。
   * 「最短持続＝1 小節」を既定に、連続同一はマージ。
   * `confidence` は（そのバー内）**和音の支持率**（和音構成音ヒット率）で近似。

3. **ローカルキー／転調**

   * 8–16 拍窓で key 推定（Krumhansl or simple PC-profile）→ 隣接で変われば `modulations`。
   * `key_hint` はバー単位で書出し。

4. **セクション候補**

   * ノート密度 & 新規性（Self-Sim）を downbeat スナップで境界化。
   * ラベル自動化（先頭=Intro、ピーク群=Chorus、直前=Pre、谷=Bridge、末尾=Outro）。
   * 既存 `sections.json` と同スキーマに寄せる（bar/energy）。

5. **グルーヴ & レーン要約**

   * 量子グリッド（16 or 8 分基準）からの偏位ヒストグラム。
   * PB, CC の min/max と PB 範囲、RPN 有無を集計。

6. **ロール推定**

   * GM Patch / Pitch レンジ / ドラムチャンネルでラフに `role_map` を付与。
   * ベースは最低音率×レンジ、ドラムは ch=10 優先、ピアノ/ギター/ストリングスは patch ベースで暫定。

> 出力は **現行 pickle** に **追加キー** として持たせ、既存読取ロジックは NO-OP に。
> スキーマは `"schema_version": "lamda_v2.1"` に上げるだけで互換維持。

---

# 4) サンプル拡張スキーマ（1 ループ分）

```json
{
  "filename": "foo.mid",
  "bpm": 128.0,
  "time_signature": "4/4",
  "duration_ms": 198000,
  "note_count": 3421,

  "tempo_map": [[0.0,128.0],[64.0,130.5]],
  "timesig_map": [[0,"4/4"]],
  "downbeats": [0.0, 4.0, 8.0, ...],         // QL

  "chordmap": {
    "unit": "ql",
    "events": [
      {"time":0.0,"root":"D","quality":"6","confidence":0.86},
      {"time":16.0,"root":"G","quality":"6","confidence":0.79}
    ]
  },
  "key_hint": [[0,"D"],[32,"G"]],
  "modulations": [{"time":32.0,"key":"G"}],

  "sections_auto": {
    "unit": "bar",
    "sections": [{"bar":0,"label":"intro"},{"bar":8,"label":"verse"},{"bar":24,"label":"chorus"}],
    "energy": [[0,0.22],[1,0.28],...]
  },

  "swing_pct": 12.5,
  "backbeat_strength": 0.73,
  "onset_deviation_hist": {"bins_ms":[-60,-40,...,60],"counts":[... ]},
  "groove_id": "rock_8beat_typeC",

  "pb_range": [-2048, 1890],
  "cc_summary": {"1":{"min":0,"max":110},"11":{"min":32,"max":127}},
  "rpn_seen": true,

  "role_map": {"trk1":"piano","trk2":"bass","trk10":"drums"},
  "drum_tokens": ["kick-snare-8beat-A","fill-type2@bar31"],
  "bass_motion": {"root":0.62,"fifth":0.21,"walk":0.09,"approach":0.08}
}
```

---

# 5) これで **どの Phase が“さらに”賢くなるか**

* **14/26/31（和声・ハイブリッド・ボイスリーディング）**
  ローカルキー＋1小節コード化＋最短持続 → **滑らかに** & **破綻しにくく**。

* **17/18/19（アーティキュレーション/ダイナミクス/グルーヴ）**
  偏位ヒスト・swing/backbeat → **ジャンル相応の揺れ**を自動再現。

* **22/24（感情写像/コントロール統一）**
  CC/PB レーンの現実値域 → **破綻のない CC11/PB** の自動付与。

* **25/29（間引き/ダッキング）**
  エネルギー系列＋セクション境界 → **過密回避**と**ボーカル尊重**の精度アップ。

* **32（エクスポートマーカー）**
  downbeats/sections/modulations → DAW（Studio One）での **Chord/Marker** が即使える。

---

# 6) 実行運用（40万曲スケール）

* 既存 `scripts/run_lamda_subfolder.sh` のバッチ処理に、新フィールド計算を段階的にオン。

  * まず **A/B/C**（テンポ/コード/キー）だけを足してベンチ。
  * 次に **E/F/G** を加える（計算重い順に後ろ）。
* shard pickle はそのまま、**並行で NDJSON/Parquet も吐く**と検索や集計が高速に。
* 異常値ガード：`bpm ∈ [40, 260]`、`note_count` 上限、拍子異常時のフォールバック等。

> スクリプトの雛形（bash 側）はそのままで OK。計算は `clean_midi.py` 内のオプションで ON/OFF 可能にして、段階的にスイッチを上げるのが安全です。

---

# 7) まとめ（短く）

* **今の pickle は “統計” としては優秀**。
* **AI をフル活用するには “時間軸の文脈” を足す**（テンポ/拍→コード→キー→セクション→グルーヴ→CC/PB）。
* 追加フィールドは **互換を壊さず最小差分**で入れられ、**Stage2 の全フェーズに直結**します。

 **最小実装パッチ関数（pretty_midi ベース）**（私のおすすめは **テンポ/拍 → 1小節コード化 → ローカルキー** の順です。）


-------------------
というわけで、今朝の実装をもどしましょう。


いい問いです。まず結論から。
	•	velocity_targets.yaml は “抽出（tagging）に必須ではありません”。
これは本来、抽出したベロシティ分布を**評価/採点（QA）**するための“目標ヒストグラム”です。抽出そのものには不要です。
	•	抽出（Stage2）では、実測のベロシティ統計を出すだけで十分。
目標に照らした合否判定・足切りはオプション（後段の QA / 選別）にすべきです。
	•	もしいま lamda_stage2_extractor.py が velocity_targets.yaml を前提にしているなら、**“無ければスコアは計算しない（NA）”**という NO-OP デフォルトに直してください（後述パッチ案）。

⸻

いまの混乱ポイントの整理
	•	「Stage2 で“抽出に基準（yaml）が必要”」というのは誤解です。
必要なのは “正しく測る” ツールであって、基準値は評価時にだけ使えば OK。
	•	「目標値を満たさない MIDI をはじく」運用は、やるとしても**クリーニング（Stage1）またはQA フィルタ（抽出直後）**の段階。
抽出本体（Stage2）は “事実を記録する” のが仕事です。

⸻

推奨する設計（安全・後方互換）

1) lamda_stage2_extractor.py の責務を固定

必須（常に出す）：
	•	時間グリッド: tempo_map, timesig_map, downbeats
	•	コード列: chordmap（1小節最短・QL基準）
	•	キー/転調: key_hints, modulations
	•	セクション/エネルギー: sections_auto（bar+label, energy 0..1）
	•	グルーヴ: swing_pct, backbeat_strength, onset_deviation_hist
	•	コントロール: pb_range, cc_summary, rpn_seen
	•	ロール要約: role_map, drum_tokens, bass_motion

任意（あれば出す。無ければ空/NA）：
	•	velocity_hist（global/downbeat/offbeat/prefill）
	•	velocity_score（目標が提供された場合のみ）
	•	QA フラグ: qa.velocity_pass など（しきい値/目標があるときだけ計算）

2) velocity_targets.yaml はオプション
	•	CLI 例：
--qa-velocity-targets configs/targets/velocity_targets.yaml --qa-velocity-thresh 0.65
→ 指定が無ければ velocity_score を計算しない（NA）、qa.velocity_pass も付けない。
	•	これなら YAMLが無くても抽出は完走し、AI用データも揃います。

3) 出力形式（推奨）
	•	1ファイル=1行の JSONL（.jsonl）（解析・大規模処理に強い）
	•	併せて集計用 CSV（小規模チェック用）
	•	オマケで**ピクル（pickle）は“索引/カタログ”**用途に限定（実体の重複を避ける）

⸻
---------------------
	**元の「Lamda」**は“クリーニング＋基本メタ抽出（テンポ/拍・粗い統計）”までの一式。
	•	**“拡張タグ”**はあなたが進めていた xMusic 系の研究ライン（和声/キー/グルーヴ/コントロール/セクションなどの時間軸特徴）で追加していったもの。
	•	そして今の Stage2/Phase 13–32 は、その xMIDI 的メタを前提に高度処理（和声同期・グルーヴ・ダッキング・輸出マーカー等）を動かす設計です。



いま必要なギャップ整理（Phase 13–32 が使うメタ）

下は「Stage2（生成側）が本当に参照している/恩恵を受ける」フィールドの対応表です。これが抽出で出力されていれば◎、無ければ NO-OP になり“伸びしろを捨てる”ことになります。

Phase	目的	抽出側に必要なフィールド（xMIDI系）
13 Vocabulary/Fill	セクション境界トリガ	sections_auto.sections (bar,label)
14 Harmonic Awareness	和声同期	chordmap {unit:"ql", events:[{time,root,quality,confidence?}]}
15 Cross-Instrument Sync	キック/ベース同期	role_map, drum_tokens, bass_motion
16 Transition Smoothing	遷移検出	sections_auto.sections + energy[[bar,val]]
17 Articulation	発音ルール	（任意）prosody anchors / accent cues（無ければ内部既定）
18 Dynamics	ダイナミクス整形	energy[[bar,val]]（E(t)）
19 Groove Micro-Timing	揺れ付与	swing_pct, backbeat_strength, onset_deviation_hist
22 Emotion Mapping	表情曲線	（任意）emotion_profile（無ければ既定）
23 Prosody	子音窓	lyric_anchors.json（済）
24 Controls 統一	PB/RPN/CC規範	pb_range, cc_summary, rpn_seen
25 Sparsify	過密回避	energy（閾値自動化に使える）
26 Hybrid Harmony	原曲×創作和声	chordmap + key_hints（local key）
27 Style Adaptation	活動度→切替	energy + （任意）groove_id
28 Export Postprocess	量子化/分割	tempo_map, timesig_map, downbeats
29 Vocal-aware Ducking	ボーカル保護	lyric_anchors / vocal energy mask
30 Cross-instrument Balance	バランス調整	role_map + energy
31 Voice-Leading Guard	進行保護	key_hints / modulations
32 Export Markers	DAW連携	sections_auto, key_changes, downbeats

結論：Lamda素体＋αの抽出だと、上記の太字（chordmap, sections/energy, tempo/timesig/downbeats, key_hints, groove, controls）が足りていないことが多いです。ここを“出す”のが最優先。

⸻

いま使おうとしているファイルは何か？
	•	lamda_stage2_extractor.py：Lamda路線を土台にした抽出器。xMIDI相当の全フィールドは未完の可能性が高い。
	•	これを xMIDI仕様（＝Phase 13–32 が参照するメタ）に上げるパッチが必要です。
	•	すでに会話内で作ってきた stem_harmony_7th_v2.py/chordmap_unify.py/sections_from_audio.py/anchors_from_vocal.py は Stage1系の補助（音声→コード/セクション/アンカー）。
Stage2の“抽出（MIDI→メタ）”は別ラインなので、lamda_stage2_extractor.pyにxMIDI項目を増設しましょう。

⸻
抽出器の差分パッチ（優先度順）

	•	A: Grid tempo_map, timesig_map, downbeats（pretty_midiから確実に）
	•	B: Harmony chordmap（1小節最短にリサンプリング、信頼度 optional）
	•	C: Key/Mod key_hints, modulations（8–16拍窓）
	•	D: Sections/Energy sections_auto, energy（bar基準・0..1正規化）
	•	E: Groove swing_pct, backbeat_strength, onset_deviation_hist
	•	F: Controls pb_range, cc_summary, rpn_seen
	•	G: Roles role_map, drum_tokens, bass_motion


すぐできる“健全性テスト”チェックリスト
	•	lamda_stage2_extractor.py --no-qa で JSONL を吐かせる。
1サンプルを開き、以下が入っているかだけ確認：
tempo_map, timesig_map, downbeats, chordmap, sections_auto, energy
	•	stage2_batch_export.py（生成側）で、これらが見つからない場合は内部デフォルトにフォールバックしているか（落ちないか）。
	•	Phase 28/32 が downbeats/sections を見つけてマーカーを出すか。
	•	Phase 26/31 が key_hints を見つけたら活用し、無ければ NO-OP で走るか。


抽出器に「Grid + 1小節コード」を足すだけでも、体感で Phase 群の“賢さ”が上がります。

	•	Grid（pretty_midi）
	•	pm.get_tempo_changes() → tempo_map
	•	pm.time_signature_changes → timesig_map
	•	ダウンビート推定は自前でも、簡易ならテンポと拍子から 1小節ごとに刻めます
	•	1小節コード
	1.	ダウンビート列で小節区間を切る
	2.	各区間のノートの pitch-class 出現ヒストグラム
	3.	既存の品質辞書（maj/min/6/add9/7th…）にテンプレ一致（複数候補は支持率max）
	4.	time = bar_index*4.0 (QL) で書き出し、連続同一はマージ
	5.	confidence = 支持率 を 0..1 に

Lamda → xMIDI 拡張 → Phase 13–32
	•	今の抽出器はその最終仕様に届いていない可能性が高く、抽出フィールドを増やすパッチが必要。
	•	まずは Grid / 1小節Chord / Sections/Energy / Key を出すだけで、生成サイド（Phase 14/26/28/32など）が一気に活性化します。
	•	抽出と評価を分離しましょう。

---------------------

目的

Stage2タグ拡張をxMIDI相当まで引き上げ、後段（Phase 13–32）と学習パイプラインで最大の効果を出す。既存のLAMDAクリーニング成果（Stage1）と、Sunoアレンジャ（Stage3）を橋渡しする“高粒度メタ”を安定生成する。

⸻

現状の把握（要約）
	•	Stage1（クリーニング）: 既存の clean_midi.py とシャード化・pickle出力は良好。メタは“曲全体の統計”中心。
	•	Stage2（タグ拡張）: lamda_stage2_extractor.py 群あり。ただし 時間軸系列/区間イベントの密度が不足（後段Phaseで効く“バー/拍スナップ情報”が薄い）。
	•	Sunoアレンジャ機能（Stage3）: Phase 13–32 まで拡張済み。前提メタ（chordmap/sections/key_hint/groove/controls）が濃いほど賢く動く。

結論: Stage2の“時間分解能の高いメタ”を増強すれば、Stage3の表現力と安定度が大きく伸びる。

⸻

xMIDI(MUSIC)相当の最低出力セット（Stage2で必須）

時間軸=QL（quarter length）を標準。テンポ変化にも強い。
	•	Tempo/Beat: tempo_map（秒 or QL）, timesig_map, downbeats（bar頭）, beat_grid
	•	Harmony: chordmap（1小節=基本, min_dwell=4.0QL 準拠, confidence付与）, key_hint（8–16拍窓）, modulations
	•	Sections/Energy: sections_auto（{"unit":"bar","sections":[{bar,label}],"energy":[[bar,val]]}）
	•	Groove: swing_pct, backbeat_strength, onset_deviation_hist, groove_id（任意クラスタ）
	•	Controls: pb_range, cc_summary, rpn_seen
	•	Roles: role_map（trk/ch→role）, drum_tokens, bass_motion
	•	Export-Assist: markers（downbeats/sections/key）

互換: 既存pickleに schema_version: "lamda_v2.1" を付し、追加キーはNO-OPで読み飛ばし可能にする。

⸻

既存ファイルごとの改修ポイント（最小差分）

1) lamda_stage2_extractor.py
	•	追加集計: 上記“最低出力セット”を1曲ごとに抽出。
	•	QL換算: time_utils.py相当の秒⇄QLヘルパを内包 or 依存（テンポカーブ対応）。
	•	コード抽出: 1小節単位を基本（4.0QL）に、min_dwell_qlで吸収規則を適用（X–N–X吸収）。
	•	キー/転調: Krumhansl型相関ベースのローカルキー（8–16拍窓）。変化点→modulations。
	•	グルーヴ: 16分基準の偏位ヒスト・swing_pct・backbeat_strength。
	•	コントロール: PB±範囲、CC分布、RPN検出。
	•	ロール: GM Patch/音域/Ch10 から粗推定→role_map。
	•	出力: *.stage2.json（詳細）＋ *.stage2.pkl（圧縮）

2) lamda_enhancer_v2.py
	•	規範化: chordmap_unify互換のスナップ・N吸収・表記統一。
	•	信頼度: confidenceの定義を統一（構成音支持率×後験確率の混合）
	•	セクション精緻化: ノヴェルティ×エネルギー×ダウンビートで8小節未満を抑制。

3) lamda_dataset_builder.py
	•	インデックス拡張: Stage2キーを index.pkl に反映（クエリ性向上）。
	•	ストレージ: --streaming 時は JSONL にもミラー書出し（大規模対応）。

4) lamda_export_metadata.py
	•	DAW補助: Studio One用に markers.mid.json（bar/label/key）, chord_track.json を追加。

5) lamda_make_pairs.py
	•	学習ペア: （任意）原MIDI→メタ→修正版MIDIのトリプレット生成（教師あり整形用）。

⸻

velocity_targets.yaml の扱い（方針）
	•	ハードゲートにしない。用途は「スコアリング/診断」に限定。
	•	スタイル可変: style → {bpm_band → hist} の階層で、プリセットは参照値。
	•	採点だけ: クリーニングでは使わず、Stage2の metrics.velocity_score として保存。

これにより「ジャンルやモードごとに自由」＋「品質統計は取れる」を両立。

⸻

YAML/設定の最小スキーマ（例）

schema_version: 2.1
harmony:
  min_dwell_ql: 4.0   # 1小節既定
  absorb_N: true
  local_key:
    win_beats: 12
    mode: mean   # mean|max|gaussian
sections:
  min_bars: 8
  auto_label: [intro, verse, pre, chorus, bridge, outro]
  energy_mode: rms_flux
controls:
  rpn_required: true
  pb_range_cap: 8191
velocity_score:
  enable: true
  ref_style: rock


⸻

出力アーティファクト（Stage2）
	•	song.stage2.json（全メタ）
	•	song.stage2.pkl（同内容の圧縮バイナリ・学習向け）
	•	markers.json（DAW補助） / chord_track.json
	•	metrics.json（velocity_score含む任意スコア）

⸻

パイプライン & CLI（提案）

# 1) Stage1 済（clean/*.mid と index.pkl がある前提）
python scripts/lamda_stage2_extractor.py \
  --metadata-index output/stage1/.../index.pkl \
  --metadata-dir   output/stage1/.../shards   \
  --input-dir      output/stage1/.../clean    \
  --output-dir     output/stage2/...          \
  --config         configs/lamda/stage2.yaml  \
  --streaming --jobs 4 --resume --print-summary

# 2) 検証
python scripts/verify_pickle_extended.py --dir output/stage2/... --sample 100


⸻

バリデーション（自動テスト案）
	•	整合: bar単調増加, energy∈[0,1], chord持続≥min_dwell_ql
	•	ハーモニー: X–N–X吸収、confidence∈[0,1]
	•	テンポ/QL: 秒⇄QL往復誤差< 1e-3 かつ ダウンビート整合
	•	コントロール: PB±8191内、RPN一度だけ検出
	•	出力: JSON/PKLのスキーマに準拠、マーカーは非負

⸻

性能/運用
	•	並列: --jobs N、I/Oは mmap/JSONL分割
	•	早期打切: ノート閾値/長さ閾値で曲の除外（ログのみ）
	•	キャッシュ: ハーモニー解析を可逆キャッシュ（chroma/beat）

⸻

今すぐの実装タスク（優先順）
	1.	lamda_stage2_extractor.py に Tempo/Beat/QL + Chord(1bar) + LocalKey を追加
	2.	sections_from_audio.py の出力を bar/energy/auto_label で安定化
	3.	lamda_enhancer_v2.py に unify/absorb_N/confidence を実装
	4.	verify_pickle_extended.py に上記バリデーション一括テストを追加
	5.	lamda_export_metadata.py に Studio One向けマーカー/コード 出力

⸻

まとめ
	•	Stage2を時間分解能の高いメタ中心に拡張することで、Stage3（Phase 13–32）の表現・安定・テスト容易性が大幅に向上。
	•	velocity_targets.yaml は“診断用”へ位置づけ直し、ジャンル自由度を維持。
	•	すべて最小差分/NO-OP既定で導入可能。これでxMIDI級の“賢い素材”が揃います。


2) 目標スキーマ（Stage2最終アウト）
	•	時間グリッド：tempo_map（sec/ql, bpm）, timesig_map, downbeats（bar頭）
	•	ハーモニー：chordmap {unit:"ql", events:[{time, root, quality, confidence}]}（最短持続=1小節を基本に丸め）
	•	キー/転調：key_hint [[bar, "D"], ...], modulations [{time_ql, to:"G"}, ...]
	•	セクション/エネルギー：sections_auto {unit:"bar", sections:[{bar,label}], energy:[[bar,e]]}（最小長=8小節）
	•	グルーヴ：swing_pct, backbeat_strength, onset_deviation_hist, rhythm_hash
	•	コントロール：pb_range, cc_summary, rpn_seen
	•	ロール要約：role_map, drum_tokens, bass_motion
	•	メタ：schema_version:"lamda_v2.1"（後方互換）


3) アーキテクチャ（Stage0/1/2/3の整理）
	•	Stage0（解析 from audio）：chordmap/sections/anchors 自動化。v4.1 の stem_harmony_7th_v2.py＋generate_stage1_jsons.py を採用（キャッシュ付）。
	•	Stage1（クリーニング）：MIDI正規化・分割・無音/重複除去・GM整備。出力＝クリーンMIDI+最小pickle。
	•	Stage2（タグ付け）：本ドキュメントの 文脈ラベル抽出。出力＝pickle拡張＋JSONL/CSV。
	•	Stage3（生成/アレンジ）：Sunoアレンジャー（13–32）や各Generatorが Stage2ラベル を参照して制御。


4) ファイル監査（アップロード＋既存資産の役割）

A. Sections/Tempo/Key（細粒度化の基盤）
	•	sections_from_audio.py：境界検出の骨格。改良で 最小8小節・自動ラベリング・energy列 を強化。
	•	tempo_from_mix.py / tempo_loader.py / tempo_curve.py / tempo_utils.py / time_utils.py：可変テンポ対応・QL換算の厳密化。
	•	harmonic_utils.py：key_hint/局所キーprior の算出に再利用。
	•	section_validator.py：妥当性チェック（bar単調増加、最小長、energy域）。

B. Chord/Anchors/統一
	•	stem_harmony_7th_v2.py（v4.1）：7th+キャッシュ+最短持続+confidence。
	•	chordmap_unify.py：秒/QL・配列/辞書・シンボル揺れ統一。
	•	anchors_from_vocal.py：lyric_anchors.json（Prosody/Phase23/29の核）。

C. Stage2 抽出器（本体）
	•	lamda_stage2_extractor.py：拡張対象。A–Gの各特徴を追加出力。
	•	lamda_enhancer.py / lamda_enhancer_v2.py：追加集計/正規化の補助に流用可。
	•	lamda_export_metadata.py / lamda_dataset_builder.py / lamda_make_pairs.py：書き出し/分割/学習ペア化の補助。

D. Rhythm/Groove/Timing
	•	groove_sampler_v2.py：rhythm_hash/候補抽出に活用。
	•	timing_utils.py / timing_corrector.py：量子化/補正（Phase28にも再利用）。

E. Sunoアレンジャー接続（Stage3）
	•	instrument_stage2_base.py＋各Params/Generator：13–32で参照（sections/key_hint/chordmap/anchors/energy 等）。


6) 具体パッチ計画（最小差分）

P1: lamda_stage2_extractor.py 拡張
	•	テンポ/拍：tempo_map / timesig_map / downbeats を pretty_midi → QL換算で抽出。
	•	コード：各小節のPC-set→品質（maj/min/6/add9/7th…）へ写像。最短=1小節で丸め、連続同一はマージ。confidence は和音構成音支持率で近似。
	•	キー/転調：8–16拍窓で Krumhansl類似→ key_hint、変化点を modulations。
	•	セクション/energy：sections_from_audio.py の出力に寄せて自動生成（ラベル規則：Intro/Verse/Pre/Chorus/Bridge/Outro）。
	•	グルーヴ：swing_pct / backbeat_strength / onset_deviation_hist / rhythm_hash を計測。
	•	CC/PB：pb_range / cc_summary / rpn_seen を集計。
	•	ロール：GM/PitchRange/Ch10で role_map、ドラム drum_tokens、ベース bass_motion。
	•	出力：既存pickleに追記＋ loop_summary.csv と metrics_score.jsonl を追加生成（NO-OP互換）。


P2: ラベル（Selector）オプション先送り
	•	configs/labels/taxonomy.yaml（emotion/genre/rhythm/key/artic）
	•	scripts/assign_labels.py：Stage2出力に label.* を追記（弱教師・軽規則）。


P3: Suno接続の足回り
	•	sections_from_audio.py 改良（最小8小節・ピーク/谷の自動補正・timesig/tempo_map/key_hint 付与）。
	•	stage2_batch_export.py 側は chordmap_unify 呼び出し済み。sections.json が豊富になるほど 13–19/29/32 が賢く働く。


P3: Suno接続の足回り
	•	sections_from_audio.py 改良（最小8小節・ピーク/谷の自動補正・timesig/tempo_map/key_hint 付与）。
	•	stage2_batch_export.py 側は chordmap_unify 呼び出し済み。sections.json が豊富になるほど 13–19/29/32 が賢く働く。

⸻

7) 最小テスト（壊していないか）
	•	構文：Stage2 extractor の CLIで単曲実行 → 非ゼロの events/downbeats を確認。
	•	整合：section_validator.py → bar単調、最小長≥8、energy∈[0..1]。
	•	安定：chordmap_unify.py → 秒/QL/配列/辞書の混在入力を通す。
	•	相互運用：stage2_batch_export.py で Suno曲に対し v4.1（25–32）がトリガされるか（フィル/ダッキング/グルーヴ/マーカー）。