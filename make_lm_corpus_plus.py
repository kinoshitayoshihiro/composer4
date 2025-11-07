# -*- coding: utf-8 -*-
"""
make_lm_corpus_plus.py — 朗読/近代小説向け KenLM 学習用コーパス作成（強化版）

機能:
  ✓ ディレクトリ配下の .txt を再帰収集（複数ディレクトリOK）
  ✓ 旧字→新字: kyujipy（任意） + CSVマップ（上書き）
  ✓ ふりがな/ノンブル等の除去（簡易ルール）
  ✓ 文分割（1文1行）
  ✓ 品質フィルタ: 長さ, ASCII/記号比率, 漢字比, NG正規表現
  ✓ 重複削減: 完全一致 + 近似（Char 3-gram SimHash, ハミング距離閾値）
  ✓ 出力: 文字N-gram用 (char) / Sudachi単語N-gram用 (word)
  ✓ シャッフル（シード指定可）/ シャード分割
  ✓ レポートJSON（件数/フィルタ理由内訳/上位文字頻度 等）

準備:
  pip install kyujipy sudachipy sudachidict-core  # 任意（導入推奨）

使い方（例）:
  python make_lm_corpus_plus.py \
    --input-dirs \
      "/Volumes/SSD-SCTU3A/ラジオ用/texts_a" \
      "/Volumes/SSD-SCTU3A/ラジオ用/texts_b" \
    --out-char \
      "/Volumes/SSD-SCTU3A/ラジオ用/ocr_output/corpus.char.txt" \
    --out-word \
      "/Volumes/SSD-SCTU3A/ラジオ用/ocr_output/corpus.word.txt" \
    --use-kyujipy \
    --kyujitai-csv \
      "/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/ocr_data/maps/kyujitai_map.csv" \
    --ng-regex \
      "/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/ocr_data/maps/ng_regex.txt" \
    --min-len 4 --max-len 600 \
    --min-kanji-ratio 0.05 --max-ascii-ratio 0.5 \
    --dedup-approx --simhash-th 4 \
    --shuffle --seed 42 \
    --shard-size 200000 \
    --report \
      "/Volumes/SSD-SCTU3A/ラジオ用/ocr_output/corpus_report.json"

KenLM 学習（文字N-gram 5-gram の例）:
  BIN="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/ocr_data/kenlm-master/build/bin"
  $BIN/lmplz -o 5 < \
    "/Volumes/SSD-SCTU3A/ラジオ用/ocr_output/corpus.char.txt" \
    > \
    "/Volumes/SSD-SCTU3A/ラジオ用/models/modern_ja.char.arpa"
  $BIN/build_binary \
    "/Volumes/SSD-SCTU3A/ラジオ用/models/modern_ja.char.arpa" \
    "/Volumes/SSD-SCTU3A/ラジオ用/models/modern_ja.char.bin"
"""
from __future__ import annotations

import re
import os
import csv
import sys
import json
import math
import random
import unicodedata
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional, Set

# ---- 任意: kyujipy（あれば使う） ----
try:
    from kyujipy import kyujitai_to_shinjitai  # type: ignore
except Exception:
    kyujitai_to_shinjitai = None

# ---- 任意: Sudachi（word出力時に使う） ----
try:
    from sudachipy import dictionary, tokenizer  # type: ignore

    _sudachi_tk = dictionary.Dictionary().create()
    _sudachi_mode = tokenizer.Tokenizer.SplitMode.C
except Exception:
    _sudachi_tk = None
    _sudachi_mode = None

# ========= 正規表現 =========
KANA = r"ぁ-ゖァ-ヺｰー･・ｦ-ﾟ"
KANJI = r"一-龥々〆ヵヶ"
# VERBOSE モード（?x）をパターン内で有効化して複数行に
SENT_SPLIT = re.compile(
    r"""(?x)
    (?<=[。！？!?])\s+    # 文末の句点／感嘆符の後の空白
  | (?<=\n)\s*\n+         # 空行（段落区切り）
"""
)  # ふりがな・ルビ（簡易）
RUBY_INLINE = [
    (re.compile(rf"(?:｜)?([{KANJI}]+)《[^》]+》"), r"\1"),
    (re.compile(rf"([{KANJI}]+)\s*[（(]\s*[{KANA}]+?\s*[)）]"), r"\1"),
]
# ページ番号など
HEADER_FOOTER = [
    re.compile(r"^\s*第?\s*\d+\s*頁?\s*$"),
    re.compile(r"^\s*\d+\s*$"),
]


# ========= 旧→新 変換表 =========
def load_kyujitai_map(csv_path: Optional[str]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    if not csv_path:
        return m
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or len(row) < 2:
                continue
            old, new = row[0].strip(), row[1].strip()
            if old and new and old != new:
                m[old] = new
    return m


def apply_kyujitai(text: str, csv_map: Dict[str, str], use_kyujipy: bool) -> str:
    s = text
    if use_kyujipy and kyujitai_to_shinjitai is not None:
        s = kyujitai_to_shinjitai(s)
    if csv_map:
        for k in sorted(csv_map.keys(), key=len, reverse=True):
            s = s.replace(k, csv_map[k])
    return s


# ========= NG パターン =========
def load_ng_regex(path: Optional[str]) -> List[re.Pattern]:
    pats: List[re.Pattern] = []
    if not path:
        return pats
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            pats.append(re.compile(ln))
    return pats


# ========= 近似重複: SimHash 実装（文字3-gram） =========
def _char_ngrams(s: str, n: int = 3) -> Iterable[str]:
    if len(s) < n:
        return []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def simhash_64(s: str, n: int = 3) -> int:
    from hashlib import blake2b

    v = [0] * 64
    for g in _char_ngrams(s, n):
        h = int.from_bytes(blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    x = 0
    for i in range(64):
        if v[i] > 0:
            x |= 1 << i
    return x


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class SimHashIndex:
    """簡易LSH: 64bit を 4バンド×16bit でバケツ分け。"""

    def __init__(self):
        self.buckets: List[Dict[int, List[int]]] = [defaultdict(list) for _ in range(4)]
        self.values: List[int] = []

    @staticmethod
    def _bands(x: int) -> Tuple[int, int, int, int]:
        return (
            (x >> 48) & 0xFFFF,
            (x >> 32) & 0xFFFF,
            (x >> 16) & 0xFFFF,
            x & 0xFFFF,
        )

    def add(self, x: int) -> None:
        idx = len(self.values)
        self.values.append(x)
        b = self._bands(x)
        for i in range(4):
            self.buckets[i][b[i]].append(idx)

    def near(self, x: int) -> Iterable[int]:
        b = self._bands(x)
        seen: Set[int] = set()
        for i in range(4):
            for idx in self.buckets[i].get(b[i], []):
                if idx not in seen:
                    seen.add(idx)
                    yield self.values[idx]


# ========= 正規化・分割・フィルタ =========
def normalize_text(s: str, use_kyujipy: bool, csv_map: Dict[str, str]) -> str:
    s = unicodedata.normalize("NFKC", s)
    for pat, rep in RUBY_INLINE:
        s = pat.sub(rep, s)
    s = apply_kyujitai(s, csv_map, use_kyujipy)
    # 記号揺れを軽く整える
    s = s.replace(",", "，").replace(".", "。")
    s = re.sub(r"[…\.]{3,}", "……", s)
    s = re.sub(r"[―ー－]{2,}", "――", s)
    s = s.replace("(", "（").replace(")", "）").replace("〜", "～")
    return s


def strip_headers(txt: str) -> str:
    lines = []
    for ln in txt.splitlines():
        if any(pat.match(ln) for pat in HEADER_FOOTER):
            continue
        lines.append(ln)
    return "\n".join(lines)


def sentence_iter_from_file(
    p: Path, use_kyujipy: bool, csv_map: Dict[str, str], min_len: int
) -> Iterable[str]:
    x = p.read_text(encoding="utf-8", errors="ignore")
    x = strip_headers(x)
    x = normalize_text(x, use_kyujipy, csv_map)
    # 文分割
    segs = [seg.strip() for seg in SENT_SPLIT.split(x) if seg and seg.strip()]
    for s in segs:
        if len(s) >= min_len:
            yield s


def quality_ok(
    s: str,
    min_len: int,
    max_len: int,
    min_kanji_ratio: float,
    max_ascii_ratio: float,
    ng_pats: List[re.Pattern],
) -> Tuple[bool, str]:
    if not (min_len <= len(s) <= max_len):
        return False, "len"
    s2 = re.sub(r"\s", "", s)
    if not s2:
        return False, "blank"
    kanji = len(re.findall(rf"[{KANJI}]", s2))
    ascii_ = len(re.findall(r"[\x00-\x7F]", s2))
    if min_kanji_ratio > 0 and kanji / max(len(s2), 1) < min_kanji_ratio:
        return False, "kanji_ratio"
    if max_ascii_ratio < 1.0 and ascii_ / max(len(s2), 1) > max_ascii_ratio:
        return False, "ascii_ratio"
    for pat in ng_pats:
        if pat.search(s):
            return False, "ng"
    return True, "ok"


# ========= Sudachi 分かち書き =========
def sudachi_tokens(s: str) -> List[str]:
    if _sudachi_tk is None:
        # 未導入なら、簡易フォールバック（文字列ベタ）
        return list(s)
    return [m.surface() for m in _sudachi_tk.tokenize(s, _sudachi_mode)]


# ========= メイン =========
def main():
    ap = argparse.ArgumentParser(description="KenLM用コーパス作成（強化版）")
    ap.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="再帰的に .txt を収集するディレクトリ（複数可）",
    )
    ap.add_argument("--out-char", default=None, help="文字N-gram用の出力パス")
    ap.add_argument("--out-word", default=None, help="単語N-gram用の出力パス（Sudachi推奨）")
    ap.add_argument("--use-kyujipy", action="store_true", help="旧→新に kyujipy を用いる")
    ap.add_argument("--kyujitai-csv", default=None, help="旧→新の上書きCSV (old,new)")
    ap.add_argument("--ng-regex", default=None, help="除外用の正規表現リスト（1行1パターン）")
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=600)
    ap.add_argument("--min-kanji-ratio", type=float, default=0.0)
    ap.add_argument("--max-ascii-ratio", type=float, default=1.0)
    ap.add_argument("--max-repeats", type=int, default=3, help="同一文の最大許容回数（完全一致）")
    ap.add_argument("--dedup-approx", action="store_true", help="近似重複(SimHash)を有効化")
    ap.add_argument(
        "--simhash-th",
        type=int,
        default=4,
        help="近似重複のハミング距離閾値 (0-64、小さいほど厳しい)",
    )
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard-size", type=int, default=0, help=">0でシャード分割（行数）")
    ap.add_argument("--report", default=None, help="集計JSON出力先")
    args = ap.parse_args()

    in_dirs = [Path(p) for p in args.input_dirs]
    paths: List[Path] = []
    for d in in_dirs:
        paths.extend(p for p in d.rglob("*.txt"))
    paths = sorted({p.resolve() for p in paths})
    if not paths:
        print("*.txt が見つかりませんでした", file=sys.stderr)
        sys.exit(1)

    csv_map = load_kyujitai_map(args.kyujitai_csv)
    ng_pats = load_ng_regex(args.ng_regex)

    out_char_f = open(args.out_char, "w", encoding="utf-8") if args.out_char else None
    out_word_f = open(args.out_word, "w", encoding="utf-8") if args.out_word else None

    # 統計
    stats = {
        "files": len(paths),
        "sent_in": 0,
        "sent_out": 0,
        "reasons": Counter(),
        "top_chars": Counter(),
    }

    # 重複管理
    seen_exact: Counter = Counter()  # 完全一致カウント
    seen_sim = SimHashIndex() if args.dedup_approx else None

    # 一旦バッファに詰めてからシャッフル/分割
    buf_char: List[str] = []
    buf_word: List[str] = []

    for p in paths:
        try:
            for s in sentence_iter_from_file(p, args.use_kyujipy, csv_map, args.min_len):
                stats["sent_in"] += 1
                ok, reason = quality_ok(
                    s,
                    args.min_len,
                    args.max_len,
                    args.min_kanji_ratio,
                    args.max_ascii_ratio,
                    ng_pats,
                )
                if not ok:
                    stats["reasons"][reason] += 1
                    continue

                # 完全一致の回数制限
                seen_exact[s] += 1
                if seen_exact[s] > args.max_repeats:
                    stats["reasons"]["repeat_cap"] += 1
                    continue

                # 近似重複
                if seen_sim is not None:
                    h = simhash_64(s)
                    dup = False
                    for hv in seen_sim.near(h):
                        if hamming(h, hv) <= args.simhash_th:
                            dup = True
                            break
                    if dup:
                        stats["reasons"]["simdup"] += 1
                        continue
                    seen_sim.add(h)

                # 出力バッファへ
                if out_char_f:
                    buf_char.append(" ".join(list(s)))
                if out_word_f:
                    toks = sudachi_tokens(s)
                    buf_word.append(" ".join(toks))

                stats["sent_out"] += 1
                stats["top_chars"].update(s)
        except Exception as e:
            print(f"[WARN] {p}: {e}", file=sys.stderr)
            continue

    # シャッフル
    if args.shuffle:
        random.Random(args.seed).shuffle(buf_char)
        random.Random(args.seed).shuffle(buf_word)

    # 書き出し（シャード対応）
    def write_sharded(lines: List[str], fpath: str):
        if not fpath or not lines:
            return
        if args.shard_size and args.shard_size > 0:
            base = Path(fpath)
            base.parent.mkdir(parents=True, exist_ok=True)
            n = 0
            shard = 0
            out = None
            for i, line in enumerate(lines):
                if n == 0:
                    shard += 1
                    if out:
                        out.close()
                    out = open(
                        base.with_suffix(base.suffix + f".part{shard:03d}"), "w", encoding="utf-8"
                    )
                out.write(line + "\n")
                n += 1
                if n >= args.shard_size:
                    n = 0
            if out:
                out.close()
        else:
            Path(fpath).parent.mkdir(parents=True, exist_ok=True)
            with open(fpath, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")

    write_sharded(buf_char, args.out_char or "")
    write_sharded(buf_word, args.out_word or "")

    # レポート
    if args.report:
        rep = {
            "files": stats["files"],
            "sent_in": stats["sent_in"],
            "sent_out": stats["sent_out"],
            "filtered": dict(stats["reasons"]),
            "top_chars": [{"char": c, "count": n} for c, n in stats["top_chars"].most_common(50)],
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Report: {args.report}")

    # 終了メッセージ
    print(
        f"Done. files={stats['files']} in={stats['sent_in']} out={stats['sent_out']} | filtered={sum(stats['reasons'].values())}"
    )


if __name__ == "__main__":
    main()
