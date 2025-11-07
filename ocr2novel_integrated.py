"""
ocr2novel_integrated.py — ローカル画像OCR → レイアウト安定化 → ルビ除去 → 旧字→新字 → KenLM文脈補正 → 小説体裁

依存:
  pip install google-cloud-vision opencv-python-headless kyujipy kenlm
（任意）sudachipy sudachidict-core  # トークン化を強めたい場合

KenLM学習（例 / macOS）:
  brew install kenlm
  # 近代日本文学などのテキストを1行1文に整形したコーパスで
  lmplz -o 5 < corpus.txt > modern_ja.arpa
  build_binary modern_ja.arpa modern_ja.bin

実行例:
  export GOOGLE_APPLICATION_CREDENTIALS="/Volumes/SSD-SCTU3A/ラジオ用/charged-camera.json"
  python ocr2novel_integrated.py \
    --input-dir "/Volumes/SSD-SCTU3A/ラジオ用/ocr_targets" \
    --output "/Volumes/SSD-SCTU3A/ラジオ用/ocr_output/novel.txt" \
    --debug-dir "/Volumes/SSD-SCTU3A/ラジオ用/ocr_output/debug_layout" \
    --kenlm "/Volumes/SSD-SCTU3A/ラジオ用/models/modern_ja.bin" \
    --kyujitai-csv "/Volumes/SSD-SCTU3A/ラジオ用/maps/kyujitai_map.csv" \
    --blocklist "/Volumes/SSD-SCTU3A/ラジオ用/maps/blocklist.txt"
"""

from __future__ import annotations

import os
import re
import csv
import sys
import math
import unicodedata
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Iterable, Dict, Optional, Set

# ========= 依存ライブラリ =========
try:
    import cv2  # type: ignore
except Exception as e:
    raise SystemExit("OpenCV(cv2) が必要です: pip install opencv-python-headless")

try:
    import numpy as np  # type: ignore
except Exception as e:
    raise SystemExit("NumPy が必要です: pip install numpy")

try:
    from google.cloud import vision  # type: ignore
except Exception:
    raise SystemExit("google-cloud-vision が必要です: pip install google-cloud-vision")

# kyujipyは任意。無ければCSVマップで代替
try:
    from kyujipy import kyujitai_to_shinjitai  # type: ignore
except Exception:
    kyujitai_to_shinjitai = None  # 型: ignore

# KenLMは任意
try:
    import kenlm  # type: ignore
except Exception:
    kenlm = None  # 型: ignore

# ========= 正規表現プロファイル =========
KANA_CHARS = r"ぁ-ゖァ-ヺｰー･・ｦ-ﾟ"
KANJI_CHARS = r"一-龥々〆ヵヶ"


# ========= ユーティリティ =========
def chunk_list(lst: List, size: int) -> Iterable[List]:
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ========= データ構造 =========
@dataclass
class Word:
    text: str
    cx: float
    cy: float
    w: float
    h: float
    is_kana_only: bool


# ========= 画像前処理（傾き補正） =========
def deskew_image_bytes(img_bytes: bytes) -> bytes:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return img_bytes
    # 自動2値化
    img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(img_bin == 0))
    if coords.size == 0:
        return img_bytes
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5:
        return img_bytes
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    _, out = cv2.imencode(".png", rotated)
    return bytes(out.tobytes())


# ========= OCR抽出 =========
def is_kana_only(s: str) -> bool:
    s2 = re.sub(r"\s", "", s)
    return bool(s2) and re.fullmatch(rf"[{KANA_CHARS}]+", s2) is not None


def poly_to_box(vertices) -> Tuple[float, float, float, float]:
    xs = [v.x for v in vertices]
    ys = [v.y for v in vertices]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)


def extract_words(res: "vision.AnnotateImageResponse") -> List[Word]:
    words: List[Word] = []
    fa = res.full_text_annotation
    if not fa or not fa.pages:
        return words
    for pg in fa.pages:
        for block in pg.blocks:
            for para in block.paragraphs:
                for w in para.words:
                    txt = "".join([s.text for s in w.symbols])
                    cx, cy, bw, bh = poly_to_box(w.bounding_box.vertices)
                    words.append(Word(txt, cx, cy, bw, bh, is_kana_only(txt)))
    return words


# ========= 余白・ルビ除去 =========
def drop_repeated_margins(words: List[Word], img_w: float, img_h: float) -> List[Word]:
    if not words:
        return words
    top_band = img_h * 0.08
    bot_band = img_h * 0.90
    tops = [w for w in words if w.cy < top_band]
    bots = [w for w in words if w.cy > bot_band]
    from collections import Counter

    def frequent_tokens(ws: List[Word]) -> set:
        c = Counter([w.text for w in ws if len(w.text) <= 8])
        return {t for t, n in c.items() if n >= 2}

    ban = frequent_tokens(tops) | frequent_tokens(bots)
    return [w for w in words if not ((w.text in ban and (w in tops or w in bots)))]


def remove_ruby(words: List[Word], vertical_hint: Optional[bool] = None) -> List[Word]:
    if not words:
        return words
    hs = sorted([w.h for w in words])
    med_h = hs[len(hs) // 2] if hs else 1.0
    small_thr = med_h * 0.65
    if vertical_hint is None:
        tall = sum(1 for w in words if w.h > w.w)
        vertical = tall > (len(words) * 0.55)
    else:
        vertical = vertical_hint
    kept: List[Word] = []
    for w in words:
        if w.is_kana_only and w.h < small_thr:
            near = [
                v
                for v in words
                if abs(v.cx - w.cx) < (med_h * 1.2) and abs(v.cy - w.cy) < (med_h * 1.2)
            ]
            has_kanji_base = any(
                (re.search(rf"[{KANJI_CHARS}]", v.text) and v.h >= med_h * 0.9) for v in near
            )
            if has_kanji_base:
                # 横:ルビは上側, 縦:左側（近傍に基底があると捨てる）
                if (not vertical and any(v.cy > w.cy for v in near)) or (
                    vertical and any(v.cx > w.cx for v in near)
                ):
                    continue
        kept.append(w)
    return kept


# ========= 読順再構築（縦横自動 / 列分割 / 短行レスキュー） =========
def detect_vertical(words: List[Word]) -> bool:
    if not words:
        return False
    tall = sum(1 for w in words if w.h > w.w)
    return tall >= 0.55 * len(words)


def group_lines(words: List[Word], vertical: bool) -> List[List[Word]]:
    if not words:
        return []
    med_h = sorted([w.h for w in words])[len(words) // 2]
    thr = med_h * (0.60 if vertical else 0.65)
    key = (lambda w: w.cx) if vertical else (lambda w: w.cy)
    ws = sorted(words, key=lambda w: (key(w), w.cy if vertical else w.cx))
    lines: List[List[Word]] = []
    cur: List[Word] = []
    for a, b in zip(ws, ws[1:] + [None]):
        cur.append(a)
        if b is None or abs(key(b) - key(a)) > thr:
            cur_sorted = sorted(cur, key=lambda w: (w.cy if vertical else w.cx))
            lines.append(cur_sorted)
            cur = []
    return lines


def split_columns(lines: List[List[Word]], vertical: bool) -> List[List[List[Word]]]:
    if not lines:
        return []
    line_cx = [(sum(w.cx for w in ln) / len(ln)) for ln in lines]
    order = sorted(range(len(lines)), key=lambda i: line_cx[i])
    xs = [line_cx[i] for i in order]
    if len(xs) <= 1:
        return [lines]
    diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    med = sorted(diffs)[len(diffs) // 2] if diffs else 1.0
    cut = med * 2.2
    cols: List[List[List[Word]]] = []
    buf: List[List[Word]] = [lines[order[0]]]
    for i, d in enumerate(diffs):
        if d > cut:
            cols.append(buf)
            buf = []
        buf.append(lines[order[i + 1]])
    if buf:
        cols.append(buf)
    if vertical:
        cols = list(reversed(cols))
    return cols


# ========= 可視化（列境界・行番号） =========
COLS = [(255, 0, 0), (0, 180, 0), (0, 0, 255), (255, 128, 0), (128, 0, 255), (0, 128, 255)]


def save_layout_debug(img_bytes: bytes, columns: List[List[List[Word]]], out_path: str):
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return
    H, W = img.shape[:2]
    for ci, col in enumerate(columns):
        xs, ys = [], []
        for ln in col:
            for w in ln:
                xs += [w.cx - w.w / 2, w.cx + w.w / 2]
                ys += [w.cy - w.h / 2, w.cy + w.h / 2]
        if not xs:
            continue
        x1, x2 = int(max(0, min(xs))), int(min(W - 1, max(xs)))
        y1, y2 = int(max(0, min(ys))), int(min(H - 1, max(ys)))
        color = COLS[ci % len(COLS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            img,
            f"COL {ci+1}",
            (x1 + 5, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )
    line_idx = 1
    for ci, col in enumerate(columns):
        color = COLS[ci % len(COLS)]
        for ln in col:
            xs, ys = [], []
            for w in ln:
                xs += [w.cx - w.w / 2, w.cx + w.w / 2]
                ys += [w.cy - w.h / 2, w.cy + w.h / 2]
            if not xs:
                continue
            x1, x2 = int(max(0, min(xs))), int(min(W - 1, max(xs)))
            y1, y2 = int(max(0, min(ys))), int(min(H - 1, max(ys)))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                img,
                f"{line_idx}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            line_idx += 1
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, img)


# ========= 行→段落 =========
PARA_END = re.compile(r"[。！？…〕】）」』〉》】]+$|――$")


def normalize_punct(s: str) -> str:
    s = s.replace(",", "，").replace(".", "。")
    s = re.sub(r"[…\.]{3,}", "……", s)
    s = re.sub(r"[―ー－]{2,}", "――", s)
    s = s.replace("(", "（").replace(")", "）").replace("〜", "～")
    # インラインルビ記号の除去
    s = re.sub(rf"(?:｜)?([{KANJI_CHARS}]+)《[^》]+》", r"\1", s)
    s = re.sub(rf"([{KANJI_CHARS}]+)\s*[（(]\s*[{KANA_CHARS}]+?\s*[)）]", r"\1", s)
    s = re.sub(rf"([{KANJI_CHARS}]+)\s*[〔【〈《]\s*[{KANA_CHARS}]+?\s*[】〉》〕]", r"\1", s)
    return s


def words_to_ordered_lines(words: List[Word]) -> List[str]:
    if not words:
        return []
    vertical = detect_vertical(words)
    lines = group_lines(words, vertical)
    columns = split_columns(lines, vertical)
    out_lines: List[str] = []
    # 列内: 上→下、行内: 横=左→右 / 縦=上→下 の順で結合
    for col in columns:
        col = sorted(col, key=lambda ln: sum(w.cy for w in ln) / len(ln))
        for ln in col:
            ws = sorted(ln, key=(lambda w: w.cy) if vertical else (lambda w: w.cx))
            out_lines.append("".join(w.text for w in ws))
    return out_lines


def rebuild_paragraphs(lines: List[str], indent: bool = True) -> str:
    paras, buf = [], []
    for raw in lines:
        line = raw.rstrip()
        if not line:
            if buf:
                paras.append("".join(buf).strip())
                buf = []
            continue
        line = normalize_punct(line)
        buf.append(line)
        if PARA_END.search(line):
            paras.append("".join(buf).strip())
            buf = []
    if buf:
        paras.append("".join(buf).strip())
    if indent:
        paras = [
            ("　" + p) if p and not p.startswith(("　", "「", "『", "（")) else p for p in paras
        ]
    return "\n\n".join(paras)


# ========= 旧字→新字 =========
def load_kyujitai_map(csv_path: Optional[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not csv_path:
        return mapping
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            old, new = row[0].strip(), row[1].strip()
            if old and new:
                mapping[old] = new
    return mapping


def convert_kyujitai(text: str, csv_mapping: Dict[str, str], blocklist: Set[str]) -> str:
    out = text
    # 1) kyujipyがあれば一次変換
    if kyujitai_to_shinjitai is not None:
        out = kyujitai_to_shinjitai(out)
    # 2) CSVマップで上書き・補完（キー長い順）
    if csv_mapping:
        for k in sorted(csv_mapping.keys(), key=len, reverse=True):
            if k in blocklist:
                continue
            out = out.replace(k, csv_mapping[k])
    # 3) 正規化
    out = unicodedata.normalize("NFKC", out)
    return out


# ========= KenLM 文脈補正 =========
CONFUSIONS: Dict[str, List[str]] = {
    "一": ["ー"],
    "ー": ["一"],
    "ソ": ["ン"],
    "ン": ["ソ"],
    "シ": ["ツ"],
    "ツ": ["シ"],
    "力": ["カ"],
    "口": ["ロ"],
    "二": ["ニ"],
    "へ": ["べ", "ぺ"],
    "髙": ["高"],
    "𠮷": ["吉"],
    "﨑": ["崎"],
}

JP_SENT_SEP = re.compile(r"([。！？!?]+)\s*")


class KenLMScorer:
    def __init__(self, model_path: Optional[str]):
        self.enabled = False
        self.model = None
        if model_path and kenlm is not None and Path(model_path).exists():
            self.model = kenlm.Model(model_path)
            self.enabled = True

    def score(self, text: str) -> float:
        if not self.enabled or self.model is None:
            return 0.0
        return self.model.score(text, bos=True, eos=True)

    def improve_sentence(self, sent: str, blocklist: Set[str]) -> str:
        if not self.enabled or self.model is None:
            return sent
        best = sent
        best_score = self.score(best)
        # 簡単なトークン分割（日本語なので厳密ではないが十分）
        # 句読点・括弧・空白で区切る
        tokens = re.split(r"([\s、，。．！？!?,.;:：「」『』（）\(\)\[\]])", sent)
        for i, tok in enumerate(tokens):
            if not tok or tok.strip() == "" or tok in blocklist:
                continue
            # 長さ2以下の短語、または混同文字を含む語だけ試す
            if len(tok) <= 2 or any(c in CONFUSIONS for c in tok):
                alts = {tok}
                for idx, ch in enumerate(tok):
                    if ch in CONFUSIONS:
                        for alt in CONFUSIONS[ch]:
                            alts.add(tok[:idx] + alt + tok[idx + 1 :])
                for cand in alts:
                    if cand == tok:
                        continue
                    trial_tokens = tokens.copy()
                    trial_tokens[i] = cand
                    trial = "".join(trial_tokens)
                    sc = self.score(trial)
                    if sc > best_score:
                        best, best_score = trial, sc
        return best

    def improve_text(self, text: str, blocklist: Set[str]) -> str:
        if not self.enabled or self.model is None:
            return text
        # 文ごとに最適化
        parts = JP_SENT_SEP.split(text)
        # parts: [seg, sep, seg, sep, ...]
        out = []
        buf = ""
        for i in range(0, len(parts), 2):
            seg = parts[i]
            sep = parts[i + 1] if i + 1 < len(parts) else ""
            if seg.strip():
                improved = self.improve_sentence(seg, blocklist)
                out.append(improved + sep)
            else:
                out.append(sep)
        return "".join(out)


# ========= メインパイプライン =========
def process_page(
    img_bytes: bytes,
    vision_client: "vision.ImageAnnotatorClient",
    language_hints: List[str],
    debug_dir: Optional[Path],
) -> str:
    # 傾き補正
    fixed = deskew_image_bytes(img_bytes)
    img = cv2.imdecode(np.frombuffer(fixed, np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2] if img is not None else (1000, 700)

    image = vision.Image(content=fixed)
    ctx = vision.ImageContext(language_hints=language_hints)
    req = vision.AnnotateImageRequest(
        image=image,
        features=[vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)],
        image_context=ctx,
    )
    resp = vision_client.batch_annotate_images(requests=[req])
    r = resp.responses[0]
    if r.error.message:
        return ""
    words = extract_words(r)
    words = drop_repeated_margins(words, w, h)
    words = remove_ruby(words, vertical_hint=None)
    # 読順・列
    vertical = detect_vertical(words)
    lines = group_lines(words, vertical)
    columns = split_columns(lines, vertical)
    # デバッグ可視化
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        idx = len(list(debug_dir.glob("page_*.png"))) + 1
        save_layout_debug(fixed, columns, str(debug_dir / f"page_{idx:04d}.png"))
    # 行→段落
    line_texts = []
    for col in columns:
        col = sorted(col, key=lambda ln: sum(w.cy for w in ln) / len(ln))
        for ln in col:
            ws = sorted(ln, key=(lambda w: w.cy) if vertical else (lambda w: w.cx))
            line_texts.append("".join(w.text for w in ws))
    page_text = rebuild_paragraphs(line_texts, indent=True)
    return page_text


def main():
    ap = argparse.ArgumentParser(
        description="Local OCR → layout → ruby removal → kyujitai→shinjitai → KenLM → novel styling"
    )
    ap.add_argument("--input-dir", required=True, help="画像フォルダ（再帰）")
    ap.add_argument("--output", required=True, help="出力テキスト")
    ap.add_argument("--batch", type=int, default=8, help="OCRバッチ（推奨<=16）")
    ap.add_argument(
        "--language-hints", default="ja,en", help="Visionへの言語ヒント（カンマ区切り）"
    )
    ap.add_argument("--debug-dir", default=None, help="版面レイアウトの可視化PNG出力先")
    ap.add_argument("--kyujitai-csv", default=None, help="旧→新のCSV（old,new）")
    ap.add_argument("--blocklist", default=None, help="変換から除外する語（1行1語）")
    ap.add_argument("--kenlm", default=None, help="KenLMモデル（.arpa or .bin）")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_path = Path(args.output)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    # 画像一覧
    valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
    paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in valid_ext)
    if not paths:
        print("対象の画像が見つかりません。", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(paths)} image(s) under: {input_dir}")

    # Visionクライアント
    vision_client = vision.ImageAnnotatorClient()
    lang_hints = [s.strip() for s in args.language_hints.split(",") if s.strip()]

    # 1) ページごとにOCR+整形
    pages: List[str] = []
    for p in paths:
        try:
            txt = process_page(p.read_bytes(), vision_client, lang_hints, debug_dir)
        except Exception as e:
            print(f"[WARN] {p}: {e}")
            txt = ""
        pages.append(txt)
    text = "\n\n".join(t for t in pages if t.strip())

    # 2) 旧字→新字
    csv_map = load_kyujitai_map(args.kyujitai_csv)
    blocklist: Set[str] = set()
    if args.blocklist and Path(args.blocklist).exists():
        blocklist = {
            ln.strip()
            for ln in Path(args.blocklist).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }
    text = convert_kyujitai(text, csv_map, blocklist)

    # 3) KenLM で文脈補正（任意）
    scorer = KenLMScorer(args.kenlm)
    if scorer.enabled:
        text = scorer.improve_text(text, blocklist)
    else:
        if args.kenlm:
            print(
                "[WARN] KenLMモデルを読み込めませんでした。kenlm未インストール or パス不正。スキップします。"
            )

    # 4) 出力
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"✅ 出力: {out_path}  ({len(text)} chars)")


if __name__ == "__main__":
    # 認証は環境変数 GOOGLE_APPLICATION_CREDENTIALS を利用
    main()
