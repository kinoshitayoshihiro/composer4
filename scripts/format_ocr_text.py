#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR整形スクリプト

機能:
- ふりがな行の削除（ひらがな/カタカナのみの短い行）
- 行の再結合（空行や区切りで段落を分割）
- 見出しの保持（短く句読点の無い行や「その一」等）
- 約物の正規化（…… と ——）
- 段落頭に全角スペースを付与（見出しは除く）

使い方:
  python3 scripts/format_ocr_text.py --input ocr_data/ocr_output/立春なみだ橋_clean.txt \
                                     --output ocr_data/ocr_output/立春なみだ橋_整形済み.txt
"""

import argparse
import re
from typing import List, Tuple


RE_HIRAKATA_ONLY = re.compile(r"^[\s\u3040-\u309F\u30A0-\u30FFー・、。]+$")


def is_furigana_line(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    # ひらがな/カタカナ/長音符/読点句点のみ かつ 短め
    return len(s) <= 20 and bool(RE_HIRAKATA_ONLY.match(s))


def is_heading_line(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    # 明示的な章節表現
    if re.fullmatch(r"その[一二三四五六七八九十百]+", t):
        return True
    # 漢字のみの短い語（例: 雪空, 浅草涙橋）
    if len(t) <= 10 and re.fullmatch(r"[一-龯々〆ヶ]+", t):
        return True
    return False


def normalize_punct(s: str) -> str:
    # いろいろな点の連続を「……」へ
    s = re.sub(r"[\.…・･·]{3,}", "……", s)
    # ハイフン/ダッシュ類の連続を「——」へ統一（ASCIIハイフン、長音、全角ダッシュ類）
    s = re.sub(r"[\-ー―—]{2,}", "——", s)
    return s


def reflow_paragraphs(lines: List[str]) -> List[Tuple[str, bool]]:
    paragraphs: List[Tuple[str, bool]] = []
    buf: List[str] = []

    def flush_buf():
        nonlocal buf
        if buf:
            text = "".join(buf).strip()
            if text:
                paragraphs.append((text, False))
        buf = []

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            # 空行は段落区切り
            flush_buf()
            continue
        # ダッシュのみの行はノイズとして落とす
        if re.fullmatch(r"[\-ー―—]+", stripped):
            continue
        if is_furigana_line(stripped):
            # ふりがな行は落とす
            continue
        if is_heading_line(stripped):
            flush_buf()
            paragraphs.append((stripped, True))
            continue
        # 通常行は連結（日本語は単語間スペース不要）
        buf.append(stripped)
    flush_buf()
    return paragraphs


def format_ocr_text(text: str) -> str:
    lines = text.splitlines()
    paras = reflow_paragraphs(lines)

    out_lines: List[str] = []
    for p, is_heading in paras:
        p = normalize_punct(p)
        if is_heading:
            out_lines.append(p)
        else:
            out_lines.append("　" + p)  # 段落頭に全角スペース
    return "\n\n".join(out_lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        src = f.read()
    dst = format_ocr_text(src)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(dst)

    # 簡易統計
    total_lines = len(dst.splitlines())
    total_chars = len(dst)
    print(f"✅ 整形完了: {args.output}  行数: {total_lines:,}  文字数: {total_chars:,}")


if __name__ == "__main__":
    main()
