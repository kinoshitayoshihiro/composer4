#!/usr/bin/env python3
"""
シンプルで正確なOCR - Google Vision APIの結果をそのまま利用
旧字→新字変換のみ実施、レイアウト破壊処理は一切なし
"""

import sys
import csv
import unicodedata
from pathlib import Path
from typing import Dict, Set, Optional
from google.cloud import vision

# kyujipyは任意
try:
    from kyujipy import kyujitai_to_shinjitai
except ImportError:
    kyujitai_to_shinjitai = None


def load_kyujitai_map(csv_path: Optional[str]) -> Dict[str, str]:
    """旧字→新字のCSVマップを読み込み"""
    mapping: Dict[str, str] = {}
    if not csv_path or not Path(csv_path).exists():
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


def load_blocklist(blocklist_path: Optional[str]) -> Set[str]:
    """変換から除外する語のリストを読み込み"""
    if not blocklist_path or not Path(blocklist_path).exists():
        return set()

    return {
        ln.strip()
        for ln in Path(blocklist_path).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }


def convert_kyujitai(text: str, csv_mapping: Dict[str, str], blocklist: Set[str]) -> str:
    """旧字→新字変換"""
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


def simple_ocr(image_path: str, language_hints: list = None) -> str:
    """
    画像から生のOCRテキストを抽出
    Google Vision APIが返す full_text_annotation.text をそのまま使用
    """
    if language_hints is None:
        language_hints = ["ja"]

    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)

    # DOCUMENT_TEXT_DETECTIONを使用
    response = client.document_text_detection(
        image=image, image_context=vision.ImageContext(language_hints=language_hints)
    )

    if response.error.message:
        raise Exception(f"{response.error.message}")

    # 生のテキストを返す（Vision APIが正しい語順で返す）
    return response.full_text_annotation.text if response.full_text_annotation else ""


def main():
    import argparse

    ap = argparse.ArgumentParser(description="シンプルで正確なOCR - Vision APIの結果をそのまま利用")
    ap.add_argument("--input-dir", required=True, help="画像フォルダ")
    ap.add_argument("--output", required=True, help="出力テキストファイル")
    ap.add_argument("--kyujitai-csv", default=None, help="旧→新のCSV（old,new）")
    ap.add_argument("--blocklist", default=None, help="変換から除外する語（1行1語）")
    ap.add_argument("--language-hints", default="ja", help="言語ヒント（カンマ区切り）")
    ap.add_argument("--page-separator", default="\n\n---\n\n", help="ページ間の区切り")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    # 画像ファイルを収集（ファイル名でソート）
    valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
    image_files = sorted(
        [p for p in input_dir.rglob("*") if p.suffix.lower() in valid_ext], key=lambda p: p.name
    )

    if not image_files:
        print(f"❌ {input_dir} に画像ファイルが見つかりません")
        sys.exit(1)

    print(f"📷 {len(image_files)}枚の画像を処理します...")

    # 言語ヒント
    lang_hints = [s.strip() for s in args.language_hints.split(",") if s.strip()]

    # 旧字→新字マップ
    csv_map = load_kyujitai_map(args.kyujitai_csv)
    blocklist = load_blocklist(args.blocklist)

    # OCR処理
    pages = []
    for i, img_path in enumerate(image_files, 1):
        print(f"  [{i}/{len(image_files)}] {img_path.name}")
        try:
            text = simple_ocr(str(img_path), lang_hints)

            # 旧字→新字変換
            if csv_map or kyujitai_to_shinjitai:
                text = convert_kyujitai(text, csv_map, blocklist)

            pages.append(text)
        except Exception as e:
            print(f"    ⚠️  エラー: {e}")
            pages.append(f"[ERROR: {img_path.name}]")

    # 出力
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_text = args.page_separator.join(pages)
    output_path.write_text(final_text, encoding="utf-8")

    total_chars = len(final_text)
    print(f"\n✅ 完了: {output_path}")
    print(f"   総文字数: {total_chars:,} 文字")


if __name__ == "__main__":
    main()
