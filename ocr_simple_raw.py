#!/usr/bin/env python3
"""
シンプルなOCR - Google Vision APIの生の結果を確認
複雑な処理を一切せず、APIが返すテキストをそのまま出力
"""

import sys
from pathlib import Path
from google.cloud import vision


def simple_ocr(image_path: str) -> str:
    """画像から生のOCRテキストを抽出"""
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)

    # DOCUMENT_TEXT_DETECTIONを使用
    response = client.document_text_detection(
        image=image, image_context=vision.ImageContext(language_hints=["ja"])
    )

    if response.error.message:
        raise Exception(f"{response.error.message}")

    # 生のテキストを返す
    return response.full_text_annotation.text if response.full_text_annotation else ""


def main():
    if len(sys.argv) < 2:
        print("使用法: python ocr_simple_raw.py <画像ディレクトリ>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("ocr_raw_output.txt")

    # 画像ファイルを収集（ファイル名でソート）
    valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    image_files = sorted(
        [p for p in input_dir.rglob("*") if p.suffix.lower() in valid_ext],
        key=lambda p: p.name,  # ファイル名でソート
    )

    if not image_files:
        print(f"❌ {input_dir} に画像ファイルが見つかりません")
        sys.exit(1)

    print(f"📷 {len(image_files)}枚の画像を処理します...")

    all_text = []
    for i, img_path in enumerate(image_files, 1):
        print(f"  [{i}/{len(image_files)}] {img_path.name}")
        try:
            text = simple_ocr(str(img_path))
            all_text.append(f"=== {img_path.name} ===\n{text}\n")
        except Exception as e:
            print(f"    ⚠️  エラー: {e}")
            all_text.append(f"=== {img_path.name} ===\n[ERROR: {e}]\n")

    # 出力
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(all_text), encoding="utf-8")

    total_chars = sum(len(t) for t in all_text)
    print(f"\n✅ 完了: {output_file}")
    print(f"   総文字数: {total_chars:,} 文字")


if __name__ == "__main__":
    main()
