import os
from pathlib import Path
from typing import Iterable, List

# 認証ファイルパス（必要に応じて調整）
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Volumes/SSD-SCTU3A/ラジオ用/charged-camera.json"

from google.cloud import vision


def chunk_list(lst: List, chunk_size: int) -> Iterable[List]:
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def ocr_process_local_images(
    input_dir: str,
    output_dir: str | None = None,
    batch_size: int = 16,
    use_document_detection: bool = True,
    language_hints: tuple[str, ...] = ("ja", "en"),
) -> None:
    """
    ローカルの画像をまとめて OCR。結果は stdout に出力しつつ、各画像ごとに .txt を保存します。
    - input_dir: 画像が入っているディレクトリ（再帰的に探索）
    - output_dir: 結果の .txt を保存する先（省略時は input_dir/_ocr）
    - batch_size: Vision API の batch_annotate_images の一括枚数（最大16推奨）
    - use_document_detection: True=DOCUMENT_TEXT_DETECTION（段組/多行向き）, False=TEXT_DETECTION（単語・短文向き）
    - language_hints: 日本語中心なら ("ja","en") が無難
    """
    feature_type = (
        vision.Feature.Type.DOCUMENT_TEXT_DETECTION
        if use_document_detection
        else vision.Feature.Type.TEXT_DETECTION
    )
    image_context = vision.ImageContext(language_hints=list(language_hints))

    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    input_root = Path(input_dir)
    output_root = Path(output_dir) if output_dir else (input_root / "_ocr")
    output_root.mkdir(parents=True, exist_ok=True)

    # 対象ファイルを収集（再帰）
    paths = sorted(
        p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts
    )
    if not paths:
        print("対象の画像ファイルが見つかりませんでした。")
        return

    print(f"Found {len(paths)} image(s) under: {input_root}")

    # Vision クライアント
    vision_client = vision.ImageAnnotatorClient()

    # AnnotateImageRequest を構築
    requests, names = [], []
    for p in paths:
        with p.open("rb") as f:
            content = f.read()
        image = vision.Image(content=content)
        req = vision.AnnotateImageRequest(
            image=image,
            features=[vision.Feature(type_=feature_type)],
            image_context=image_context,
        )
        requests.append(req)
        names.append(p)

    # バッチ実行
    all_responses = []
    for batch_idx, batch in enumerate(chunk_list(requests, batch_size), start=1):
        print(f"Processing batch {batch_idx} ({len(batch)} images)...")
        resp = vision_client.batch_annotate_images(requests=batch)
        all_responses.extend(resp.responses)

    # 出力（stdout と .txt 保存）
    for path, res in zip(names, all_responses):
        rel = path.relative_to(input_root)
        print("-" * 60)
        print(f"File: {rel}")

        if res.error.message:
            print(f"Error: {res.error.message}")
            # エラーファイルも空のTXTを出す/出さないはお好みで
            continue

        # DOCUMENT_TEXT_DETECTION のときは full_text_annotation が最も信頼できる
        text = ""
        if res.full_text_annotation and res.full_text_annotation.text:
            text = res.full_text_annotation.text
        elif res.text_annotations:
            text = res.text_annotations[0].description
        else:
            print("No text detected.")
            continue

        print(text.strip())

        out_txt = output_root / rel.with_suffix(".txt")
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text(text, encoding="utf-8")

    print("\n✅ 完了：OCR結果を保存しました →", output_root)


if __name__ == "__main__":
    # 例：ローカルの画像フォルダを直接指定
    ocr_process_local_images(
        input_dir="/Volumes/SSD-SCTU3A/ラジオ用/ocr_targets/不義士右門",  # ここをあなたの画像フォルダに
        output_dir=None,  # 省略可（自動で <input>/_ocr に保存）
        batch_size=16,  # Vision API の上限16が安全
        use_document_detection=True,  # 段組PDFや誌面写真のような「文書」にはこちらが高精度
        language_hints=("ja", "en"),  # 日本語中心。英数字混在ならこのままでOK
    )
