"""PaddleOCR推論結果をJSONファイルに保存するスクリプト."""

import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel


class OcrTextItem(BaseModel):
    """OCRで検出された個別のテキスト領域.

    Attributes:
        text: 認識されたテキスト文字列.
        score: 認識の信頼度スコア (0.0〜1.0).
        polygon: テキスト領域の4頂点座標 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
        angle: テキスト行の向き (0=横書き, 1=縦書き).
    """

    text: str
    score: float
    polygon: list[list[int]]
    angle: int = 0


class OcrResult(BaseModel):
    """1画像分のOCR推論結果.

    Attributes:
        input_path: 入力画像のパス.
        text_count: 検出されたテキスト領域の数.
        items: 検出されたテキスト領域のリスト.
    """

    input_path: str
    text_count: int
    items: list[OcrTextItem]


def run_ocr(image_path: str, lang: str = "japan") -> OcrResult:
    """PaddleOCRで画像を推論し、結果を返す.

    Args:
        image_path: 入力画像のファイルパス.
        lang: OCRの言語設定.

    Returns:
        result: OCR推論結果.
    """
    # PaddleOCRのインポートは重いため関数内で遅延インポート
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )
    predictions = ocr.predict(image_path)

    items: list[OcrTextItem] = []
    for prediction in predictions:
        res = prediction.json["res"]
        texts: list[str] = res["rec_texts"]
        scores: list[float] = res["rec_scores"]
        polys: list[list[list[int]]] = res["rec_polys"]
        angles: list[int] = res.get("textline_orientation_angles", [0] * len(texts))

        for text, score, poly, angle in zip(texts, scores, polys, angles):
            items.append(OcrTextItem(text=text, score=score, polygon=poly, angle=angle))

    return OcrResult(
        input_path=image_path,
        text_count=len(items),
        items=items,
    )


def save_result(result: OcrResult, output_path: str) -> None:
    """OCR結果をJSONファイルに保存する.

    Args:
        result: 保存するOCR結果.
        output_path: 出力JSONファイルのパス.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)


def main() -> None:
    """メインエントリポイント. コマンドライン引数から画像パスを受け取り推論・保存する."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        stem = Path(image_path).stem
        output_path = str(Path(image_path).parent / f"{stem}_ocr.json")

    # PaddlePaddle 3.3.0のoneDNNバグ回避
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    print(f"入力: {image_path}")
    result = run_ocr(image_path)
    save_result(result, output_path)
    print(f"検出テキスト数: {result.text_count}")
    print(f"出力: {output_path}")


if __name__ == "__main__":
    main()
