"""OCR 結果の保存."""

import json
from pathlib import Path

from ppocr_playground.models import OcrResult


def save_ocr_result(result: OcrResult, output_path: str) -> None:
    """OCR結果をJSONファイルに保存する.

    Args:
        result (OcrResult): 保存するOCR結果.
        output_path (str): 出力JSONファイルのパス.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
