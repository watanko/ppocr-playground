"""OCR / 構造解析 / テーブル認識結果の保存."""

import json
from pathlib import Path

import numpy as np

from ppocr_playground.models import OcrResult, StructureResult, TableResult


class NumpyEncoder(json.JSONEncoder):
    """numpy型をJSON直列化可能にするエンコーダ."""

    def default(self, o: object) -> object:
        """numpy型をPython標準型に変換する.

        Args:
            o: 直列化対象のオブジェクト.

        Returns:
            converted: Python標準型に変換されたオブジェクト.
        """
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_ocr_result(result: OcrResult, output_path: str) -> None:
    """OCR結果をJSONファイルに保存する.

    Args:
        result: 保存するOCR結果.
        output_path: 出力JSONファイルのパス.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)


def save_structure_result(result: StructureResult, output_path: str) -> None:
    """構造解析結果をJSONファイルに保存する.

    Args:
        result: 保存する構造解析結果.
        output_path: 出力JSONファイルのパス.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            result.model_dump(), f, ensure_ascii=False, indent=2, cls=NumpyEncoder
        )


def save_table_result(result: TableResult, output_path: str) -> None:
    """テーブル認識結果をJSONファイルに保存する.

    Args:
        result: 保存するテーブル認識結果.
        output_path: 出力JSONファイルのパス.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
