"""OCR エンジンの抽象基底クラス."""

from abc import ABC, abstractmethod

from ppocr_playground.models import OcrResult


class OcrEngine(ABC):
    """OCR エンジンの抽象基底クラス.

    Strategy パターンで PaddleOCR / ONNX を切り替える。
    """

    @abstractmethod
    def run(self, image_path: str) -> OcrResult:
        """1枚の画像を OCR する.

        Args:
            image_path (str): 入力画像のファイルパス.

        Returns:
            result (OcrResult): OCR推論結果.
        """
        ...
