"""PaddleOCR ベースの OCR エンジン."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ppocr_playground.engine.base import OcrEngine
from ppocr_playground.models import OcrResult, OcrTextItem

if TYPE_CHECKING:
    from paddleocr import PaddleOCR


class PaddleOcrEngine(OcrEngine):
    """PaddleOCR を使用する OCR エンジン.

    Attributes:
        lang: OCR の言語設定.
        _engine: PaddleOCR インスタンス.
    """

    def __init__(self, lang: str = "japan") -> None:
        """エンジンを初期化する.

        Args:
            lang (str): OCR の言語設定.
        """
        self.lang = lang
        self._engine: PaddleOCR | None = None

    def _get_engine(self) -> PaddleOCR:
        """PaddleOCR インスタンスを遅延初期化して返す.

        Returns:
            engine (PaddleOCR): PaddleOCR インスタンス.
        """
        if self._engine is None:
            from paddleocr import PaddleOCR

            self._engine = PaddleOCR(
                lang=self.lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
            )
        return self._engine

    def run(self, image_path: str) -> OcrResult:
        """PaddleOCR で画像を推論し、結果を返す.

        Args:
            image_path (str): 入力画像のファイルパス.

        Returns:
            result (OcrResult): OCR推論結果.
        """
        ocr = self._get_engine()
        predictions = ocr.predict(image_path)

        items: list[OcrTextItem] = []
        for prediction in predictions:
            res = prediction.json["res"]
            texts: list[str] = res["rec_texts"]
            scores: list[float] = res["rec_scores"]
            polys: list[list[list[int]]] = res["rec_polys"]
            angles: list[int] = res.get("textline_orientation_angles", [0] * len(texts))

            for text, score, poly, angle in zip(texts, scores, polys, angles):
                items.append(
                    OcrTextItem(text=text, score=score, polygon=poly, angle=angle)
                )

        return OcrResult(
            input_path=image_path,
            text_count=len(items),
            items=items,
        )
