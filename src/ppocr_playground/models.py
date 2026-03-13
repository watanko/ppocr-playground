"""OCR Pydantic モデル定義."""

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
