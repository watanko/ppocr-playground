"""全 Pydantic モデル定義.

OCR / 構造解析 / テーブル認識の共通データモデルを集約する。
"""

from pydantic import BaseModel

# --- OCR ---


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


# --- 構造解析 ---


class LayoutBox(BaseModel):
    """レイアウト検出で検出された個別の領域.

    Attributes:
        label: 領域のラベル (table, text, figure, title, number 等).
        score: 検出の信頼度スコア (0.0〜1.0).
        coordinate: バウンディングボックス [x1, y1, x2, y2].
    """

    label: str
    score: float
    coordinate: list[float]


class ParsingBlock(BaseModel):
    """パース済みのレイアウトブロック.

    Attributes:
        block_id: ブロックID.
        block_order: ブロックの読み順.
        block_label: ブロックのラベル (table, text, title, number 等).
        block_bbox: バウンディングボックス [x1, y1, x2, y2].
        block_content: 抽出されたコンテンツ (テキスト, HTMLテーブル等).
    """

    block_id: int | None = None
    block_order: int | None = None
    block_label: str
    block_bbox: list[float]
    block_content: str


class StructureResult(BaseModel):
    """1画像分の文書構造解析結果.

    Attributes:
        input_path: 入力画像のパス.
        width: 画像幅.
        height: 画像高さ.
        layout_boxes: レイアウト検出結果のリスト.
        parsing_blocks: パース済みブロックのリスト.
    """

    input_path: str
    width: int
    height: int
    layout_boxes: list[LayoutBox]
    parsing_blocks: list[ParsingBlock]


# --- テーブル ---


class TableCell(BaseModel):
    """テーブルの1セル.

    Attributes:
        row: 行インデックス (0始まり).
        col: 列インデックス (0始まり).
        rowspan: 行方向の結合数.
        colspan: 列方向の結合数.
        text: セル内のテキスト.
    """

    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    text: str = ""


class CellBox(BaseModel):
    """セル検出のバウンディングボックス.

    Attributes:
        bbox: セル領域 [x1, y1, x2, y2].
    """

    bbox: list[float]


class OcrItem(BaseModel):
    """テーブル内OCR検出結果.

    Attributes:
        text: 認識テキスト.
        score: 信頼度スコア.
        bbox: バウンディングボックス [x1, y1, x2, y2].
    """

    text: str
    score: float
    bbox: list[float]


class TableResult(BaseModel):
    """テーブル認識結果.

    Attributes:
        input_path: 入力画像のパス.
        html: 認識されたHTMLテーブル.
        cells: HTMLパース結果のセルリスト.
        grid: 2次元グリッド表現 (行×列).
        cell_boxes: セル検出のバウンディングボックスリスト.
        ocr_items: テーブル内OCR検出結果のリスト.
    """

    input_path: str
    html: str
    cells: list[TableCell]
    grid: list[list[str]]
    cell_boxes: list[CellBox] = []
    ocr_items: list[OcrItem] = []
