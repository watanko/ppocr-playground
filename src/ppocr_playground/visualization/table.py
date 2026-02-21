"""テーブル認識結果の可視化."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ppocr_playground.models import TableResult

_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

_CELL_BORDER_COLOR_BGR = (200, 0, 0)
_OCR_TEXT_COLOR_RGB = (0, 120, 0)
_CELL_FILL_COLORS_RGB = [
    (255, 230, 230),
    (230, 255, 230),
    (230, 230, 255),
    (255, 255, 210),
    (255, 220, 255),
    (220, 255, 255),
]


def draw_cell_boxes(image: np.ndarray, result: TableResult) -> np.ndarray:
    """セル検出のバウンディングボックスを罫線として描画する.

    Args:
        image: 元画像 (BGR, numpy配列).
        result: テーブル認識結果.

    Returns:
        drawn: 罫線描画済みの画像.
    """
    overlay = image.copy()
    for i, cell_box in enumerate(result.cell_boxes):
        x1, y1, x2, y2 = [int(c) for c in cell_box.bbox]
        fill_color_rgb = _CELL_FILL_COLORS_RGB[i % len(_CELL_FILL_COLORS_RGB)]
        fill_color_bgr = (fill_color_rgb[2], fill_color_rgb[1], fill_color_rgb[0])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color_bgr, -1)

    alpha = 0.25
    drawn = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # 罫線（枠線）を描画
    for cell_box in result.cell_boxes:
        x1, y1, x2, y2 = [int(c) for c in cell_box.bbox]
        cv2.rectangle(drawn, (x1, y1), (x2, y2), _CELL_BORDER_COLOR_BGR, 2)

    return drawn


def draw_ocr_texts(image: np.ndarray, result: TableResult) -> np.ndarray:
    """OCR認識テキストをセル内に描画する.

    Args:
        image: 罫線描画済みの画像 (BGR, numpy配列).
        result: テーブル認識結果.

    Returns:
        drawn: テキスト描画済みの画像.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    font_size = max(12, pil_image.height // 15)
    font = ImageFont.truetype(_FONT_PATH, font_size)

    for item in result.ocr_items:
        x1, y1, _x2, _y2 = [int(c) for c in item.bbox]
        # テキストをbbox上部に描画
        draw.text(
            (x1, y1 - font_size - 2), item.text, fill=_OCR_TEXT_COLOR_RGB, font=font
        )

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_cell_indices(image: np.ndarray, result: TableResult) -> np.ndarray:
    """各セルボックスの左上にインデックス番号を描画する.

    Args:
        image: 描画対象の画像 (BGR, numpy配列).
        result: テーブル認識結果.

    Returns:
        drawn: インデックス描画済みの画像.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    font_size = max(10, pil_image.height // 20)
    font = ImageFont.truetype(_FONT_PATH, font_size)

    for i, cell_box in enumerate(result.cell_boxes):
        x1, y1 = int(cell_box.bbox[0]), int(cell_box.bbox[1])
        label = str(i)
        bbox = draw.textbbox((x1 + 2, y1 + 2), label, font=font)
        draw.rectangle(
            [bbox[0] - 1, bbox[1] - 1, bbox[2] + 3, bbox[3] + 1],
            fill=(200, 0, 0),
        )
        draw.text((x1 + 2, y1 + 2), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def visualize_table(
    image_path: str,
    result: TableResult,
    output_path: str,
) -> None:
    """テーブル認識結果を元画像上に可視化して保存する.

    Args:
        image_path: 元画像のファイルパス.
        result: テーブル認識結果.
        output_path: 出力画像のファイルパス.
    """
    image = cv2.imread(image_path)
    image = draw_cell_boxes(image, result)
    image = draw_ocr_texts(image, result)
    image = draw_cell_indices(image, result)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)
