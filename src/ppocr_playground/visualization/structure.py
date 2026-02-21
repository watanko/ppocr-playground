"""構造解析結果の可視化."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ppocr_playground.models import LayoutBox, ParsingBlock, StructureResult

_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# ラベルごとの色定義 (RGB)
_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "table": (255, 100, 100),
    "text": (100, 200, 100),
    "title": (100, 100, 255),
    "figure": (255, 165, 0),
    "number": (200, 200, 0),
    "header": (0, 200, 200),
    "footer": (200, 0, 200),
    "reference": (150, 150, 150),
    "equation": (255, 100, 255),
    "seal": (100, 255, 255),
    "chart": (255, 200, 100),
    "caption": (180, 100, 50),
    "image": (50, 180, 220),
}
_DEFAULT_COLOR: tuple[int, int, int] = (128, 128, 128)


def _get_color(label: str) -> tuple[int, int, int]:
    """ラベルに応じたRGB色を返す.

    Args:
        label: レイアウト領域のラベル.

    Returns:
        color: RGB色タプル.
    """
    return _LABEL_COLORS.get(label, _DEFAULT_COLOR)


def _rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """RGB色をBGR色に変換する.

    Args:
        color: RGB色タプル.

    Returns:
        bgr: BGR色タプル.
    """
    return (color[2], color[1], color[0])


def draw_layout_boxes(
    image: np.ndarray,
    layout_boxes: list[LayoutBox],
) -> np.ndarray:
    """レイアウト検出ボックスを描画する.

    Args:
        image: 元画像 (BGR, numpy配列).
        layout_boxes: レイアウト検出結果のリスト.

    Returns:
        result: ボックス描画済みの画像.
    """
    overlay = image.copy()
    for box in layout_boxes:
        x1, y1, x2, y2 = [int(c) for c in box.coordinate]
        color_bgr = _rgb_to_bgr(_get_color(box.label))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)

    alpha = 0.2
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # 枠線は半透明なしで描画
    for box in layout_boxes:
        x1, y1, x2, y2 = [int(c) for c in box.coordinate]
        color_bgr = _rgb_to_bgr(_get_color(box.label))
        cv2.rectangle(result, (x1, y1), (x2, y2), color_bgr, 3)

    return result


def draw_labels(
    image: np.ndarray,
    layout_boxes: list[LayoutBox],
) -> np.ndarray:
    """レイアウトボックスにラベルと信頼度スコアを描画する.

    Args:
        image: ボックス描画済みの画像 (BGR, numpy配列).
        layout_boxes: レイアウト検出結果のリスト.

    Returns:
        result: ラベル描画済みの画像.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    font_size = max(16, pil_image.height // 120)
    font = ImageFont.truetype(_FONT_PATH, font_size)

    for box in layout_boxes:
        x1, y1 = int(box.coordinate[0]), int(box.coordinate[1])
        color = _get_color(box.label)
        label_text = f"{box.label} ({box.score:.2f})"

        # ラベル背景
        bbox = draw.textbbox((x1, y1 - font_size - 6), label_text, font=font)
        draw.rectangle(
            [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
            fill=(255, 255, 255, 200),
        )
        draw.text((x1, y1 - font_size - 6), label_text, fill=color, font=font)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_parsing_blocks(
    image: np.ndarray,
    parsing_blocks: list[ParsingBlock],
) -> np.ndarray:
    """パースブロックの領域と読み順を描画する.

    Args:
        image: 元画像 (BGR, numpy配列).
        parsing_blocks: パースブロックのリスト.

    Returns:
        result: ブロック描画済みの画像.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    font_size = max(20, pil_image.height // 100)
    font = ImageFont.truetype(_FONT_PATH, font_size)

    for block in parsing_blocks:
        x1, y1, _x2, _y2 = [int(c) for c in block.block_bbox]
        color = _get_color(block.block_label)

        # 読み順の番号を左上に表示
        order_text = (
            f"#{block.block_order}"
            if block.block_order is not None
            else block.block_label
        )
        bbox = draw.textbbox((x1, y1), order_text, font=font)
        draw.rectangle(
            [bbox[0] - 2, bbox[1] - 2, bbox[2] + 6, bbox[3] + 2],
            fill=color,
        )
        draw.text((x1 + 2, y1), order_text, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_structure_legend(image: np.ndarray, labels_used: set[str]) -> np.ndarray:
    """使用されたラベルの凡例を画像左上に描画する.

    Args:
        image: 描画対象の画像 (BGR, numpy配列).
        labels_used: 画像中で使用されたラベルの集合.

    Returns:
        result: 凡例描画済みの画像.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font_size = max(16, pil_image.height // 150)
    font = ImageFont.truetype(_FONT_PATH, font_size)

    legends = [
        (color, label) for label, color in _LABEL_COLORS.items() if label in labels_used
    ]

    x, y = 20, 20
    padding = 8
    line_height = font_size + padding

    bg_h = line_height * len(legends) + padding * 2
    bg_w = font_size * 14
    draw.rectangle([x, y, x + bg_w, y + bg_h], fill=(255, 255, 255, 220))

    for color, label in legends:
        cy = y + padding
        draw.rectangle(
            [x + padding, cy, x + padding + font_size, cy + font_size], fill=color
        )
        draw.text(
            (x + padding + font_size + padding, cy), label, fill=(0, 0, 0), font=font
        )
        y += line_height

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def visualize_structure(
    image_path: str,
    result: StructureResult,
    output_path: str,
) -> None:
    """構造解析結果を元画像上に可視化して保存する.

    Args:
        image_path: 元画像のファイルパス.
        result: 構造解析結果.
        output_path: 出力画像のファイルパス.
    """
    image = cv2.imread(image_path)
    image = draw_layout_boxes(image, result.layout_boxes)
    image = draw_labels(image, result.layout_boxes)
    image = draw_parsing_blocks(image, result.parsing_blocks)

    labels_used = {box.label for box in result.layout_boxes}
    image = draw_structure_legend(image, labels_used)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)
