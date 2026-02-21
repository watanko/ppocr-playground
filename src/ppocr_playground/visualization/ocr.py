"""OCR 結果の可視化."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ppocr_playground.models import OcrResult, OcrTextItem

_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# スコアに応じた色分けの閾値
_HIGH_SCORE_THRESH = 0.9
_MID_SCORE_THRESH = 0.7

# 横書き用BGR色
_H_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "high": (0, 200, 0),
    "mid": (0, 200, 255),
    "low": (0, 0, 255),
}
# 縦書き用BGR色
_V_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "high": (200, 0, 200),
    "mid": (200, 100, 200),
    "low": (200, 0, 100),
}
# 横書き用RGB色（PIL描画用）
_H_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "high": (0, 200, 0),
    "mid": (255, 200, 0),
    "low": (255, 0, 0),
}
# 縦書き用RGB色（PIL描画用）
_V_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "high": (200, 0, 200),
    "mid": (200, 100, 200),
    "low": (100, 0, 200),
}


def _pick_color(
    score: float,
    palette: dict[str, tuple[int, int, int]],
) -> tuple[int, int, int]:
    """信頼度スコアに応じてパレットから色を返す.

    Args:
        score: 認識の信頼度スコア (0.0〜1.0).
        palette: high/mid/low の3色パレット.

    Returns:
        color: 選択された色タプル.
    """
    if score >= _HIGH_SCORE_THRESH:
        return palette["high"]
    if score >= _MID_SCORE_THRESH:
        return palette["mid"]
    return palette["low"]


def draw_boxes(image: np.ndarray, items: list[OcrTextItem]) -> np.ndarray:
    """元画像上にOCR検出領域のポリゴンを描画する.

    Args:
        image: 元画像 (BGR, numpy配列).
        items: OCR検出結果のリスト.

    Returns:
        result: ポリゴン描画済みの画像.
    """
    overlay = image.copy()
    for item in items:
        pts = np.array(item.polygon, dtype=np.int32)
        palette = _V_COLORS_BGR if item.angle == 1 else _H_COLORS_BGR
        color = _pick_color(item.score, palette)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
        cv2.fillPoly(overlay, [pts], color=color)

    alpha = 0.15
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_texts(image: np.ndarray, items: list[OcrTextItem]) -> np.ndarray:
    """検出テキストをポリゴンの上部にラベルとして描画する.

    Args:
        image: ポリゴン描画済みの画像 (BGR, numpy配列).
        items: OCR検出結果のリスト.

    Returns:
        result: テキストラベル描画済みの画像.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    font_size = max(12, pil_image.height // 200)
    font = ImageFont.truetype(_FONT_PATH, font_size)

    for item in items:
        palette = _V_COLORS_RGB if item.angle == 1 else _H_COLORS_RGB
        color = _pick_color(item.score, palette)
        top_left = item.polygon[0]

        if item.angle == 1:
            # 縦書き: ポリゴン右側に1文字ずつ縦に配置
            x = top_left[0] + (item.polygon[1][0] - top_left[0]) + 4
            y = top_left[1]
            for ch in item.text:
                draw.text((x, y), ch, fill=color, font=font)
                y += font_size
        else:
            x = top_left[0]
            y = top_left[1] - font_size - 2
            draw.text((x, y), item.text, fill=color, font=font)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_legend(image: np.ndarray) -> np.ndarray:
    """凡例を画像左上に描画する.

    Args:
        image: 描画対象の画像 (BGR, numpy配列).

    Returns:
        result: 凡例描画済みの画像.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font_size = max(16, pil_image.height // 150)
    font = ImageFont.truetype(_FONT_PATH, font_size)

    legends = [
        ((0, 200, 0), f"横書き 高信頼 (>= {_HIGH_SCORE_THRESH})"),
        ((255, 200, 0), f"横書き 中信頼 (>= {_MID_SCORE_THRESH})"),
        ((255, 0, 0), f"横書き 低信頼 (< {_MID_SCORE_THRESH})"),
        ((200, 0, 200), f"縦書き 高信頼 (>= {_HIGH_SCORE_THRESH})"),
        ((200, 100, 200), f"縦書き 中信頼 (>= {_MID_SCORE_THRESH})"),
        ((100, 0, 200), f"縦書き 低信頼 (< {_MID_SCORE_THRESH})"),
    ]

    x, y = 20, 20
    padding = 8
    line_height = font_size + padding

    bg_h = line_height * len(legends) + padding * 2
    bg_w = font_size * 22
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


def visualize(
    image_path: str,
    result: OcrResult,
    output_path: str,
    *,
    show_text: bool = True,
) -> None:
    """OCR結果を元画像上に可視化して保存する.

    Args:
        image_path: 元画像のファイルパス.
        result: OCR推論結果.
        output_path: 出力画像のファイルパス.
        show_text: テキストラベルを表示するかどうか.
    """
    image = cv2.imread(image_path)
    image = draw_boxes(image, result.items)
    if show_text:
        image = draw_texts(image, result.items)
    image = draw_legend(image)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)
