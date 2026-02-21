"""ボックスソート・回転クロップ."""

import cv2
import numpy as np


def sort_boxes(dt_boxes: list[np.ndarray]) -> list[np.ndarray]:
    """テキストボックスを読み順（上→下、左→右）にソートする.

    Args:
        dt_boxes (list[np.ndarray]): テキストボックスのリスト.

    Returns:
        sorted_boxes (list[np.ndarray]): ソート済みボックスリスト.
    """
    if len(dt_boxes) == 0:
        return []

    num_boxes = len(dt_boxes)
    _boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(_boxes)

    # 同じ行（Y座標差が10px以内）のボックスをX座標でソート
    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if (
                abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10
                and _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                _boxes[j], _boxes[j + 1] = _boxes[j + 1], _boxes[j]
            else:
                break

    return _boxes


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """検出ボックスから画像を切り出して回転補正する.

    Args:
        img (np.ndarray): 入力画像 (BGR, HWC).
        points (np.ndarray): 4頂点の座標 (4, 2).

    Returns:
        cropped (np.ndarray): クロップ・変換済み画像.
    """
    points = np.array(points, dtype=np.float32)

    crop_w = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
    )
    crop_h = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]),
        )
    )
    crop_w = max(crop_w, 1)
    crop_h = max(crop_h, 1)

    dst_points = np.array(
        [[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]],
        dtype=np.float32,
    )

    m = cv2.getPerspectiveTransform(points, dst_points)
    cropped = cv2.warpPerspective(
        img,
        m,
        (crop_w, crop_h),
        borderMode=cv2.BORDER_REPLICATE,
    )

    # 縦長テキスト（縦横比1.5以上）は90度回転
    if cropped.shape[0] > cropped.shape[1] * 1.5:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    return cropped
