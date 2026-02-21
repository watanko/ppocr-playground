"""SAHI タイル検出 + NMS."""

import numpy as np
import onnxruntime as ort

from ppocr_playground.onnx_ops import SAHI_NMS_THRESH, SAHI_OVERLAP, SAHI_TILE_SIZE
from ppocr_playground.onnx_ops.detection import postprocess_det, preprocess_det


def quad_to_aabb(box: np.ndarray) -> tuple[int, int, int, int]:
    """4点ポリゴンを軸並行バウンディングボックスに変換する.

    Args:
        box (np.ndarray): 4頂点 (4, 2).

    Returns:
        aabb (tuple[int, int, int, int]): (x1, y1, x2, y2).
    """
    x1 = int(box[:, 0].min())
    y1 = int(box[:, 1].min())
    x2 = int(box[:, 0].max())
    y2 = int(box[:, 1].max())
    return x1, y1, x2, y2


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """2つの4点ポリゴンのAABBベースIoUを計算する.

    Args:
        box_a (np.ndarray): 4頂点 (4, 2).
        box_b (np.ndarray): 4頂点 (4, 2).

    Returns:
        iou (float): IoU値.
    """
    ax1, ay1, ax2, ay2 = quad_to_aabb(box_a)
    bx1, by1, bx2, by2 = quad_to_aabb(box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    if union < 1e-6:
        return 0.0
    return inter / union


def nms_quad_boxes(
    boxes: list[np.ndarray],
    scores: list[float],
    iou_thresh: float = SAHI_NMS_THRESH,
) -> tuple[list[np.ndarray], list[float]]:
    """AABBベースのNMSで重複テキストボックスを除去する.

    Args:
        boxes (list[np.ndarray]): テキストボックスのリスト (各 (4, 2)).
        scores (list[float]): 各ボックスのスコア.
        iou_thresh (float): IoU閾値. これ以上重なるボックスを抑制.

    Returns:
        kept_boxes (list[np.ndarray]): NMS後のボックス.
        kept_scores (list[float]): NMS後のスコア.
    """
    if len(boxes) == 0:
        return [], []

    # スコア降順でソート
    order = np.argsort(scores)[::-1].tolist()

    keep: list[int] = []
    suppressed: set[int] = set()

    for idx in order:
        if idx in suppressed:
            continue
        keep.append(idx)
        for jdx in order:
            if jdx in suppressed or jdx == idx:
                continue
            if compute_iou(boxes[idx], boxes[jdx]) > iou_thresh:
                suppressed.add(jdx)

    return [boxes[i] for i in keep], [scores[i] for i in keep]


def generate_tiles(
    h: int,
    w: int,
    tile_size: int = SAHI_TILE_SIZE,
    overlap: int = SAHI_OVERLAP,
) -> list[tuple[int, int, int, int]]:
    """画像をオーバーラップ付きタイルに分割する座標を生成する.

    Args:
        h (int): 画像高さ.
        w (int): 画像幅.
        tile_size (int): タイルサイズ (正方形).
        overlap (int): タイル間のオーバーラップ.

    Returns:
        tiles (list[tuple[int, int, int, int]]): (y1, x1, y2, x2) のリスト.
    """
    stride = tile_size - overlap
    tiles: list[tuple[int, int, int, int]] = []

    y = 0
    while y < h:
        y2 = min(y + tile_size, h)
        x = 0
        while x < w:
            x2 = min(x + tile_size, w)
            tiles.append((y, x, y2, x2))
            if x2 >= w:
                break
            x += stride
        if y2 >= h:
            break
        y += stride

    return tiles


def detect_with_sahi(
    image: np.ndarray,
    det_session: ort.InferenceSession,
) -> tuple[list[np.ndarray], list[float]]:
    """SAHI方式のパッチ推論でテキスト検出を行う.

    画像をオーバーラップ付きタイルに分割し、各タイルで検出を実行する。
    重複除去は「センター・イン・コア」方式: 各タイルの中央領域（コア）に
    ボックスの重心が含まれる場合のみ採用する。

    Args:
        image (np.ndarray): 入力画像 (BGR, HWC).
        det_session (ort.InferenceSession): 検出モデルセッション.

    Returns:
        boxes (list[np.ndarray]): テキストボックスのリスト.
        scores (list[float]): 各ボックスのスコア.
    """
    h, w = image.shape[:2]
    det_input_name = det_session.get_inputs()[0].name

    tiles = generate_tiles(h, w)
    all_boxes: list[np.ndarray] = []
    all_scores: list[float] = []

    margin = SAHI_OVERLAP / 2

    for y1, x1, y2, x2 in tiles:
        # タイルのコア領域（元画像座標）
        # 画像端に接するタイルはマージンを取らない
        core_x1 = x1 + (margin if x1 > 0 else 0)
        core_y1 = y1 + (margin if y1 > 0 else 0)
        core_x2 = x2 - (margin if x2 < w else 0)
        core_y2 = y2 - (margin if y2 < h else 0)

        tile = image[y1:y2, x1:x2]
        det_input, shape_info = preprocess_det(tile)
        det_output = det_session.run(None, {det_input_name: det_input})[0]
        tile_boxes, tile_scores = postprocess_det(det_output, shape_info)

        for box, score in zip(tile_boxes, tile_scores):
            # タイル座標 → 元画像座標
            box[:, 0] += x1
            box[:, 1] += y1

            # ボックス重心がコア領域内にあるもののみ採用
            cx = float(box[:, 0].mean())
            cy = float(box[:, 1].mean())
            if core_x1 <= cx <= core_x2 and core_y1 <= cy <= core_y2:
                all_boxes.append(box)
                all_scores.append(score)

    # コアフィルタで大半の重複は除去されるが、端のケースをNMSで処理
    return nms_quad_boxes(all_boxes, all_scores)
