"""テキスト検出の前処理・後処理."""

import cv2
import numpy as np
import onnxruntime as ort

from ppocr_playground.onnx_ops import (
    DET_DB_BOX_THRESH,
    DET_DB_THRESH,
    DET_DB_UNCLIP_RATIO,
    DET_LIMIT_SIDE_LEN,
    DET_LIMIT_TYPE,
    DET_MAX_SIDE_LIMIT,
)


def preprocess_det(
    image: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, float, float]]:
    """テキスト検出モデルの前処理を行う.

    Args:
        image (np.ndarray): 入力画像 (BGR, HWC).

    Returns:
        input_tensor (np.ndarray): モデル入力テンソル (1, 3, H, W).
        shape_info (tuple[int, int, float, float]): (src_h, src_w, ratio_h, ratio_w).
    """
    h, w = image.shape[:2]

    # 小さすぎる画像のパディング
    if h + w < 64:
        pad_h = max(32, h)
        pad_w = max(32, w)
        padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        padded[:h, :w, :] = image
        image = padded
        h, w = image.shape[:2]

    if DET_LIMIT_TYPE == "max":
        ratio = (
            float(DET_LIMIT_SIDE_LEN) / max(h, w)
            if max(h, w) > DET_LIMIT_SIDE_LEN
            else 1.0
        )
    elif DET_LIMIT_TYPE == "min":
        ratio = (
            float(DET_LIMIT_SIDE_LEN) / min(h, w)
            if min(h, w) < DET_LIMIT_SIDE_LEN
            else 1.0
        )
    else:
        ratio = float(DET_LIMIT_SIDE_LEN) / max(h, w)

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    # max_side_limit 超過時はさらに縮小
    if max(resize_h, resize_w) > DET_MAX_SIDE_LIMIT:
        limit_ratio = float(DET_MAX_SIDE_LIMIT) / max(resize_h, resize_w)
        resize_h = int(resize_h * limit_ratio)
        resize_w = int(resize_w * limit_ratio)

    # 32の倍数に調整
    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)

    resized = cv2.resize(image, (resize_w, resize_h))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    # ImageNet正規化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - mean) / std

    # HWC -> NCHW
    transposed = normalized.transpose(2, 0, 1)
    batched = np.expand_dims(transposed, axis=0).astype(np.float32)

    return batched, (h, w, ratio_h, ratio_w)


def box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
    """ボックス内のスコアを高速に計算する.

    Args:
        bitmap (np.ndarray): 確率マップ (H, W).
        box (np.ndarray): ボックス座標 (N, 2).

    Returns:
        score (float): ボックス内の平均確率.
    """
    h, w = bitmap.shape[:2]
    _box = box.copy()
    xmin = np.clip(np.floor(_box[:, 0].min()).astype("int32"), 0, w - 1)
    xmax = np.clip(np.ceil(_box[:, 0].max()).astype("int32"), 0, w - 1)
    ymin = np.clip(np.floor(_box[:, 1].min()).astype("int32"), 0, h - 1)
    ymax = np.clip(np.ceil(_box[:, 1].max()).astype("int32"), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    _box[:, 0] = _box[:, 0] - xmin
    _box[:, 1] = _box[:, 1] - ymin
    cv2.fillPoly(mask, _box.reshape(1, -1, 2).astype("int32"), 1)
    return float(cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0])


def unclip(box: np.ndarray, unclip_ratio: float) -> np.ndarray:
    """cv2.minAreaRect ベースでボックスを拡張する.

    Args:
        box (np.ndarray): 入力ボックス (N, 2).
        unclip_ratio (float): 拡張比率.

    Returns:
        expanded (np.ndarray): 拡張後のボックス (1, N, 2).
    """
    box = np.array(box, dtype=np.float32)
    if len(box) < 3:
        return box.reshape(1, -1, 2)

    rect = cv2.minAreaRect(box.reshape(-1, 1, 2))
    center, (width, height), angle = rect

    if width < 1e-6 or height < 1e-6:
        return box.reshape(1, -1, 2)

    area = width * height
    perimeter = 2 * (width + height)
    distance = area * unclip_ratio / perimeter * 1.10

    new_width = width + 2 * distance
    new_height = height + 2 * distance
    new_rect = (center, (new_width, new_height), angle)
    expanded = cv2.boxPoints(new_rect)

    return expanded.astype(np.float32).reshape(1, -1, 2)


def get_mini_boxes(contour: np.ndarray) -> tuple[np.ndarray, float]:
    """輪郭の最小外接矩形を取得してソートする.

    Args:
        contour (np.ndarray): 輪郭点列.

    Returns:
        box (np.ndarray): ソート済みの4頂点 (4, 2).
        sside (float): 短辺の長さ.
    """
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0
    if points[3][1] > points[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return np.array(box), min(bounding_box[1])


def postprocess_det(
    pred: np.ndarray,
    shape_info: tuple[int, int, float, float],
) -> tuple[list[np.ndarray], list[float]]:
    """DB後処理: 確率マップからテキストボックスを抽出する.

    Args:
        pred (np.ndarray): モデル出力 (1, 1, H, W).
        shape_info (tuple[int, int, float, float]): (ori_h, ori_w, ratio_h, ratio_w).

    Returns:
        boxes (list[np.ndarray]): テキストボックスのリスト.
        scores (list[float]): 各ボックスのスコア.
    """
    ori_h, ori_w, _ratio_h, _ratio_w = shape_info
    pred_map = pred[0, 0]
    height, width = pred_map.shape

    # 二値化
    mask = ((pred_map > DET_DB_THRESH) * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = min(len(contours), 1000)

    boxes: list[np.ndarray] = []
    scores: list[float] = []

    for index in range(num_contours):
        contour = contours[index]

        points, sside = get_mini_boxes(contour)
        if sside < 3:
            continue

        score = box_score_fast(pred_map, points.reshape(-1, 2))
        if score < DET_DB_BOX_THRESH:
            continue

        expanded = unclip(points, DET_DB_UNCLIP_RATIO)
        if expanded is None or len(expanded) == 0:
            continue
        expanded = expanded.reshape(-1, 1, 2)

        box, sside = get_mini_boxes(expanded)
        if sside < 5:
            continue
        box = np.array(box)

        # 元のサイズにスケーリング
        box[:, 0] = np.clip(np.round(box[:, 0] / width * ori_w), 0, ori_w)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * ori_h), 0, ori_h)

        boxes.append(box.astype("int32"))
        scores.append(score)

    return boxes, scores


def detect_single(
    image: np.ndarray,
    det_session: ort.InferenceSession,
) -> tuple[list[np.ndarray], list[float]]:
    """1枚の画像でテキスト検出を行う（SAHIなし）.

    Args:
        image (np.ndarray): 入力画像 (BGR, HWC).
        det_session (ort.InferenceSession): 検出モデルセッション.

    Returns:
        boxes (list[np.ndarray]): テキストボックスのリスト.
        scores (list[float]): 各ボックスのスコア.
    """
    det_input_name = det_session.get_inputs()[0].name
    det_input, shape_info = preprocess_det(image)
    det_output = det_session.run(None, {det_input_name: det_input})[0]
    return postprocess_det(det_output, shape_info)
