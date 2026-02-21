"""DETR 前処理・後処理 + テーブル分類前処理."""

import cv2
import numpy as np
import onnxruntime as ort

from ppocr_playground.onnx_ops import TABLE_CLS_CROP_SIZE


def preprocess_detr(
    image: np.ndarray,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DETR系モデル（レイアウト検出・セル検出）の前処理を行う.

    800x800 or 640x640 にリサイズし、/255.0 で正規化、CHW形式に変換する。
    im_shape と scale_factor も返す。

    Args:
        image (np.ndarray): 入力画像 (BGR, HWC).
        target_size (tuple[int, int]): ターゲットサイズ (w, h).

    Returns:
        image_tensor (np.ndarray): モデル入力 (1, 3, H, W).
        im_shape (np.ndarray): 画像サイズ [[h, w]] (1, 2).
        scale_factor (np.ndarray): スケール係数 [[h_scale, w_scale]] (1, 2).
    """
    ori_h, ori_w = image.shape[:2]
    tw, th = target_size

    resized = cv2.resize(image, (tw, th))
    img_float = resized.astype(np.float32) / 255.0
    img_chw = img_float.transpose(2, 0, 1)
    image_tensor = np.expand_dims(img_chw, axis=0)

    im_shape = np.array([[float(th), float(tw)]], dtype=np.float32)
    scale_factor = np.array([[th / ori_h, tw / ori_w]], dtype=np.float32)

    return image_tensor, im_shape, scale_factor


def run_detr(
    session: ort.InferenceSession,
    image: np.ndarray,
    target_size: tuple[int, int],
    labels: list[str],
    threshold: float,
) -> list[dict[str, object]]:
    """DETR系モデルで推論し、検出結果を返す.

    Args:
        session (ort.InferenceSession): 推論セッション.
        image (np.ndarray): 入力画像 (BGR, HWC).
        target_size (tuple[int, int]): リサイズ先 (w, h).
        labels (list[str]): ラベルリスト.
        threshold (float): スコア閾値.

    Returns:
        boxes (list[dict]): 検出結果。各要素は label, score, coordinate キーを持つ。
    """
    ori_h, ori_w = image.shape[:2]
    img_tensor, im_shape, scale_factor = preprocess_detr(image, target_size)

    input_names = [inp.name for inp in session.get_inputs()]
    feed: dict[str, np.ndarray] = {}
    for name in input_names:
        if "im_shape" in name:
            feed[name] = im_shape
        elif "scale_factor" in name:
            feed[name] = scale_factor
        else:
            feed[name] = img_tensor

    output = session.run(None, feed)[0]

    boxes: list[dict[str, object]] = []
    for row in output:
        cls_id, score = int(row[0]), float(row[1])
        if cls_id < 0 or score < threshold:
            continue
        x1 = int(max(0, round(row[2])))
        y1 = int(max(0, round(row[3])))
        x2 = int(min(ori_w, round(row[4])))
        y2 = int(min(ori_h, round(row[5])))
        if x2 <= x1 or y2 <= y1:
            continue
        label = labels[cls_id] if cls_id < len(labels) else f"class_{cls_id}"
        boxes.append({"label": label, "score": score, "coordinate": [x1, y1, x2, y2]})

    return boxes


def preprocess_table_cls(image: np.ndarray) -> np.ndarray:
    """テーブル分類モデルの前処理を行う.

    resize_short=256 → center_crop=224 → ImageNet正規化 → CHW。

    Args:
        image (np.ndarray): テーブル領域画像 (BGR, HWC).

    Returns:
        input_tensor (np.ndarray): モデル入力 (1, 3, 224, 224).
    """
    h, w = image.shape[:2]
    scale = 256.0 / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    cy, cx = new_h // 2, new_w // 2
    half = TABLE_CLS_CROP_SIZE // 2
    cropped = resized[cy - half : cy + half, cx - half : cx + half]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = cropped.astype(np.float32) / 255.0
    normalized = (normalized - mean) / std

    return np.expand_dims(normalized.transpose(2, 0, 1), axis=0).astype(np.float32)
