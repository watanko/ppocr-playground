"""テキスト行方向分類の前処理・後処理."""

import cv2
import numpy as np

from ppocr_playground.onnx_ops.session import CLS_MEAN, CLS_RESIZE, CLS_STD


def preprocess_cls(images: list[np.ndarray]) -> np.ndarray:
    """テキスト行方向分類モデルの前処理を行う.

    160x80にリサイズし、ImageNet正規化。

    Args:
        images (list[np.ndarray]): クロップ画像のリスト (BGR, HWC).

    Returns:
        batch (np.ndarray): バッチテンソル (N, 3, 80, 160).
    """
    processed: list[np.ndarray] = []
    for img in images:
        resized = cv2.resize(img, CLS_RESIZE)
        resized = resized.astype(np.float32) / 255.0
        resized = (resized - CLS_MEAN) / CLS_STD
        resized = resized.transpose(2, 0, 1)
        processed.append(resized)
    return np.array(processed, dtype=np.float32)


def postprocess_cls(pred: np.ndarray) -> list[int]:
    """方向分類の後処理を行う.

    Args:
        pred (np.ndarray): モデル出力 (N, 2).

    Returns:
        labels (list[int]): 0=0_degree, 1=180_degree.
    """
    return pred.argmax(axis=-1).tolist()
