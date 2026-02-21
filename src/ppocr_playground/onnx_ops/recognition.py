"""テキスト認識の前処理・後処理."""

import math

import cv2
import numpy as np

from ppocr_playground.onnx_ops import REC_IMAGE_SHAPE


def resize_norm_img_rec(
    img: np.ndarray,
    max_wh_ratio: float | None = None,
) -> np.ndarray:
    """認識用に1枚の画像をリサイズして正規化する.

    Args:
        img (np.ndarray): 入力画像 (BGR, HWC).
        max_wh_ratio (float | None): 最大幅/高さ比率. Noneの場合はデフォルト.

    Returns:
        padded (np.ndarray): 正規化・パディング済みテンソル (3, H, W).
    """
    img_c, img_h, img_w = REC_IMAGE_SHAPE

    if max_wh_ratio is not None:
        img_w = int(img_h * max_wh_ratio)

    h, w = img.shape[:2]
    ratio = w / float(h)
    resized_w = (
        img_w if math.ceil(img_h * ratio) > img_w else int(math.ceil(img_h * ratio))
    )

    resized_image = cv2.resize(img, (resized_w, img_h))
    resized_image = resized_image.astype("float32")

    # 正規化: (x / 255 - 0.5) / 0.5
    resized_image = resized_image.transpose((2, 0, 1)) / 255.0
    resized_image -= 0.5
    resized_image /= 0.5

    padding_im = np.zeros((img_c, img_h, img_w), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image

    return padding_im


def preprocess_rec_batch(images: list[np.ndarray]) -> np.ndarray:
    """認識モデル用のバッチ前処理を行う.

    Args:
        images (list[np.ndarray]): クロップ画像のリスト (BGR, HWC).

    Returns:
        batch (np.ndarray): バッチテンソル (N, 3, 48, W).
    """
    _, img_h, img_w = REC_IMAGE_SHAPE

    width_list = [img.shape[1] / float(img.shape[0]) for img in images]
    max_wh_ratio = img_w / img_h
    for w_ratio in width_list:
        max_wh_ratio = max(max_wh_ratio, w_ratio)

    norm_img_batch = []
    for img in images:
        norm_img = resize_norm_img_rec(img, max_wh_ratio)
        norm_img_batch.append(norm_img[np.newaxis, :])

    return np.concatenate(norm_img_batch).astype(np.float32)


def build_char_mask(
    char_dict: list[str],
    allowed_chars: set[str],
) -> np.ndarray:
    """許可文字のみを残すマスクベクトルを構築する.

    Args:
        char_dict (list[str]): "blank" + 辞書文字のリスト.
        allowed_chars (set[str]): 許可する文字の集合.

    Returns:
        mask (np.ndarray): shape (C,). 許可文字=0, 非許可文字=-inf.
    """
    mask = np.full(len(char_dict), -np.inf, dtype=np.float32)
    mask[0] = 0.0  # blank は常に許可
    for i, ch in enumerate(char_dict):
        if ch in allowed_chars:
            mask[i] = 0.0
    return mask


def postprocess_rec(
    pred: np.ndarray,
    char_dict: list[str],
    char_mask: np.ndarray | None = None,
) -> list[tuple[str, float]]:
    """CTC後処理: 予測確率から文字列とスコアを得る.

    Args:
        pred (np.ndarray): モデル出力 (N, T, C).
        char_dict (list[str]): "blank" + 辞書文字のリスト.
        char_mask (np.ndarray | None): 文字マスク. build_char_mask() で生成.
            None の場合は全文字を使用する.

    Returns:
        results (list[tuple[str, float]]): (テキスト, スコア) のリスト.
    """
    if char_mask is not None:
        num_classes = pred.shape[2]
        if char_mask.shape[0] < num_classes:
            # モデル出力が辞書+1 (末尾 blank) の場合、マスクを拡張
            pad = np.full(num_classes - char_mask.shape[0], -np.inf, dtype=np.float32)
            char_mask = np.concatenate([char_mask, pad])
        pred = pred + char_mask[np.newaxis, np.newaxis, :num_classes]

    preds_idx = pred.argmax(axis=2)
    preds_prob = pred.max(axis=2)

    results: list[tuple[str, float]] = []

    for batch_idx in range(len(preds_idx)):
        text_index = preds_idx[batch_idx]
        text_prob = preds_prob[batch_idx]

        # 重複排除 + blank除去
        selection = np.ones(len(text_index), dtype=bool)
        selection[1:] = text_index[1:] != text_index[:-1]
        selection &= text_index != 0

        char_list: list[str] = []
        conf_list: list[float] = []
        for i, selected in enumerate(selection):
            if selected and text_index[i] < len(char_dict):
                char_list.append(char_dict[text_index[i]])
                conf_list.append(float(text_prob[i]))

        text = "".join(char_list)
        confidence = float(np.mean(conf_list)) if len(conf_list) > 0 else 0.0
        results.append((text, confidence))

    return results
