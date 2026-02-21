"""ONNX Runtime 2フェーズ方式 OCR エンジン (VRAM 節約型)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ppocr_playground.engine.base import OcrEngine
from ppocr_playground.models import OcrResult, OcrTextItem
from ppocr_playground.onnx_ops import CLS_BATCH_SIZE, DROP_SCORE, REC_BATCH_SIZE
from ppocr_playground.onnx_ops.classification import postprocess_cls, preprocess_cls
from ppocr_playground.onnx_ops.crop import get_rotate_crop_image, sort_boxes
from ppocr_playground.onnx_ops.detection import detect_single
from ppocr_playground.onnx_ops.recognition import (
    build_char_mask,
    postprocess_rec,
    preprocess_rec_batch,
)
from ppocr_playground.onnx_ops.session import (
    PROVIDERS,
    load_allowed_chars,
    load_character_dict,
)

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "models"


class OnnxTwoPhaseEngine(OcrEngine):
    """ONNX Runtime 2フェーズ方式エンジン.

    det と rec+cls のセッションを同時に保持せず、フェーズごとに
    生成・破棄することで VRAM のピーク使用量を抑える。

    Attributes:
        model_dir: モデルディレクトリのパス.
        score_thresh: スコア閾値.
        char_dict: 文字辞書.
        has_cls: 方向分類モデルが存在するかどうか.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        score_thresh: float = DROP_SCORE,
        lang: str | None = None,
    ) -> None:
        """エンジンを初期化する（セッションはまだ作らない）.

        Args:
            model_dir (str | None): モデルディレクトリ. None の場合はデフォルト.
            score_thresh (float): スコア閾値.
            lang (str | None): 言語フィルタ. "en" で英語+数字のみに制限.
        """
        self.model_dir = (
            Path(model_dir) if model_dir is not None else _DEFAULT_MODEL_DIR
        )
        self.score_thresh = score_thresh
        self.char_dict = load_character_dict(self.model_dir / "rec_char_dict.txt")
        self.has_cls = (self.model_dir / "PP-LCNet_x1_0_textline_ori.onnx").exists()

        self.char_mask: np.ndarray | None = None
        if lang == "en":
            allowed = load_allowed_chars(self.model_dir / "en_dict.txt")
            self.char_mask = build_char_mask(self.char_dict, allowed)

    def _create_det_session(self) -> ort.InferenceSession:
        """検出モデルのセッションを生成する.

        Returns:
            session (ort.InferenceSession): 検出モデルセッション.
        """
        return ort.InferenceSession(
            str(self.model_dir / "PP-OCRv5_server_det.onnx"), providers=PROVIDERS
        )

    def _create_rec_session(self) -> ort.InferenceSession:
        """認識モデルのセッションを生成する.

        Returns:
            session (ort.InferenceSession): 認識モデルセッション.
        """
        return ort.InferenceSession(
            str(self.model_dir / "PP-OCRv5_server_rec.onnx"), providers=PROVIDERS
        )

    def _create_cls_session(self) -> ort.InferenceSession | None:
        """方向分類モデルのセッションを生成する.

        Returns:
            session (ort.InferenceSession | None): 分類モデルセッション.
        """
        if not self.has_cls:
            return None
        return ort.InferenceSession(
            str(self.model_dir / "PP-LCNet_x1_0_textline_ori.onnx"),
            providers=PROVIDERS,
        )

    def run(self, image_path: str) -> OcrResult:
        """ONNX Runtime で OCR 推論を行う（2フェーズ方式）.

        フェーズ1: det セッション生成 → 検出 → det セッション破棄
        フェーズ2: rec+cls セッション生成 → 認識 → rec+cls セッション破棄

        Args:
            image_path (str): 入力画像のファイルパス.

        Returns:
            result (OcrResult): OCR推論結果.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"画像を読み込めません: {image_path}")

        ori_im = image.copy()

        # === フェーズ1: 検出（det セッション） ===
        det_session = self._create_det_session()
        dt_boxes, _det_scores = detect_single(image, det_session)
        del det_session  # det のアリーナを解放

        if len(dt_boxes) == 0:
            return OcrResult(input_path=image_path, text_count=0, items=[])

        dt_boxes = sort_boxes(dt_boxes)

        img_crop_list: list[np.ndarray] = []
        for box in dt_boxes:
            tmp_box = copy.deepcopy(box)
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        # === フェーズ2: 分類 + 認識（rec+cls セッション） ===
        cls_session = self._create_cls_session()
        rec_session = self._create_rec_session()

        orientations: list[int] = [0] * len(img_crop_list)
        if cls_session is not None:
            cls_input_name = cls_session.get_inputs()[0].name
            for bi in range(0, len(img_crop_list), CLS_BATCH_SIZE):
                batch = img_crop_list[bi : bi + CLS_BATCH_SIZE]
                cls_input = preprocess_cls(batch)
                cls_output = cls_session.run(None, {cls_input_name: cls_input})[0]
                orientations[bi : bi + CLS_BATCH_SIZE] = postprocess_cls(cls_output)

            for i, ori in enumerate(orientations):
                if ori == 1:
                    img_crop_list[i] = cv2.rotate(img_crop_list[i], cv2.ROTATE_180)

        rec_input_name = rec_session.get_inputs()[0].name
        rec_results: list[tuple[str, float]] = []
        for bi in range(0, len(img_crop_list), REC_BATCH_SIZE):
            batch = img_crop_list[bi : bi + REC_BATCH_SIZE]
            rec_input = preprocess_rec_batch(batch)
            rec_output = rec_session.run(None, {rec_input_name: rec_input})[0]
            rec_results.extend(
                postprocess_rec(rec_output, self.char_dict, self.char_mask)
            )

        del rec_session  # rec+cls のアリーナを解放
        del cls_session

        # 結果組み立て
        items: list[OcrTextItem] = []
        for i, (text, score) in enumerate(rec_results):
            if score < self.score_thresh:
                continue
            if not text.strip():
                continue
            poly = dt_boxes[i].astype(int).tolist()
            angle = orientations[i]
            items.append(OcrTextItem(text=text, score=score, polygon=poly, angle=angle))

        return OcrResult(
            input_path=image_path,
            text_count=len(items),
            items=items,
        )

    def run_batch(
        self,
        image_paths: list[str],
        on_det_done: Callable[[int, str], None] | None = None,
        on_rec_done: Callable[[int, str], None] | None = None,
    ) -> list[OcrResult]:
        """複数画像をバッチで OCR 推論する（2フェーズ方式）.

        フェーズ1: det セッション生成 → 全画像の検出 → det セッション破棄
        フェーズ2: rec+cls セッション生成 → 全クロップの認識 → rec+cls セッション破棄

        Args:
            image_paths (list[str]): 入力画像のファイルパスリスト.
            on_det_done (Callable | None): 検出完了時コールバック (index, path).
            on_rec_done (Callable | None): 認識完了時コールバック (index, path).

        Returns:
            results (list[OcrResult]): 各画像のOCR推論結果.
        """
        # === フェーズ1: 全画像の検出 ===
        det_session = self._create_det_session()

        det_results: list[_DetResult | None] = []
        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"画像を読み込めません: {image_path}")

            ori_im = image.copy()
            dt_boxes, _scores = detect_single(image, det_session)

            if len(dt_boxes) == 0:
                det_results.append(None)
            else:
                dt_boxes = sort_boxes(dt_boxes)
                crops = [
                    get_rotate_crop_image(ori_im, copy.deepcopy(box))
                    for box in dt_boxes
                ]
                det_results.append(
                    _DetResult(
                        image_path=image_path, dt_boxes=dt_boxes, img_crops=crops
                    )
                )

            if on_det_done is not None:
                on_det_done(i, image_path)

        del det_session

        # === フェーズ2: 全クロップの分類 + 認識 ===
        cls_session = self._create_cls_session()
        rec_session = self._create_rec_session()
        rec_input_name = rec_session.get_inputs()[0].name

        results: list[OcrResult] = []
        for i, det_r in enumerate(det_results):
            if det_r is None:
                results.append(
                    OcrResult(input_path=image_paths[i], text_count=0, items=[])
                )
                if on_rec_done is not None:
                    on_rec_done(i, image_paths[i])
                continue

            img_crop_list = det_r.img_crops

            # 方向分類
            orientations: list[int] = [0] * len(img_crop_list)
            if cls_session is not None:
                cls_input_name = cls_session.get_inputs()[0].name
                for bi in range(0, len(img_crop_list), CLS_BATCH_SIZE):
                    batch = img_crop_list[bi : bi + CLS_BATCH_SIZE]
                    cls_input = preprocess_cls(batch)
                    cls_output = cls_session.run(None, {cls_input_name: cls_input})[0]
                    orientations[bi : bi + CLS_BATCH_SIZE] = postprocess_cls(cls_output)

                for j, ori in enumerate(orientations):
                    if ori == 1:
                        img_crop_list[j] = cv2.rotate(img_crop_list[j], cv2.ROTATE_180)

            # 認識
            rec_results: list[tuple[str, float]] = []
            for bi in range(0, len(img_crop_list), REC_BATCH_SIZE):
                batch = img_crop_list[bi : bi + REC_BATCH_SIZE]
                rec_input = preprocess_rec_batch(batch)
                rec_output = rec_session.run(None, {rec_input_name: rec_input})[0]
                rec_results.extend(
                    postprocess_rec(rec_output, self.char_dict, self.char_mask)
                )

            # 結果組み立て
            items: list[OcrTextItem] = []
            for j, (text, score) in enumerate(rec_results):
                if score < self.score_thresh:
                    continue
                if not text.strip():
                    continue
                poly = det_r.dt_boxes[j].astype(int).tolist()
                angle = orientations[j]
                items.append(
                    OcrTextItem(text=text, score=score, polygon=poly, angle=angle)
                )

            results.append(
                OcrResult(
                    input_path=det_r.image_path,
                    text_count=len(items),
                    items=items,
                )
            )

            if on_rec_done is not None:
                on_rec_done(i, det_r.image_path)

        del rec_session
        del cls_session

        return results


@dataclass
class _DetResult:
    """1画像分の検出フェーズ中間結果.

    Attributes:
        image_path: 入力画像のパス.
        dt_boxes: ソート済みテキストボックス.
        img_crops: クロップ済み画像のリスト.
    """

    image_path: str
    dt_boxes: list[np.ndarray]
    img_crops: list[np.ndarray]
