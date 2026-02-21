"""ONNX Runtime + SAHI パッチ推論 OCR エンジン."""

import copy
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ppocr_playground.engine.base import OcrEngine
from ppocr_playground.models import OcrResult, OcrTextItem
from ppocr_playground.onnx_ops import CLS_BATCH_SIZE, DROP_SCORE, REC_BATCH_SIZE
from ppocr_playground.onnx_ops.classification import postprocess_cls, preprocess_cls
from ppocr_playground.onnx_ops.crop import get_rotate_crop_image, sort_boxes
from ppocr_playground.onnx_ops.recognition import (
    build_char_mask,
    postprocess_rec,
    preprocess_rec_batch,
)
from ppocr_playground.onnx_ops.sahi import detect_with_sahi
from ppocr_playground.onnx_ops.session import (
    create_session,
    load_allowed_chars,
    load_character_dict,
)

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "models"


class OnnxSahiEngine(OcrEngine):
    """ONNX Runtime + SAHI パッチ推論エンジン.

    3つのセッション (det/rec/cls) を保持し、SAHI パッチ推論で検出する。

    Attributes:
        model_dir: モデルディレクトリのパス.
        score_thresh: スコア閾値.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        score_thresh: float = DROP_SCORE,
        lang: str | None = None,
    ) -> None:
        """エンジンを初期化する.

        Args:
            model_dir (str | None): モデルディレクトリ. None の場合はデフォルト.
            score_thresh (float): スコア閾値.
            lang (str | None): 言語フィルタ. "en" で英語+数字のみに制限.
        """
        self.model_dir = (
            Path(model_dir) if model_dir is not None else _DEFAULT_MODEL_DIR
        )
        self.score_thresh = score_thresh
        self.lang = lang

        self._det_session: ort.InferenceSession | None = None
        self._rec_session: ort.InferenceSession | None = None
        self._cls_session: ort.InferenceSession | None = None
        self._char_dict: list[str] | None = None
        self._char_mask: np.ndarray | None = None

    def _ensure_loaded(self) -> None:
        """モデルを遅延ロードする."""
        if self._det_session is not None:
            return

        self._det_session = create_session(self.model_dir / "PP-OCRv5_server_det.onnx")
        self._rec_session = create_session(self.model_dir / "PP-OCRv5_server_rec.onnx")
        self._char_dict = load_character_dict(self.model_dir / "rec_char_dict.txt")

        cls_path = self.model_dir / "PP-LCNet_x1_0_textline_ori.onnx"
        self._cls_session = create_session(cls_path) if cls_path.exists() else None

        if self.lang == "en":
            allowed = load_allowed_chars(self.model_dir / "en_dict.txt")
            self._char_mask = build_char_mask(self._char_dict, allowed)

    def run(self, image_path: str) -> OcrResult:
        """ONNX Runtime + SAHI で OCR 推論を行う.

        Args:
            image_path (str): 入力画像のファイルパス.

        Returns:
            result (OcrResult): OCR推論結果.
        """
        self._ensure_loaded()
        assert self._det_session is not None
        assert self._rec_session is not None
        assert self._char_dict is not None

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"画像を読み込めません: {image_path}")

        ori_im = image.copy()

        # 1. テキスト検出（SAHI パッチ推論）
        dt_boxes, _det_scores = detect_with_sahi(image, self._det_session)

        if len(dt_boxes) == 0:
            return OcrResult(input_path=image_path, text_count=0, items=[])

        # 2. ボックスソート
        dt_boxes = sort_boxes(dt_boxes)

        # 3. テキスト領域クロップ
        img_crop_list: list[np.ndarray] = []
        for box in dt_boxes:
            tmp_box = copy.deepcopy(box)
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        # 4. テキスト行方向分類
        orientations: list[int] = [0] * len(img_crop_list)
        if self._cls_session is not None:
            cls_input_name = self._cls_session.get_inputs()[0].name
            for bi in range(0, len(img_crop_list), CLS_BATCH_SIZE):
                batch = img_crop_list[bi : bi + CLS_BATCH_SIZE]
                cls_input = preprocess_cls(batch)
                cls_output = self._cls_session.run(None, {cls_input_name: cls_input})[0]
                orientations[bi : bi + CLS_BATCH_SIZE] = postprocess_cls(cls_output)

            for i, ori in enumerate(orientations):
                if ori == 1:
                    img_crop_list[i] = cv2.rotate(img_crop_list[i], cv2.ROTATE_180)

        # 5. テキスト認識
        rec_input_name = self._rec_session.get_inputs()[0].name
        rec_results: list[tuple[str, float]] = []
        for bi in range(0, len(img_crop_list), REC_BATCH_SIZE):
            batch = img_crop_list[bi : bi + REC_BATCH_SIZE]
            rec_input = preprocess_rec_batch(batch)
            rec_output = self._rec_session.run(None, {rec_input_name: rec_input})[0]
            rec_results.extend(
                postprocess_rec(rec_output, self._char_dict, self._char_mask)
            )

        # 6. 結果組み立て
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
