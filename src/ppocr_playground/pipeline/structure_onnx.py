"""ONNX Runtime 版の文書構造解析パイプライン."""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ppocr_playground.models import (
    LayoutBox,
    OcrResult,
    OcrTextItem,
    ParsingBlock,
    StructureResult,
)
from ppocr_playground.onnx_ops import (
    CELL_DET_TARGET_SIZE,
    CELL_DET_THRESHOLD,
    CLS_BATCH_SIZE,
    LAYOUT_LABELS,
    LAYOUT_TARGET_SIZE,
    LAYOUT_THRESHOLD,
    REC_BATCH_SIZE,
    TEXT_LABELS,
)
from ppocr_playground.onnx_ops.classification import postprocess_cls, preprocess_cls
from ppocr_playground.onnx_ops.crop import get_rotate_crop_image, sort_boxes
from ppocr_playground.onnx_ops.detr import preprocess_table_cls, run_detr
from ppocr_playground.onnx_ops.recognition import postprocess_rec, preprocess_rec_batch
from ppocr_playground.onnx_ops.sahi import detect_with_sahi
from ppocr_playground.onnx_ops.session import create_session, load_character_dict

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "models"


def _match_ocr_to_region(
    ocr_result: OcrResult,
    bbox: list[int],
    used_indices: set[int],
) -> tuple[str, set[int]]:
    """OCR結果から bbox 内に含まれるテキストを抽出する.

    ボックスの重心が bbox 内にあるテキストを抽出して改行で連結する。

    Args:
        ocr_result (OcrResult): OCR推論結果.
        bbox (list[int]): 領域のバウンディングボックス [x1, y1, x2, y2].
        used_indices (set[int]): 既に使用済みのOCR項目インデックス.

    Returns:
        text (str): 改行連結されたテキスト.
        new_used (set[int]): 更新済みの使用済みインデックス集合.
    """
    x1, y1, x2, y2 = bbox
    matched: list[tuple[float, str]] = []
    new_used = set(used_indices)

    for i, item in enumerate(ocr_result.items):
        if i in new_used:
            continue
        poly = np.array(item.polygon, dtype=np.float32)
        cx = float(poly[:, 0].mean())
        cy = float(poly[:, 1].mean())
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            matched.append((cy, item.text))
            new_used.add(i)

    # Y座標順にソートして結合
    matched.sort(key=lambda t: t[0])
    return "\n".join(t[1] for t in matched), new_used


def _ocr_region(
    region_img: np.ndarray,
    det_session: ort.InferenceSession,
    rec_session: ort.InferenceSession,
    cls_session: ort.InferenceSession | None,
    char_dict: list[str],
) -> str:
    """画像領域に対してテキスト検出+認識を行い、テキストを返す.

    Args:
        region_img (np.ndarray): 対象領域画像 (BGR, HWC).
        det_session (ort.InferenceSession): テキスト検出セッション.
        rec_session (ort.InferenceSession): テキスト認識セッション.
        cls_session (ort.InferenceSession | None): 方向分類セッション（任意）.
        char_dict (list[str]): 文字辞書.

    Returns:
        text (str): 認識テキスト（改行区切り）.
    """
    dt_boxes, _scores = detect_with_sahi(region_img, det_session)
    if len(dt_boxes) == 0:
        return ""

    dt_boxes = sort_boxes(dt_boxes)

    # クロップ
    img_crops: list[np.ndarray] = []
    for box in dt_boxes:
        crop = get_rotate_crop_image(region_img, box.copy())
        img_crops.append(crop)

    # 方向分類
    if cls_session is not None:
        cls_input_name = cls_session.get_inputs()[0].name
        for bi in range(0, len(img_crops), CLS_BATCH_SIZE):
            batch = img_crops[bi : bi + CLS_BATCH_SIZE]
            cls_input = preprocess_cls(batch)
            cls_output = cls_session.run(None, {cls_input_name: cls_input})[0]
            labels = postprocess_cls(cls_output)
            for j, label in enumerate(labels):
                if label == 1:
                    img_crops[bi + j] = cv2.rotate(img_crops[bi + j], cv2.ROTATE_180)

    # テキスト認識
    rec_input_name = rec_session.get_inputs()[0].name
    texts: list[str] = []
    for bi in range(0, len(img_crops), REC_BATCH_SIZE):
        batch = img_crops[bi : bi + REC_BATCH_SIZE]
        rec_input = preprocess_rec_batch(batch)
        rec_output = rec_session.run(None, {rec_input_name: rec_input})[0]
        results = postprocess_rec(rec_output, char_dict)
        for text, _score in results:
            if text.strip():
                texts.append(text.strip())

    return "\n".join(texts)


def _process_table_region(
    image: np.ndarray,
    table_bbox: list[int],
    table_cls_session: ort.InferenceSession | None,
    wired_cell_session: ort.InferenceSession | None,
    wireless_cell_session: ort.InferenceSession | None,
    det_session: ort.InferenceSession,
    rec_session: ort.InferenceSession,
    cls_session: ort.InferenceSession | None,
    char_dict: list[str],
) -> str:
    """テーブル領域のセル検出 + OCR を行い、テキストを返す.

    Args:
        image (np.ndarray): 全体画像 (BGR, HWC).
        table_bbox (list[int]): テーブル領域 [x1, y1, x2, y2].
        table_cls_session: テーブル分類セッション（任意）.
        wired_cell_session: 有線テーブルセル検出セッション（任意）.
        wireless_cell_session: 無線テーブルセル検出セッション（任意）.
        det_session: テキスト検出セッション.
        rec_session: テキスト認識セッション.
        cls_session: テキスト行方向分類セッション（任意）.
        char_dict (list[str]): 文字辞書.

    Returns:
        content (str): テーブル内のテキスト.
    """
    x1, y1, x2, y2 = table_bbox
    table_img = image[y1:y2, x1:x2]

    if table_img.size == 0:
        return ""

    # セル検出セッションが無い場合はテーブル領域でテキスト検出
    if wired_cell_session is None and wireless_cell_session is None:
        return _ocr_region(table_img, det_session, rec_session, cls_session, char_dict)

    # テーブル分類
    is_wired = True
    if table_cls_session is not None:
        cls_input = preprocess_table_cls(table_img)
        cls_name = table_cls_session.get_inputs()[0].name
        cls_output = table_cls_session.run(None, {cls_name: cls_input})[0]
        is_wired = int(cls_output[0].argmax()) == 0

    cell_session = wired_cell_session if is_wired else wireless_cell_session
    if cell_session is None:
        cell_session = wired_cell_session or wireless_cell_session
    assert cell_session is not None

    # セル検出
    cells = run_detr(
        cell_session, table_img, CELL_DET_TARGET_SIZE, ["cell"], CELL_DET_THRESHOLD
    )

    if len(cells) == 0:
        return _ocr_region(table_img, det_session, rec_session, cls_session, char_dict)

    # セルを上→下、左→右でソート
    cells.sort(key=lambda c: (c["coordinate"][1], c["coordinate"][0]))  # type: ignore[index]

    # 各セルのOCRテキストを取得
    cell_texts: list[str] = []
    for cell in cells:
        cx1, cy1, cx2, cy2 = cell["coordinate"]  # type: ignore[index]
        cell_img = table_img[cy1:cy2, cx1:cx2]
        if cell_img.size == 0:
            continue
        text = _ocr_region(cell_img, det_session, rec_session, cls_session, char_dict)
        if text.strip():
            cell_texts.append(text.strip())

    return "\n".join(cell_texts)


def _sort_blocks(
    blocks: list[ParsingBlock],
) -> list[ParsingBlock]:
    """パースブロックを読み順（上→下、左→右）にソートする.

    Args:
        blocks (list[ParsingBlock]): パースブロックのリスト.

    Returns:
        sorted_blocks (list[ParsingBlock]): ソート済みブロックリスト.
    """
    sorted_blocks = sorted(blocks, key=lambda b: (b.block_bbox[1], b.block_bbox[0]))
    for i, block in enumerate(sorted_blocks):
        block.block_order = i
        block.block_id = i
    return sorted_blocks


def run_structure(
    image_path: str,
    model_dir: str | None = None,
) -> StructureResult:
    """ONNX Runtimeを使って文書構造解析を行う.

    パイプライン: レイアウト検出 → 全画像OCR → 領域ごとテキスト割当
    テーブル領域はセル検出 + セルOCR で処理する。

    Args:
        image_path (str): 入力画像のファイルパス.
        model_dir (str | None): モデルディレクトリ. Noneの場合はデフォルト.

    Returns:
        result (StructureResult): 文書構造解析結果.
    """
    mdir = Path(model_dir) if model_dir is not None else _DEFAULT_MODEL_DIR

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"画像を読み込めません: {image_path}")
    ori_h, ori_w = image.shape[:2]

    # --- モデルロード ---
    layout_session = create_session(mdir / "PP-DocLayout_plus-L.onnx")

    # OCR用モデル
    rec_session = create_session(mdir / "PP-OCRv5_server_rec.onnx")
    char_dict = load_character_dict(mdir / "rec_char_dict.txt")
    det_session = create_session(mdir / "PP-OCRv5_server_det.onnx")

    cls_path = mdir / "PP-LCNet_x1_0_textline_ori.onnx"
    cls_session = create_session(cls_path) if cls_path.exists() else None

    # テーブル用モデル（任意）
    table_cls_path = mdir / "PP-LCNet_x1_0_table_cls.onnx"
    table_cls_session = (
        create_session(table_cls_path) if table_cls_path.exists() else None
    )
    wired_path = mdir / "RT-DETR-L_wired_table_cell_det.onnx"
    wired_session = create_session(wired_path) if wired_path.exists() else None
    wireless_path = mdir / "RT-DETR-L_wireless_table_cell_det.onnx"
    wireless_session = create_session(wireless_path) if wireless_path.exists() else None

    # --- 1. レイアウト検出 ---
    print("  レイアウト検出中...")
    layout_boxes_raw = run_detr(
        layout_session, image, LAYOUT_TARGET_SIZE, LAYOUT_LABELS, LAYOUT_THRESHOLD
    )
    print(f"  レイアウト検出: {len(layout_boxes_raw)} 領域")

    # --- 2. 全画像テキスト検出 + 認識（OCR） ---
    print("  テキスト検出中...")
    dt_boxes, _det_scores = detect_with_sahi(image, det_session)
    print(f"  テキスト検出: {len(dt_boxes)} ボックス")

    ocr_items: list[OcrTextItem] = []
    if len(dt_boxes) > 0:
        dt_boxes = sort_boxes(dt_boxes)

        img_crops: list[np.ndarray] = []
        for box in dt_boxes:
            crop = get_rotate_crop_image(image, box.copy())
            img_crops.append(crop)

        # 方向分類
        orientations: list[int] = [0] * len(img_crops)
        if cls_session is not None:
            cls_input_name = cls_session.get_inputs()[0].name
            for bi in range(0, len(img_crops), CLS_BATCH_SIZE):
                batch = img_crops[bi : bi + CLS_BATCH_SIZE]
                cls_input = preprocess_cls(batch)
                cls_output = cls_session.run(None, {cls_input_name: cls_input})[0]
                orientations[bi : bi + CLS_BATCH_SIZE] = postprocess_cls(cls_output)
            for i, ori in enumerate(orientations):
                if ori == 1:
                    img_crops[i] = cv2.rotate(img_crops[i], cv2.ROTATE_180)

        # テキスト認識
        print("  テキスト認識中...")
        rec_input_name = rec_session.get_inputs()[0].name
        rec_results: list[tuple[str, float]] = []
        for bi in range(0, len(img_crops), REC_BATCH_SIZE):
            batch = img_crops[bi : bi + REC_BATCH_SIZE]
            rec_input = preprocess_rec_batch(batch)
            rec_output = rec_session.run(None, {rec_input_name: rec_input})[0]
            rec_results.extend(postprocess_rec(rec_output, char_dict))

        for i, (text, score) in enumerate(rec_results):
            if not text.strip():
                continue
            poly = dt_boxes[i].astype(int).tolist()
            ocr_items.append(
                OcrTextItem(text=text, score=score, polygon=poly, angle=orientations[i])
            )

    ocr_result = OcrResult(
        input_path=image_path, text_count=len(ocr_items), items=ocr_items
    )
    print(f"  OCRテキスト: {len(ocr_items)} 項目")

    # --- 3. LayoutBox リスト構築 ---
    layout_boxes: list[LayoutBox] = []
    for box in layout_boxes_raw:
        layout_boxes.append(
            LayoutBox(
                label=str(box["label"]),
                score=float(box["score"]),  # type: ignore[arg-type]
                coordinate=[float(c) for c in box["coordinate"]],  # type: ignore[union-attr]
            )
        )

    # --- 4. 各レイアウト領域にコンテンツ割り当て ---
    print("  コンテンツ割り当て中...")
    parsing_blocks: list[ParsingBlock] = []
    used_ocr_indices: set[int] = set()

    for idx, box in enumerate(layout_boxes_raw):
        label = str(box["label"])
        bbox = [int(c) for c in box["coordinate"]]  # type: ignore[union-attr]

        if label == "table":
            content = _process_table_region(
                image,
                bbox,
                table_cls_session,
                wired_session,
                wireless_session,
                det_session,
                rec_session,
                cls_session,
                char_dict,
            )
            _, used_ocr_indices = _match_ocr_to_region(
                ocr_result, bbox, used_ocr_indices
            )
        elif label in TEXT_LABELS:
            content, used_ocr_indices = _match_ocr_to_region(
                ocr_result, bbox, used_ocr_indices
            )
        else:
            content = ""

        parsing_blocks.append(
            ParsingBlock(
                block_id=idx,
                block_order=idx,
                block_label=label,
                block_bbox=[float(c) for c in bbox],
                block_content=content,
            )
        )

    # 読み順ソート
    parsing_blocks = _sort_blocks(parsing_blocks)

    return StructureResult(
        input_path=image_path,
        width=ori_w,
        height=ori_h,
        layout_boxes=layout_boxes,
        parsing_blocks=parsing_blocks,
    )
