"""PaddlePaddle 版の文書構造解析パイプライン."""

from ppocr_playground.models import LayoutBox, ParsingBlock, StructureResult


def run_structure(image_path: str, lang: str = "japan") -> StructureResult:
    """PP-StructureV3 (PaddlePaddle) で画像を推論し、結果を返す.

    Args:
        image_path (str): 入力画像のファイルパス.
        lang (str): OCRの言語設定.

    Returns:
        result (StructureResult): 文書構造解析結果.
    """
    from paddleocr import PPStructureV3

    pipeline = PPStructureV3(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        # 12GB VRAM環境でOOMを回避するためmobileモデルを使用
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        text_det_limit_side_len=960,
    )
    predictions = pipeline.predict(image_path)
    r = predictions[0]

    res = r.json["res"]

    layout_boxes: list[LayoutBox] = []
    for box in res.get("layout_det_res", {}).get("boxes", []):
        layout_boxes.append(
            LayoutBox(
                label=box["label"],
                score=float(box["score"]),
                coordinate=[float(c) for c in box["coordinate"]],
            )
        )

    parsing_blocks: list[ParsingBlock] = []
    for block in res.get("parsing_res_list", []):
        parsing_blocks.append(
            ParsingBlock(
                block_id=block.get("block_id", 0),
                block_order=block.get("block_order", 0),
                block_label=block["block_label"],
                block_bbox=[float(c) for c in block["block_bbox"]],
                block_content=block.get("block_content", ""),
            )
        )

    return StructureResult(
        input_path=image_path,
        width=res.get("width", 0),
        height=res.get("height", 0),
        layout_boxes=layout_boxes,
        parsing_blocks=parsing_blocks,
    )
