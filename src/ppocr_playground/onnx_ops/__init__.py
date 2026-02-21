"""ONNX Runtime OCR の低レベル演算モジュール."""

# 定数
REC_IMAGE_SHAPE = (3, 48, 320)
DROP_SCORE = 0.0

CLS_BATCH_SIZE = 64
REC_BATCH_SIZE = 32

DET_LIMIT_SIDE_LEN = 64
DET_LIMIT_TYPE = "min"
DET_MAX_SIDE_LIMIT = 4000
DET_DB_THRESH = 0.3
DET_DB_BOX_THRESH = 0.6
DET_DB_UNCLIP_RATIO = 1.5

SAHI_TILE_SIZE = 960
SAHI_OVERLAP = 320
SAHI_NMS_THRESH = 0.3

LAYOUT_TARGET_SIZE = (800, 800)
LAYOUT_THRESHOLD = 0.5

CELL_DET_TARGET_SIZE = (640, 640)
CELL_DET_THRESHOLD = 0.3

TABLE_CLS_CROP_SIZE = 224

LAYOUT_LABELS = [
    "paragraph_title",
    "image",
    "text",
    "number",
    "abstract",
    "content",
    "figure_title",
    "formula",
    "table",
    "reference",
    "doc_title",
    "footnote",
    "header",
    "algorithm",
    "footer",
    "seal",
    "chart",
    "formula_number",
    "aside_text",
    "reference_content",
]

TEXT_LABELS = {
    "paragraph_title",
    "text",
    "number",
    "abstract",
    "content",
    "doc_title",
    "footnote",
    "header",
    "footer",
    "reference",
    "aside_text",
    "formula_number",
    "reference_content",
    "algorithm",
    "figure_title",
    "formula",
    "seal",
}
