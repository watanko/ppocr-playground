"""テーブル認識パイプライン."""

import csv
import io

from ppocr_playground.models import CellBox, OcrItem, TableCell, TableResult


def run_table_recognition(image_path: str, lang: str = "japan") -> TableResult:
    """PP-StructureV3でテーブル画像を認識する.

    Args:
        image_path (str): 入力テーブル画像のファイルパス.
        lang (str): OCRの言語設定.

    Returns:
        result (TableResult): テーブル認識結果.
    """
    from paddleocr import PPStructureV3

    pipeline = PPStructureV3(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_table_recognition=True,
        use_formula_recognition=False,
        use_seal_recognition=False,
        use_chart_recognition=False,
    )
    predictions = pipeline.predict(image_path)
    r = predictions[0]
    res = r.json["res"]

    table_list = res.get("table_res_list", [])
    table_data = table_list[0] if table_list else {}
    html = table_data.get("pred_html", "")

    cells, grid = parse_html_table(html)

    cell_boxes = [
        CellBox(bbox=[float(c) for c in box])
        for box in table_data.get("cell_box_list", [])
    ]

    ocr_pred = table_data.get("table_ocr_pred", {})
    ocr_items: list[OcrItem] = []
    for text, score, bbox in zip(
        ocr_pred.get("rec_texts", []),
        ocr_pred.get("rec_scores", []),
        ocr_pred.get("rec_boxes", []),
    ):
        ocr_items.append(
            OcrItem(text=text, score=float(score), bbox=[float(c) for c in bbox])
        )

    return TableResult(
        input_path=image_path,
        html=html,
        cells=cells,
        grid=grid,
        cell_boxes=cell_boxes,
        ocr_items=ocr_items,
    )


def parse_html_table(html: str) -> tuple[list[TableCell], list[list[str]]]:
    """HTMLテーブルをパースしてセルリストと2次元グリッドを生成する.

    Args:
        html (str): HTMLテーブル文字列.

    Returns:
        cells (list[TableCell]): セルのリスト.
        grid (list[list[str]]): 2次元グリッド表現.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if not table:
        return [], []

    rows = table.find_all("tr")

    # まずセル情報を収集
    cells: list[TableCell] = []
    # 結合セルの占有状況を追跡するマップ
    occupied: dict[tuple[int, int], bool] = {}

    for row_idx, row in enumerate(rows):
        col_idx = 0
        for td in row.find_all(["td", "th"]):
            # 既に結合セルで占有されている列をスキップ
            while occupied.get((row_idx, col_idx), False):
                col_idx += 1

            rowspan = int(td.get("rowspan", 1))
            colspan = int(td.get("colspan", 1))
            text = td.get_text(strip=True)

            cells.append(
                TableCell(
                    row=row_idx,
                    col=col_idx,
                    rowspan=rowspan,
                    colspan=colspan,
                    text=text,
                )
            )

            # 占有マップを更新
            for dr in range(rowspan):
                for dc in range(colspan):
                    occupied[(row_idx + dr, col_idx + dc)] = True

            col_idx += colspan

    # グリッドサイズを決定
    max_row = max((c.row + c.rowspan for c in cells), default=0)
    max_col = max((c.col + c.colspan for c in cells), default=0)

    # 2次元グリッドを構築（結合セルは左上セルのテキストで埋める）
    grid: list[list[str]] = [[""] * max_col for _ in range(max_row)]
    for cell in cells:
        for dr in range(cell.rowspan):
            for dc in range(cell.colspan):
                grid[cell.row + dr][cell.col + dc] = cell.text

    return cells, grid


def grid_to_csv(grid: list[list[str]]) -> str:
    """2次元グリッドをCSV文字列に変換する.

    Args:
        grid (list[list[str]]): 2次元グリッド表現.

    Returns:
        csv_str (str): CSV形式の文字列.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    for row in grid:
        writer.writerow(row)
    return output.getvalue()


def grid_to_markdown(grid: list[list[str]]) -> str:
    """2次元グリッドをMarkdownテーブル文字列に変換する.

    Args:
        grid (list[list[str]]): 2次元グリッド表現.

    Returns:
        md_str (str): Markdownテーブル形式の文字列.
    """
    if not grid:
        return ""

    def escape(text: str) -> str:
        return text.replace("|", "\\|")

    lines: list[str] = []
    for i, row in enumerate(grid):
        line = "| " + " | ".join(escape(cell) for cell in row) + " |"
        lines.append(line)
        if i == 0:
            lines.append("| " + " | ".join("---" for _ in row) + " |")

    return "\n".join(lines) + "\n"
