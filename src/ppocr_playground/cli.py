"""argparse ベースの統一 CLI エントリポイント."""

import argparse
import os
import sys
from pathlib import Path

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def _collect_images(input_path: Path) -> list[Path]:
    """入力パスから画像ファイルのリストを収集する.

    単一ファイルの場合はそのまま、ディレクトリの場合は
    画像拡張子で glob して名前順にソートする。

    Args:
        input_path (Path): 入力画像パスまたはディレクトリパス.

    Returns:
        images (list[Path]): 画像ファイルパスのリスト.
    """
    if input_path.is_dir():
        images = sorted(
            [p for p in input_path.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS],
            key=lambda p: p.name,
        )
        if not images:
            print(f"画像が見つかりません: {input_path}", file=sys.stderr)
            sys.exit(1)
        return images
    if not input_path.is_file():
        print(f"ファイルが存在しません: {input_path}", file=sys.stderr)
        sys.exit(1)
    return [input_path]


def _resolve_output_dir(
    input_path: Path,
    output_arg: str | None,
) -> Path:
    """出力ディレクトリを決定する.

    -o 指定時はそのパスを使用。未指定時は入力の親ディレクトリを使用。
    入力がディレクトリの場合、ディレクトリ名のサブフォルダを追加する。

    Args:
        input_path (Path): 入力パス（画像 or ディレクトリ）.
        output_arg (str | None): -o オプションの値.

    Returns:
        out_dir (Path): 出力先ディレクトリ.
    """
    base = Path(output_arg) if output_arg is not None else input_path.parent
    if input_path.is_dir():
        return base / input_path.name
    return base


def _cmd_ocr(args: argparse.Namespace) -> None:
    """OCR サブコマンドの実行.

    Args:
        args (argparse.Namespace): パース済みコマンドライン引数.
    """
    from ppocr_playground.engine import EngineType, create_engine
    from ppocr_playground.io import save_ocr_result
    from ppocr_playground.visualization.ocr import visualize

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    engine_type = EngineType(args.engine)
    kwargs: dict[str, object] = {}
    if args.model_dir is not None:
        kwargs["model_dir"] = args.model_dir
    if args.lang is not None:
        kwargs["lang"] = args.lang
    ocr_engine = create_engine(engine_type, **kwargs)

    input_path = Path(args.input)
    images = _collect_images(input_path)
    out_dir = _resolve_output_dir(input_path, args.output)

    if len(images) > 1:
        print(f"入力ディレクトリ: {input_path} ({len(images)}枚)")
        print(f"出力先: {out_dir}")

    for i, img_path in enumerate(images, 1):
        if len(images) > 1:
            print(f"[{i}/{len(images)}] {img_path.name}")
        else:
            print(f"入力: {img_path}")

        result = ocr_engine.run(str(img_path))
        stem = img_path.stem
        json_path = out_dir / f"{stem}.json"
        vis_path = out_dir / f"{stem}.png"

        save_ocr_result(result, str(json_path))
        visualize(str(img_path), result, str(vis_path), show_text=not args.no_text)

        print(
            f"  テキスト数: {result.text_count}"
            f" | JSON: {json_path}"
            f" | 可視化: {vis_path}"
        )

    if len(images) > 1:
        print(f"完了: {len(images)}枚処理しました")


def _cmd_structure(args: argparse.Namespace) -> None:
    """構造解析サブコマンドの実行.

    Args:
        args (argparse.Namespace): パース済みコマンドライン引数.
    """
    from ppocr_playground.io import save_structure_result
    from ppocr_playground.visualization.structure import visualize_structure

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    input_path = Path(args.input)
    images = _collect_images(input_path)
    out_dir = _resolve_output_dir(input_path, args.output)

    if len(images) > 1:
        print(f"入力ディレクトリ: {input_path} ({len(images)}枚)")
        print(f"出力先: {out_dir}")

    for i, img_path in enumerate(images, 1):
        if len(images) > 1:
            print(f"[{i}/{len(images)}] {img_path.name}")
        else:
            print(f"入力: {img_path}")

        stem = img_path.stem

        if args.backend == "onnx":
            from ppocr_playground.pipeline.structure_onnx import (
                run_structure as run_structure_onnx,
            )

            result = run_structure_onnx(str(img_path), model_dir=args.model_dir)
        else:
            from ppocr_playground.pipeline.structure_paddle import (
                run_structure as run_structure_paddle,
            )

            result = run_structure_paddle(str(img_path))

        json_path = out_dir / f"{stem}.json"
        vis_path = out_dir / f"{stem}.png"

        save_structure_result(result, str(json_path))
        visualize_structure(str(img_path), result, str(vis_path))

        print(
            f"  レイアウト: {len(result.layout_boxes)}"
            f" | ブロック: {len(result.parsing_blocks)}"
            f" | JSON: {json_path}"
            f" | 可視化: {vis_path}"
        )

    if len(images) > 1:
        print(f"完了: {len(images)}枚処理しました")


def _cmd_table(args: argparse.Namespace) -> None:
    """テーブル認識サブコマンドの実行.

    Args:
        args (argparse.Namespace): パース済みコマンドライン引数.
    """
    from ppocr_playground.io import save_table_result
    from ppocr_playground.pipeline.table import (
        grid_to_csv,
        grid_to_markdown,
        run_table_recognition,
    )
    from ppocr_playground.visualization.table import visualize_table

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    input_path = Path(args.input)
    images = _collect_images(input_path)
    out_dir = _resolve_output_dir(input_path, args.output)

    if len(images) > 1:
        print(f"入力ディレクトリ: {input_path} ({len(images)}枚)")
        print(f"出力先: {out_dir}")

    for i, img_path in enumerate(images, 1):
        if len(images) > 1:
            print(f"[{i}/{len(images)}] {img_path.name}")
        else:
            print(f"入力: {img_path}")

        stem = img_path.stem
        result = run_table_recognition(str(img_path))

        json_path = out_dir / f"{stem}.json"
        csv_path = out_dir / f"{stem}.csv"
        md_path = out_dir / f"{stem}.md"
        vis_path = out_dir / f"{stem}.png"

        # JSON
        save_table_result(result, str(json_path))

        # CSV
        csv_str = grid_to_csv(result.grid)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(csv_str, encoding="utf-8")

        # Markdown
        md_str = grid_to_markdown(result.grid)
        md_path.write_text(md_str, encoding="utf-8")

        # 可視化
        visualize_table(str(img_path), result, str(vis_path))

        cols = len(result.grid[0]) if result.grid else 0
        print(
            f"  セル数: {len(result.cells)}"
            f" | グリッド: {len(result.grid)}x{cols}"
            f" | JSON: {json_path}"
            f" | 可視化: {vis_path}"
        )
        if len(images) == 1:
            print()
            print(md_str)

    if len(images) > 1:
        print(f"完了: {len(images)}枚処理しました")


def _cmd_export_onnx(_args: argparse.Namespace) -> None:
    """ONNX エクスポートサブコマンドの実行.

    Args:
        _args (argparse.Namespace): パース済みコマンドライン引数（未使用）.
    """
    from ppocr_playground.export_onnx import run_export

    run_export()


def build_parser() -> argparse.ArgumentParser:
    """CLI の ArgumentParser を構築する.

    Returns:
        parser (argparse.ArgumentParser): 構築済みパーサ.
    """
    parser = argparse.ArgumentParser(
        prog="ppocr",
        description="PaddleOCR / ONNX Runtime OCR ツールキット",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- ocr ---
    p_ocr = subparsers.add_parser("ocr", help="OCR を実行する")
    p_ocr.add_argument(
        "-i", "--input", required=True, help="入力画像またはディレクトリのパス"
    )
    p_ocr.add_argument("-o", "--output", default=None, help="出力先ディレクトリ")
    p_ocr.add_argument(
        "--engine",
        default="paddle",
        choices=["paddle", "onnx-sahi", "onnx-two-phase"],
        help="OCRエンジン (default: paddle)",
    )
    p_ocr.add_argument(
        "--no-text",
        action="store_true",
        help="可視化でテキストラベルを非表示",
    )
    p_ocr.add_argument("--model-dir", default=None, help="ONNXモデルディレクトリ")
    p_ocr.add_argument(
        "--lang", default=None, choices=["en"], help="言語フィルタ (en: 英語+数字のみ)"
    )
    p_ocr.set_defaults(func=_cmd_ocr)

    # --- structure ---
    p_struct = subparsers.add_parser("structure", help="文書構造解析を実行する")
    p_struct.add_argument(
        "-i", "--input", required=True, help="入力画像またはディレクトリのパス"
    )
    p_struct.add_argument("-o", "--output", default=None, help="出力先ディレクトリ")
    p_struct.add_argument(
        "--backend",
        default="paddle",
        choices=["paddle", "onnx"],
        help="バックエンド (default: paddle)",
    )
    p_struct.add_argument("--model-dir", default=None, help="ONNXモデルディレクトリ")
    p_struct.set_defaults(func=_cmd_structure)

    # --- table ---
    p_table = subparsers.add_parser("table", help="テーブル認識を実行する")
    p_table.add_argument(
        "-i", "--input", required=True, help="入力テーブル画像またはディレクトリのパス"
    )
    p_table.add_argument("-o", "--output", default=None, help="出力先ディレクトリ")
    p_table.set_defaults(func=_cmd_table)

    # --- export-onnx ---
    p_export = subparsers.add_parser(
        "export-onnx", help="PaddlePaddle モデルを ONNX にエクスポート"
    )
    p_export.set_defaults(func=_cmd_export_onnx)

    return parser


def main() -> None:
    """CLI のメインエントリポイント."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
