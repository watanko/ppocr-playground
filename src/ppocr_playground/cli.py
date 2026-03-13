"""argparse ベースの統一 CLI エントリポイント."""

import argparse
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
    base = Path(output_arg) if output_arg is not None else Path("output")
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


def build_parser() -> argparse.ArgumentParser:
    """CLI の ArgumentParser を構築する.

    Returns:
        parser (argparse.ArgumentParser): 構築済みパーサ.
    """
    parser = argparse.ArgumentParser(
        prog="ppocr",
        description="ONNX Runtime OCR ツールキット",
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
        default="onnx-sahi",
        choices=["onnx-sahi", "onnx-two-phase"],
        help="OCRエンジン (default: onnx-sahi)",
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

    return parser


def main() -> None:
    """CLI のメインエントリポイント."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
