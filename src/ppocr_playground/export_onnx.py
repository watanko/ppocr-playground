"""PaddlePaddleモデルをONNX形式にエクスポートするスクリプト.

det / rec / cls / en_rec モデルを ~/.paddlex/official_models/ から読み込み、
models/ ディレクトリに .onnx ファイルとして出力する。
英語 rec モデルの辞書は inference.yml から抽出して .txt に保存する。

NOTE: paddle2onnx は Python 3.13 で利用不可のため、このスクリプトは
Google Colab 等の Python 3.10/3.11 環境で実行する必要がある。
"""

import subprocess
from pathlib import Path

import yaml

_PADDLEX_DIR = Path.home() / ".paddlex" / "official_models"
_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "models"

# エクスポート対象: (Paddleモデル名, 出力ONNXファイル名)
_MODELS: list[tuple[str, str]] = [
    ("PP-OCRv5_server_det", "PP-OCRv5_server_det.onnx"),
    ("PP-OCRv5_server_rec", "PP-OCRv5_server_rec.onnx"),
    ("PP-LCNet_x1_0_textline_ori", "PP-LCNet_x1_0_textline_ori.onnx"),
    ("en_PP-OCRv5_mobile_rec", "en_PP-OCRv5_mobile_rec.onnx"),
]


def export_model(model_name: str, output_filename: str) -> None:
    """Paddle推論モデルをONNX形式にエクスポートする.

    paddlex --paddle2onnx CLI を使用する。

    Args:
        model_name (str): ~/.paddlex/official_models/ 配下のモデルディレクトリ名.
        output_filename (str): 出力ONNXファイル名.
    """
    model_dir = _PADDLEX_DIR / model_name
    output_dir = _OUTPUT_DIR / model_name
    output_file = _OUTPUT_DIR / output_filename

    if not model_dir.exists():
        raise FileNotFoundError(f"モデルディレクトリが見つかりません: {model_dir}")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "paddlex",
        "--paddle2onnx",
        "--paddle_model_dir",
        str(model_dir),
        "--onnx_model_dir",
        str(output_dir),
        "--opset_version",
        "7",
    ]
    print(f"エクスポート中: {model_name}")
    print(f"  コマンド: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # paddlex は出力ディレクトリ内に inference.onnx を生成するのでリネーム
    inference_onnx = output_dir / "inference.onnx"
    if inference_onnx.exists():
        inference_onnx.rename(output_file)
        output_dir.rmdir()
        print(
            f"  完了: {output_file} ({output_file.stat().st_size / 1024 / 1024:.1f} MB)"
        )
    else:
        print(
            f"  警告: {inference_onnx} が見つかりません。"
            f"{output_dir} を確認してください。"
        )


def extract_char_dict(model_name: str, output_filename: str) -> None:
    """inference.yml から文字辞書を抽出してテキストファイルに保存する.

    Args:
        model_name (str): ~/.paddlex/official_models/ 配下のモデルディレクトリ名.
        output_filename (str): 出力辞書ファイル名.
    """
    yml_path = _PADDLEX_DIR / model_name / "inference.yml"
    if not yml_path.exists():
        raise FileNotFoundError(f"inference.yml が見つかりません: {yml_path}")

    with open(yml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    chars: list[str] = config["PostProcess"]["character_dict"]
    output_path = _OUTPUT_DIR / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(chars) + "\n", encoding="utf-8")
    print(f"辞書抽出: {output_path} ({len(chars)}文字)")


def run_export() -> None:
    """全モデルをONNXにエクスポートする."""
    for model_name, output_filename in _MODELS:
        export_model(model_name, output_filename)

    # 英語 rec モデルの辞書を抽出
    extract_char_dict("en_PP-OCRv5_mobile_rec", "en_dict.txt")

    print("\n全モデルのエクスポートが完了しました。")
