"""ONNX Runtime セッション管理・文字辞書読み込み."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

# VRAM 節約用プロバイダオプション
PROVIDERS: list[str | tuple[str, dict[str, str]]] = [
    (
        "CUDAExecutionProvider",
        {
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_use_max_workspace": "0",
            "cudnn_conv_algo_search": "HEURISTIC",
        },
    ),
    "CPUExecutionProvider",
]

# デフォルト（アリーナ制限なし）
PROVIDERS_DEFAULT: list[str] = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

# CLS 前処理の ImageNet 統計量
CLS_RESIZE = (160, 80)  # (w, h)
CLS_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
CLS_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def create_session(
    onnx_path: Path,
    *,
    vram_saving: bool = False,
) -> ort.InferenceSession:
    """ONNX モデルの推論セッションを作成する.

    Args:
        onnx_path (Path): ONNX モデルファイルのパス.
        vram_saving (bool): True の場合 VRAM 節約用プロバイダオプションを使用する.

    Returns:
        session (ort.InferenceSession): 推論セッション.
    """
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNXモデルが見つかりません: {onnx_path}")
    providers = PROVIDERS if vram_saving else PROVIDERS_DEFAULT
    return ort.InferenceSession(str(onnx_path), providers=providers)


def load_character_dict(dict_path: Path) -> list[str]:
    """文字辞書ファイルを読み込む.

    blank トークンを先頭に追加する。

    Args:
        dict_path (Path): 辞書ファイルのパス.

    Returns:
        character_list (list[str]): "blank" + 辞書文字のリスト.
    """
    if not dict_path.exists():
        raise FileNotFoundError(f"文字辞書が見つかりません: {dict_path}")
    with open(dict_path, encoding="utf-8") as f:
        chars = [line.strip() for line in f]
    return ["blank"] + chars


def load_allowed_chars(dict_path: Path) -> set[str]:
    """フィルタ用の許可文字セットを辞書ファイルから読み込む.

    Args:
        dict_path (Path): 辞書ファイルのパス (例: en_dict.txt).

    Returns:
        allowed (set[str]): 許可文字の集合.
    """
    if not dict_path.exists():
        raise FileNotFoundError(f"辞書が見つかりません: {dict_path}")
    with open(dict_path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}
