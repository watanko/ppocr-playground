"""OCR エンジン: Strategy パターンで PaddleOCR / ONNX を切り替える."""

from enum import StrEnum

from ppocr_playground.engine.base import OcrEngine


class EngineType(StrEnum):
    """OCR エンジンの種別.

    Attributes:
        PADDLE: PaddleOCR ベースのエンジン.
        ONNX_SAHI: ONNX Runtime + SAHI パッチ推論エンジン.
        ONNX_TWO_PHASE: ONNX Runtime 2フェーズ方式 (VRAM 節約型) エンジン.
    """

    PADDLE = "paddle"
    ONNX_SAHI = "onnx-sahi"
    ONNX_TWO_PHASE = "onnx-two-phase"


def create_engine(engine_type: EngineType, **kwargs: object) -> OcrEngine:
    """指定された種別の OCR エンジンを生成する.

    Args:
        engine_type (EngineType): エンジンの種別.
        **kwargs: エンジン固有のパラメータ.

    Returns:
        engine (OcrEngine): 生成されたエンジンインスタンス.
    """
    if engine_type == EngineType.PADDLE:
        from ppocr_playground.engine.paddle import PaddleOcrEngine

        return PaddleOcrEngine(**kwargs)  # type: ignore[arg-type]
    if engine_type == EngineType.ONNX_SAHI:
        from ppocr_playground.engine.onnx_sahi import OnnxSahiEngine

        return OnnxSahiEngine(**kwargs)  # type: ignore[arg-type]
    if engine_type == EngineType.ONNX_TWO_PHASE:
        from ppocr_playground.engine.onnx_two_phase import OnnxTwoPhaseEngine

        return OnnxTwoPhaseEngine(**kwargs)  # type: ignore[arg-type]

    msg = f"未知のエンジン種別: {engine_type}"
    raise ValueError(msg)


__all__ = ["EngineType", "OcrEngine", "create_engine"]
