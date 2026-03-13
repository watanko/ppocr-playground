# ppocr-playground

ONNX Runtime を用いた OCR ツールキット。

## 特徴

- **2つの OCR エンジン** を Strategy パターンで切り替え可能
  - `onnx-sahi` - ONNX Runtime + SAHI パッチ推論 (大画像対応, デフォルト)
  - `onnx-two-phase` - ONNX Runtime 2フェーズ方式 (VRAM 節約型)
- **言語フィルタ** (`--lang en`) で英語+数字のみの認識に切り替え可能
- **argparse CLI** で統一的なインターフェース (`-i` で入力、`-o` で出力指定)

## セットアップ

```bash
# 依存インストール
uv sync

# ONNX モデルの配置
# models/ ディレクトリに以下のファイルを配置:
#   - PP-OCRv5_server_det.onnx        (テキスト検出)
#   - PP-OCRv5_server_rec.onnx        (テキスト認識)
#   - PP-LCNet_x1_0_textline_ori.onnx (テキスト行方向分類)
#   - rec_char_dict.txt                (多言語文字辞書)
#   - en_dict.txt                      (英語文字辞書, --lang en 用)
```

## 使い方

```bash
# ONNX SAHI エンジン (デフォルト) - 単一画像
uv run ppocr ocr -i <image_path>

# ONNX 2フェーズエンジン (VRAM 節約)
uv run ppocr ocr -i <image_path> --engine onnx-two-phase

# 英語+数字のみ認識
uv run ppocr ocr -i <image_path> --lang en

# ディレクトリ一括処理
uv run ppocr ocr -i <directory> -o <output_dir>

# テキストラベル非表示
uv run ppocr ocr -i <image_path> --no-text

# python -m での実行
uv run python -m ppocr_playground ocr -i <image_path>
```

## 出力パス規則

| 入力 | `-o` | 出力先 |
|------|------|--------|
| 単一画像 | 未指定 | `output/` |
| 単一画像 | `-o out/` | `out/` |
| ディレクトリ | 未指定 | `output/{dir_name}/` |
| ディレクトリ | `-o out/` | `out/{dir_name}/` |

ファイル名は `{stem}.json` + `{stem}.png` (可視化画像)。

## 言語フィルタ (`--lang en`)

`--lang en` は **server_rec モデルの出力を英語辞書でマスク** する方式で動作する。
専用の英語モデルは不要で、既存の server_rec + rec_char_dict.txt をそのまま使用する。

### 仕組み

1. `en_dict.txt` (94文字: 0-9, A-Z, a-z, ASCII 基本記号) を読み込み
2. `rec_char_dict.txt` の中で `en_dict.txt` に含まれない文字のスコアを `-inf` にマスク
3. CTC デコードの argmax 前にマスクを適用 → 英語+数字+記号のみが出力される

## プロジェクト構成

```
src/ppocr_playground/
├── __init__.py
├── __main__.py              # python -m ppocr_playground エントリポイント
├── cli.py                   # argparse CLI (ocr)
├── models.py                # Pydantic モデル (OcrResult, OcrTextItem)
├── io.py                    # JSON 保存
├── engine/
│   ├── __init__.py          # EngineType enum + create_engine() ファクトリ
│   ├── base.py              # OcrEngine ABC
│   ├── onnx_sahi.py         # OnnxSahiEngine (SAHI パッチ推論)
│   └── onnx_two_phase.py    # OnnxTwoPhaseEngine (VRAM 節約型)
├── onnx_ops/
│   ├── __init__.py          # 定数定義
│   ├── session.py           # セッション生成・文字辞書読み込み
│   ├── detection.py         # テキスト検出の前処理・後処理
│   ├── recognition.py       # テキスト認識の前処理・後処理・文字マスク
│   ├── classification.py    # テキスト行方向分類
│   ├── crop.py              # ボックスソート・回転クロップ
│   └── sahi.py              # SAHI タイル検出 + NMS
└── visualization/
    ├── __init__.py
    └── ocr.py               # OCR 結果の可視化
```

## アーキテクチャ

### OCR エンジン (Strategy パターン)

```
OcrEngine (ABC)
├── OnnxSahiEngine      - ONNX Runtime + SAHI パッチ推論
└── OnnxTwoPhaseEngine  - det→破棄→rec+cls→破棄 (VRAM 節約)
```

`create_engine(EngineType, **kwargs)` ファクトリで生成。
`lang="en"` を渡すと英語フィルタが有効になる。

## モデルファイル一覧

| ファイル | 用途 |
|---------|------|
| `PP-OCRv5_server_det.onnx` | テキスト検出 |
| `PP-OCRv5_server_rec.onnx` | テキスト認識 (多言語) |
| `PP-LCNet_x1_0_textline_ori.onnx` | テキスト行方向分類 |
| `rec_char_dict.txt` | 多言語文字辞書 |
| `en_dict.txt` | 英語文字辞書 (`--lang en` 用) |

## 環境

- Python >= 3.13
- NVIDIA GPU (CUDA)
- ONNX Runtime GPU 1.24+

## 開発

```bash
# lint + format
uv run ruff check src/ --fix && uv run ruff format src/
```
