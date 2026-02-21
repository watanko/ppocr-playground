# ppocr-playground

PaddleOCR v5 / ONNX Runtime を用いた OCR・文書構造解析・テーブル認識ツールキット。

## 特徴

- **3つの OCR エンジン** を Strategy パターンで切り替え可能
  - `paddle` - PaddleOCR (PaddlePaddle GPU)
  - `onnx-sahi` - ONNX Runtime + SAHI パッチ推論 (大画像対応)
  - `onnx-two-phase` - ONNX Runtime 2フェーズ方式 (VRAM 節約型)
- **言語フィルタ** (`--lang en`) で英語+数字のみの認識に切り替え可能
- **文書構造解析** (PaddlePaddle / ONNX Runtime)
- **テーブル認識** (PP-StructureV3)
- **argparse CLI** で統一的なインターフェース (`-i` で入力、`-o` で出力指定)

## セットアップ

```bash
# 依存インストール
uv sync

# ONNX モデルの配置
# models/ ディレクトリに以下のファイルを配置:
#   - PP-OCRv5_server_det.onnx      (テキスト検出)
#   - PP-OCRv5_server_rec.onnx      (テキスト認識)
#   - PP-LCNet_x1_0_textline_ori.onnx (テキスト行方向分類)
#   - rec_char_dict.txt              (多言語文字辞書, 18383文字)
#   - en_dict.txt                    (英語文字辞書, 436文字, --lang en 用)
#   - PP-DocLayout_plus-L.onnx       (構造解析用)
#   - PP-LCNet_x1_0_table_cls.onnx   (テーブル分類用, 任意)
#   - RT-DETR-L_wired_table_cell_det.onnx   (テーブルセル検出用, 任意)
#   - RT-DETR-L_wireless_table_cell_det.onnx (テーブルセル検出用, 任意)
```

## 使い方

### OCR

```bash
# PaddleOCR エンジン (デフォルト) - 単一画像
uv run ppocr ocr -i <image_path>

# ONNX SAHI エンジン
uv run ppocr ocr -i <image_path> --engine onnx-sahi

# ONNX 2フェーズエンジン (VRAM 節約)
uv run ppocr ocr -i <image_path> --engine onnx-two-phase

# 英語+数字のみ認識 (ONNX エンジンで使用可能)
uv run ppocr ocr -i <image_path> --engine onnx-sahi --lang en

# ディレクトリ一括処理 (glob で画像を自動収集)
uv run ppocr ocr -i <directory> -o <output_dir>

# テキストラベル非表示
uv run ppocr ocr -i <image_path> --no-text
```

### 文書構造解析

```bash
# PaddlePaddle バックエンド (デフォルト)
uv run ppocr structure -i <image_path>

# ONNX バックエンド
uv run ppocr structure -i <image_path> --backend onnx

# ディレクトリ一括処理
uv run ppocr structure -i <directory> -o <output_dir> --backend onnx
```

### テーブル認識

```bash
uv run ppocr table -i <image_path>

# ディレクトリ一括処理
uv run ppocr table -i <directory> -o <output_dir>
```

### ONNX エクスポート

```bash
uv run ppocr export-onnx
```

### python -m での実行

```bash
uv run python -m ppocr_playground ocr -i <image_path>
uv run python -m ppocr_playground structure -i <image_path> --backend onnx
```

## 出力パス規則

| 入力 | `-o` | 出力先 |
|------|------|--------|
| 単一画像 | 未指定 | 入力画像と同じディレクトリ |
| 単一画像 | `-o out/` | `out/` |
| ディレクトリ | 未指定 | 入力の親ディレクトリ / `{dir_name}/` |
| ディレクトリ | `-o out/` | `out/{dir_name}/` |

ファイル名は常に `{stem}.json` + `{stem}.png`。
テーブル認識では追加で `{stem}.csv` + `{stem}.md` も出力。

## 言語フィルタ (`--lang en`)

`--lang en` は **server_rec モデルの出力を英語辞書でマスク** する方式で動作する。
専用の英語モデルは不要で、既存の server_rec + rec_char_dict.txt をそのまま使用する。

### 仕組み

1. `en_dict.txt` (94文字: 0-9, A-Z, a-z, ASCII 基本記号) を読み込み
2. `rec_char_dict.txt` (18383文字) の中で `en_dict.txt` に含まれない文字のスコアを `-inf` にマスク
3. CTC デコードの argmax 前にマスクを適用 → 英語+数字+記号のみが出力される

### 比較例

| モード | テキスト例 | 備考 |
|--------|-----------|------|
| 通常 | `内部仕上表54` | 日本語+数字 |
| `--lang en` | `54` | 数字のみ残る |
| 通常 | `大阪府立成人病センター整備事業` | 日本語 |
| `--lang en` | (低スコアで実質除去) | マスクにより無効化 |
| 通常 | `ID:4002687-20161221-141343` | 英数字 |
| `--lang en` | `ID:4002687-20161221-141343` | そのまま |

## プロジェクト構成

```
src/ppocr_playground/
├── __init__.py
├── __main__.py              # python -m ppocr_playground エントリポイント
├── cli.py                   # argparse CLI (ocr / structure / table / export-onnx)
├── models.py                # 全 Pydantic モデル
├── io.py                    # JSON 保存ユーティリティ
├── export_onnx.py           # ONNX エクスポート + 辞書抽出
├── engine/
│   ├── __init__.py          # EngineType enum + create_engine() ファクトリ
│   ├── base.py              # OcrEngine ABC
│   ├── paddle.py            # PaddleOcrEngine
│   ├── onnx_sahi.py         # OnnxSahiEngine (SAHI パッチ推論, lang対応)
│   └── onnx_two_phase.py    # OnnxTwoPhaseEngine (VRAM 節約型, lang対応)
├── onnx_ops/
│   ├── __init__.py          # 定数定義
│   ├── session.py           # セッション生成・文字辞書読み込み・許可文字読み込み
│   ├── detection.py         # テキスト検出の前処理・後処理
│   ├── recognition.py       # テキスト認識の前処理・後処理・文字マスク
│   ├── classification.py    # テキスト行方向分類
│   ├── crop.py              # ボックスソート・回転クロップ
│   ├── sahi.py              # SAHI タイル検出 + NMS
│   └── detr.py              # DETR 前処理・後処理 + テーブル分類
├── pipeline/
│   ├── __init__.py
│   ├── structure_paddle.py  # 構造解析 (PaddlePaddle 版)
│   ├── structure_onnx.py    # 構造解析 (ONNX Runtime 版)
│   └── table.py             # テーブル認識 + HTML パース + CSV/Markdown 変換
└── visualization/
    ├── __init__.py
    ├── ocr.py               # OCR 結果の可視化
    ├── structure.py          # 構造解析結果の可視化
    └── table.py             # テーブル認識結果の可視化
```

## アーキテクチャ

### OCR エンジン (Strategy パターン)

```
OcrEngine (ABC)
├── PaddleOcrEngine     - PaddleOCR を直接使用
├── OnnxSahiEngine      - ONNX Runtime + SAHI パッチ推論 (lang対応)
└── OnnxTwoPhaseEngine  - det→破棄→rec+cls→破棄 (VRAM 節約, lang対応)
```

`create_engine(EngineType, **kwargs)` ファクトリで生成。
`lang="en"` を渡すと英語フィルタが有効になる。

### 文字マスクの流れ

```
en_dict.txt (94文字)
    ↓ load_allowed_chars()
allowed_chars: set[str]
    ↓ build_char_mask(char_dict, allowed_chars)
char_mask: np.ndarray  shape=(18384,)  許可=0, 非許可=-inf
    ↓ postprocess_rec(pred, char_dict, char_mask)
pred + mask → argmax → CTC decode → 英語テキストのみ
```

### VRAM 使用量の比較

| 構成 | ピーク VRAM | 推論速度 |
|------|-----------|---------|
| PaddleOCR | 1,576 MiB | 0.218秒/枚 |
| ONNX SAHI | 1,964 MiB | 1.019秒/枚 |
| ONNX 2フェーズ | 1,588 MiB | 0.728秒/枚 |

詳細は [docs/benchmark-paddleocr-vs-onnx.md](docs/benchmark-paddleocr-vs-onnx.md) を参照。

## モデルファイル一覧

| ファイル | サイズ | 用途 |
|---------|--------|------|
| `PP-OCRv5_server_det.onnx` | ~100 MB | テキスト検出 |
| `PP-OCRv5_server_rec.onnx` | ~80 MB | テキスト認識 (多言語) |
| `PP-LCNet_x1_0_textline_ori.onnx` | ~5 MB | テキスト行方向分類 |
| `rec_char_dict.txt` | ~200 KB | 多言語文字辞書 (18383文字) |
| `en_dict.txt` | ~0.2 KB | 英語文字辞書 (94文字, ASCII のみ) |
| `PP-DocLayout_plus-L.onnx` | ~100 MB | 文書レイアウト検出 |
| `PP-LCNet_x1_0_table_cls.onnx` | ~5 MB | テーブル有線/無線分類 (任意) |
| `RT-DETR-L_wired_table_cell_det.onnx` | ~120 MB | 有線テーブルセル検出 (任意) |
| `RT-DETR-L_wireless_table_cell_det.onnx` | ~120 MB | 無線テーブルセル検出 (任意) |

## 環境

- Python >= 3.13
- NVIDIA GPU (CUDA)
- PaddlePaddle GPU 3.2.x
- ONNX Runtime GPU 1.24+

## 開発

```bash
# lint + format
uv run ruff check src/ --fix && uv run ruff format src/
```
