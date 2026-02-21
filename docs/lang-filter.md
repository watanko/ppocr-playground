# 言語フィルタ (`--lang en`) 実装メモ

## 概要

server_rec (多言語) モデルの CTC 出力に対して、英語辞書に含まれない文字のスコアを `-inf` にマスクすることで、追加モデルなしで英語+数字のみの認識を実現する。

## 背景

- PaddleOCR v5 の英語専用 rec モデルは `en_PP-OCRv5_mobile_rec` (mobile) のみで server 版は存在しない
- server_rec の出力次元 (18385) は rec_char_dict.txt (18383文字 + blank) に紐づいており、辞書だけ差し替えるとインデックスがずれる
- → argmax 前にマスクを適用する方式を採用

## 変更ファイル

### 新規ファイル

| ファイル | 内容 |
|---------|------|
| `models/en_dict.txt` | 英語文字辞書 (94文字). ASCII 印字可能文字のみ |

### 変更ファイル

#### `onnx_ops/recognition.py`

- `build_char_mask(char_dict, allowed_chars)` を追加
  - `char_dict` (18384要素) に対して `allowed_chars` に含まれない文字を `-inf` にしたマスクベクトルを返す
  - blank (index=0) は常に許可
- `postprocess_rec()` に `char_mask` 引数を追加 (デフォルト `None`)
  - `None` の場合は従来通り全文字を使用
  - マスク指定時は `pred + mask` してから argmax
  - モデル出力次元 (18385) と辞書サイズ (18384) のずれは末尾を `-inf` パディングで吸収

#### `onnx_ops/session.py`

- `load_allowed_chars(dict_path)` を追加
  - 辞書ファイルから `set[str]` を返す

#### `engine/onnx_sahi.py`

- `__init__()` に `lang: str | None = None` 引数を追加
- `_ensure_loaded()` で `lang == "en"` の場合に `en_dict.txt` を読み込み `build_char_mask()` でマスクを生成
- `run()` 内の `postprocess_rec()` 呼び出しに `self._char_mask` を渡す

#### `engine/onnx_two_phase.py`

- `__init__()` に `lang: str | None = None` 引数を追加
- コンストラクタで `lang == "en"` の場合にマスクを生成
- `run()` / `run_batch()` 内の `postprocess_rec()` 呼び出しに `self.char_mask` を渡す

#### `cli.py`

- ocr サブコマンドに `--lang {en}` オプションを追加
- `create_engine()` に `lang` を渡す

#### `export_onnx.py`

- `_MODELS` リストに `en_PP-OCRv5_mobile_rec` を追加
- `extract_char_dict()` を追加: inference.yml から文字辞書を YAML パースして txt に保存
- `run_export()` の末尾で `extract_char_dict("en_PP-OCRv5_mobile_rec", "en_dict.txt")` を実行

## データフロー

```
[エンジン初期化]
  en_dict.txt → load_allowed_chars() → allowed_chars: set[str]
  rec_char_dict.txt → load_character_dict() → char_dict: list[str]  (18384要素)
  build_char_mask(char_dict, allowed_chars) → char_mask: ndarray (18384,)

[推論時]
  rec_session.run() → pred: ndarray (N, T, 18385)
  postprocess_rec(pred, char_dict, char_mask):
    pred[:, :, :] += char_mask[np.newaxis, np.newaxis, :]  # 非許可文字を -inf に
    argmax → CTC decode → 英語テキストのみ出力
```

## en_dict.txt の内容 (94文字)

ASCII 印字可能文字のみ:

- 数字: `0-9` (10文字)
- 英大文字: `A-Z` (26文字)
- 英小文字: `a-z` (26文字)
- 基本記号: `` !"#$%&'()*+,-./:;<=>?@[\]_`{|}^~ `` (32文字)
