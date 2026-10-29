# 検査成績書 OCR 良否判定

検査成績書の画像から計測値を OCR で読み取り、規格に対する良否を判定するツールです。

検査成績書は社外秘のため、クラウドの AI サービスは使わず、**すべてこの PC 上で処理**します。画像が外部に送信されることはありません。

## できること

- 計測表（長さ・幅・高さ・表面磁束密度 × No.1〜20）、検査規格、最大値/最小値/平均値、c/n、ロットNo.、磁気特性値（Br / Hcb / Hcj / BHmax）を読み取る
- 各計測値が規格内か判定する（OK / NG）
- OCR の読み誤りの疑いがある値を「要確認」として知らせる
- 総合判定（合格 / 不合格 / 要目視確認）を表示し、終了コードでも返す

対象は `sample/inspection_sample.gif` と同じ様式（NeoMag の検査成績書）のみです。

## セットアップ

### 1. uv

Python 環境は [uv](https://docs.astral.sh/uv/) で管理しています。

```powershell
uv sync
```

`.venv` が作られ、必要なパッケージがすべて入ります。

### 2. OCR エンジン

2 種類から選べます。どちらもローカルで動きます。

| エンジン | 準備 |
| --- | --- |
| Tesseract（既定） | 本体を別途インストール: `winget install UB-Mannheim.TesseractOCR` |
| indra-ocr | `uv sync` で導入済み。初回実行時にモデル（約 76MB）を Hugging Face からダウンロードし、以降はキャッシュを使うため通信しない |

ネットワークにつながらない PC で indra-ocr を使う場合は、別の PC でダウンロードしたモデルフォルダをコピーし、`config.yaml` の `indra_model_dir` に指定してください。

## 使い方

```powershell
uv run inspection_ocr.py sample/inspection_sample.gif
uv run inspection_ocr.py sample/inspection_sample.gif --engine indra   # indra-ocr で読む
uv run inspection_ocr.py scans/a.gif scans/b.gif                       # 複数ファイル
uv run inspection_ocr.py scans/a.gif --debug                           # セル切り出し位置を debug/ に画像出力
```

> `python inspection_ocr.py ...` で実行するとシステムの Python が使われ、indra-ocr が見つかりません。`uv run` を使うか、先に `.venv\Scripts\Activate.ps1` で `.venv` を有効にしてください。

| オプション | 内容 |
| --- | --- |
| `--engine tesseract\|indra` | OCR エンジン（省略時は `config.yaml` の `engine`） |
| `--config パス` | 設定ファイル（省略時は `config.yaml`） |
| `--debug` | 読み取ったセルの位置を赤枠で描いた画像を `debug/` に保存 |

### 出力例

```text
ファイル : sample\inspection_sample.gif  (OCR: indra)
ロットNo.: NN14092301

磁気特性値: Br=10.44  Hcb=9.92  Hcj=22.49  BHmax=26.05

     No.      長さ(mm)        幅(mm)      高さ(mm)         Gauss
    規格     19.9-20.1      9.9-10.1     1.95-2.05     1200-1400
----------------------------------------------------------------
       1         20.02         10.03          1.99          1310
       ...
     c/n          0/20          0/20          0/20          0/20
(NG=規格外, ?=要確認)

■ 総合判定: 合格
```

NG や要確認があると、表の該当値に `NG` / `?` が付き、「■ 指摘事項」に理由が一覧表示されます。

```text
■ 指摘事項
  [NG] 高さ(mm) No.5: 1.96 - 規格外 (1.97〜2.03)
  [要確認] 高さ(mm): 帳票の c/n 0/20 ≠ 判定結果 3/20

■ 総合判定: 要目視確認
```

### 終了コード

| コード | 意味 |
| --- | --- |
| 0 | 合格 |
| 1 | 不合格（NG あり、要確認なし） |
| 2 | 要目視確認（要確認が 1 件でもある） |
| 3 | 読み取りエラー（表の構造を検出できない、エンジンが見つからない） |

複数ファイルを指定した場合は、最も悪い結果のコードを返します。

## 設定（config.yaml）

```yaml
engine: tesseract          # tesseract / indra
tesseract_cmd: null        # null なら自動検出
indra_model_dir: null      # null なら初回ダウンロード＆キャッシュ
min_confidence: 60         # これ未満の OCR 信頼度は要確認

columns:
  length: {decimals: 2, spec: from_document}
  width:  {decimals: 2, spec: from_document}
  height: {decimals: 2, spec: from_document}
  gauss:  {decimals: 0, spec: {min: 1200, max: 1400}}

magnetic: {}               # 例: Br: {min: 10.0, max: 11.0}
```

- `spec: from_document` は帳票の「検査規格」欄を読み取って使います。`{min: .., max: ..}` を書くとそちらが優先されます。
- 境界値（下限・上限ちょうど）は合格扱いです。
- **Gauss の規格は帳票に「N極中心」としか書かれていないため、`config.yaml` の値（初期値 1200〜1400 は仮の値）を実際の規格に設定してください。**
- 磁気特性値は規格を設定した項目だけ判定し、未設定の項目は読み取り結果の表示のみです。

## 判定の仕組み

1. **表構造の検出** — 罫線を検出し、3 か所の二重罫線（「測定値」行の下 / No.20 の下 / c/n の下）を基準に各行・各列の位置を求めます。スキャン時の位置ずれには追従しますが、様式が変わると検出できません。
2. **OCR** — Tesseract はセルごとに、indra-ocr はページ全体を 1 回読んで各行をセルに割り当てます。
3. **良否判定** — 規格の範囲内なら OK、外れたら NG。
4. **読み誤りの検出** — 次の場合はその値を「要確認」にします。
   - 数字以外の文字が混じっている、小数桁数が設定と違う
   - OCR の信頼度が `min_confidence` 未満
   - indra-ocr で、1 行が隣のセルまでつながって読まれた
   - 読み取った 20 値から再計算した最大値・最小値・平均値が、帳票の値と合わない
   - 帳票の c/n と、判定した NG 件数が合わない

要確認が 1 件でもあれば、総合判定は「要目視確認」になります。読み誤りを数字に当てはめて隠さないよう、OCR の文字種はあえて制限していません。

## エンジンの比較

サンプル画像と、それを人工的に劣化させた画像での正読数（全 80 値中）です。

| 画像 | Tesseract | indra-ocr |
| --- | --- | --- |
| 元画像 | 80 | 80 |
| ノイズ | 38 | 76 |
| JPEG 高圧縮 | 56 | 78 |
| ぼかし | 2 | 80 |
| 0.8 倍に縮小 | 11 | 80 |
| 0.3° 傾き | 65 | 80 |

処理時間は 1 枚あたり、Tesseract が 10〜20 秒、indra-ocr が約 0.5 秒（初回はモデル読み込みで約 2 秒）でした。どの条件でも、誤読があるのに「合格」と出たケースはありませんでした。

## 注意事項

- **indra-ocr のライセンス**: コードは Apache 2.0 ですが、モデルの重みは独自条件付きの OpenRAIL-M です。無償利用は研究・個人利用と、資金調達額または売上が 2,000 万ドル未満の企業に限られます。業務で使う前に、[ライセンス条件](https://huggingface.co/datalab-to/indra-ocr/blob/main/LICENSE)を確認してください。
- **見逃しの可能性**: 形式が正しく規格内に収まる誤読（例: `1310` → `1330`）は、最大値・最小値に影響しない限り、平均値との照合だけでは見つけにくいことがあります。重要な判定では、要目視確認になったものだけでなく、定期的に原本との突き合わせを行ってください。
- `--debug` で出力した画像は検査成績書の内容を含みます。不要になったら `debug/` を削除してください。

## テスト

```powershell
uv run pytest tests
```

サンプル画像を両エンジンで読み、全値が正解と一致すること、規格外の値を NG にすること、読み誤りを要確認にすることを確認します（Tesseract 未導入の場合、Tesseract のテストはスキップされます）。

## ファイル構成

```text
inspection_ocr.py     本体
config.yaml           規格・エンジン等の設定
pyproject.toml        依存パッケージ（uv）
uv.lock
tests/test_sample.py  テスト
sample/               サンプル画像
```
