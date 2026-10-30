"""画像・PDF・Excel を、ページ画像とテキスト層に読み込む。

どの形式も「ページ画像（グレースケール）」にそろえるので、位置合わせ・枠の指定・手書きマークの判定は
画像と同じ処理を使える。電子 PDF の文字と Excel のセル値は「テキスト層」として位置付きで返し、
枠の中にテキスト層の文字があれば、OCR の代わりにそれを読む（読み誤りが起きない）。

  画像（png / jpg / gif / bmp / tif） … 1 ページ。テキスト層なし
  PDF                                   … ページごとに画像にする。透明でない文字があればテキスト層にする
                                          （スキャナーが OCR で埋め込んだ透明な文字は使わない）
  Excel（xlsx / xlsm）                  … 開いたときに表示されるシートを、罫線とセル値を描いた画像にする

どれもこの PC 上で処理し、通信しない。
"""

from __future__ import annotations

import ctypes
import datetime as dt
import re
import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_common import load_image_gray

PDF_SUFFIXES = {".pdf"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"}
SUFFIXES = IMAGE_SUFFIXES | PDF_SUFFIXES | EXCEL_SUFFIXES

PDF_DPI = 200  # PDF をページ画像にするときの解像度（スキャンの標準的な解像度）

# (box (x0, y0, x1, y1), 文字列, 信頼度 0〜100)。ocr_common.page_lines と同じ形
Word = tuple[tuple[float, float, float, float], str, float]


@dataclass
class Page:
    gray: np.ndarray
    words: list[Word] | None = None  # テキスト層（gray の座標）。無ければ None

    def scaled(self, s: float) -> Page:
        import cv2

        gray = cv2.resize(self.gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        words = None if self.words is None else [(tuple(v * s for v in b), t, c) for b, t, c in self.words]
        return Page(gray, words)


def page_count(path: Path) -> int:
    """判定の単位になるページ数（PDF はページ数、それ以外は 1）。"""
    if Path(path).suffix.lower() in PDF_SUFFIXES:
        import pypdfium2 as pdfium

        return len(pdfium.PdfDocument(str(path)))
    return 1


def page_label(path: Path, page: int, count: int) -> str:
    return f"{Path(path).name}" if count <= 1 else f"{Path(path).name} ({page + 1}/{count} ページ)"


def load_source(path: Path, page: int = 0, *, width: int | None = None) -> Page:
    """ファイルの 1 ページを読み込む。width を指定すると、PDF はその幅（px）で画像にする。"""
    suffix = Path(path).suffix.lower()
    if suffix in PDF_SUFFIXES:
        return _load_pdf(Path(path), page, width)
    if suffix in EXCEL_SUFFIXES:
        return render_sheet(Path(path))
    return Page(load_image_gray(path))


# ---------------------------------------------------------------- PDF

def _load_pdf(path: Path, page: int, width: int | None) -> Page:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    if not 0 <= page < len(pdf):
        raise IndexError(f"{path.name} に {page + 1} ページ目はありません（{len(pdf)} ページ）")
    pg = pdf[page]
    w, _ = pg.get_size()
    scale = width / w if width else PDF_DPI / 72
    bitmap = pg.render(scale=scale)  # 白背景で描画される
    gray = np.array(bitmap.to_pil().convert("L"))
    words = _pdf_words(pg, gray.shape[1], gray.shape[0])
    return Page(gray, words or None)


def _pdf_words(pg, dev_w: int, dev_h: int) -> list[Word]:
    """テキスト層の文字を、行内で近い文字ごとにまとめて画像の座標で返す。

    使わない文字: 透明な文字（スキャナーの OCR が埋め込んだもの。読み誤りを含みうる）、
    フォントに文字コードの対応表が無く文字化けする文字、PDFium が補った空白・改行。
    """
    import pypdfium2.raw as raw

    tp = pg.get_textpage()
    n = tp.count_chars()
    dx, dy = ctypes.c_int(), ctypes.c_int()

    def to_dev(x: float, y: float) -> tuple[int, int]:
        # ページの回転（/Rotate）も含めて、描画した画像の座標に変換する
        raw.FPDF_PageToDevice(pg.raw, 0, 0, dev_w, dev_h, 0, x, y, dx, dy)
        return dx.value, dy.value

    chars = []  # (x0, y0, x1, y1, 文字) または区切り None
    for i in range(n):
        ch = chr(raw.FPDFText_GetUnicode(tp.raw, i))
        obj = raw.FPDFText_GetTextObject(tp.raw, i)
        if (
            raw.FPDFText_IsGenerated(tp.raw, i)
            or raw.FPDFText_HasUnicodeMapError(tp.raw, i)
            or (obj and raw.FPDFTextObj_GetTextRenderMode(obj) == raw.FPDF_TEXTRENDERMODE_INVISIBLE)
            or not ch.strip()
            or ch == "\ufffd"
        ):
            chars.append(None)
            continue
        left, bottom, right, top = tp.get_charbox(i)
        (ax, ay), (bx, by) = to_dev(left, top), to_dev(right, bottom)
        chars.append((min(ax, bx), min(ay, by), max(ax, bx), max(ay, by), ch))

    words: list[Word] = []
    cur: list[tuple] = []

    def flush() -> None:
        if cur:
            box = (min(c[0] for c in cur), min(c[1] for c in cur), max(c[2] for c in cur), max(c[3] for c in cur))
            words.append((tuple(float(v) for v in box), "".join(c[4] for c in cur), 100.0))
            cur.clear()

    for c in chars:
        if c is None:
            flush()
            continue
        if cur:
            p = cur[-1]
            h = max(p[3] - p[1], c[3] - c[1], 1)
            same_line = abs((p[1] + p[3]) / 2 - (c[1] + c[3]) / 2) < h * 0.5
            if not same_line or c[0] - p[2] > h * 0.8 or c[0] < p[0]:
                flush()  # 別の行、または離れた位置（別のセル）の文字
        cur.append(c)
    flush()
    return words


# ---------------------------------------------------------------- Excel

XL_SCALE = 2  # Excel の 1 px（100% 表示）を画像の何 px にするか
MAX_ROWS, MAX_COLS = 300, 80  # 描画するシートの範囲の上限
_FONTS = [  # 日本語を含むフォントの候補（Windows / macOS / Linux）
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def _font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for f in _FONTS:
        try:
            return ImageFont.truetype(f, px)
        except OSError:
            continue
    return ImageFont.load_default(px)  # 日本語は表示できない（セル値はテキスト層で読むので判定には影響しない）


def _sheet(path: Path):
    from openpyxl import load_workbook

    # data_only: 数式は Excel が最後に保存したときの計算結果を読む（保存されていなければ空欄）
    return load_workbook(path, data_only=True).active


def _grid(ws) -> tuple[list[int], list[int]]:
    """列の左端・行の上端の座標（画像の px、末尾は右端・下端）。非表示の行・列は幅 0。"""
    n_rows, n_cols = min(ws.max_row, MAX_ROWS), min(ws.max_column, MAX_COLS)
    sf = ws.sheet_format
    default_w = sf.defaultColWidth or (sf.baseColWidth or 8) + 0.43  # Excel の既定は 8.43
    default_h = sf.defaultRowHeight or 15
    widths = [float(default_w)] * (n_cols + 1)
    hidden_c = [False] * (n_cols + 1)
    for dim in ws.column_dimensions.values():
        lo, hi = dim.min or 0, dim.max or 0
        for c in range(max(lo, 1), min(hi, n_cols) + 1):
            if dim.width:
                widths[c] = dim.width
            hidden_c[c] = bool(dim.hidden)
    xs = [0]
    for c in range(1, n_cols + 1):
        px = 0 if hidden_c[c] else round(widths[c] * 7 + 5)  # Excel の列幅（文字数）→ 100% 表示の px
        xs.append(xs[-1] + px * XL_SCALE)
    ys = [0]
    for r in range(1, n_rows + 1):
        d = ws.row_dimensions.get(r)
        if d is not None and d.hidden:
            pt = 0
        elif d is not None and d.ht:
            pt = d.ht
        else:  # 高さの指定が無い行は、Excel と同じく行内で最も大きい文字に合わせる
            sizes = [c.font.sz for c in ws[r][:n_cols] if c.value is not None and c.font is not None and c.font.sz]
            pt = max([default_h] + [s * 1.33 for s in sizes])
        ys.append(ys[-1] + round(pt * 96 / 72) * XL_SCALE)
    return xs, ys


def _merged(ws) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """結合セルの左上 (行, 列) → (最小行, 最小列, 最大行, 最大列)。"""
    return {(m.min_row, m.min_col): (m.min_row, m.min_col, m.max_row, m.max_col) for m in ws.merged_cells.ranges}


def cell_boxes(path: Path) -> dict[str, tuple[int, int, int, int]]:
    """セル番地（"B3" など。結合セルは左上の番地）→ 描画した画像上の枠。"""
    ws = _sheet(Path(path))
    xs, ys = _grid(ws)
    return {addr: box for addr, box, _ in _cells(ws, xs, ys)}


def _cells(ws, xs, ys):
    """(番地, 枠, セル) を描画範囲内のセルについて返す。結合セルは左上だけを結合範囲の枠で返す。"""
    merged = _merged(ws)
    covered = {
        (r, c)
        for (r0, c0, r1, c1) in merged.values()
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) != (r0, c0)
    }
    n_rows, n_cols = len(ys) - 1, len(xs) - 1
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            if (r, c) in covered:
                continue
            r1, c1 = merged.get((r, c), (r, c, r, c))[2:]
            box = (xs[c - 1], ys[r - 1], xs[min(c1, n_cols)], ys[min(r1, n_rows)])
            if box[2] > box[0] and box[3] > box[1]:
                cell = ws.cell(r, c)
                yield cell.coordinate, box, cell


def _gray_of(color) -> int | None:
    """openpyxl の色（ARGB の rgb 指定のみ）を輝度にする。テーマ色などは扱わない。"""
    try:
        rgb = color.rgb if color is not None and color.type == "rgb" else None
    except AttributeError:
        return None
    if not isinstance(rgb, str) or len(rgb) < 6:
        return None
    r, g, b = (int(rgb[-6:][i : i + 2], 16) for i in (0, 2, 4))
    return int(0.299 * r + 0.587 * g + 0.114 * b)


_BORDER_PX = {"thin": 1, "hair": 1, "dotted": 1, "dashed": 1, "medium": 2, "mediumDashed": 2, "double": 3, "thick": 3}


def render_sheet(path: Path) -> Page:
    """シートを、罫線・塗りつぶし・セル値を描いた画像にし、セル値をテキスト層にする。

    図形・画像・グラフ・条件付き書式は描かない。
    """
    ws = _sheet(path)
    xs, ys = _grid(ws)
    w, h = max(xs[-1], 1), max(ys[-1], 1)
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    cells = list(_cells(ws, xs, ys))

    for _, (x0, y0, x1, y1), cell in cells:  # 塗りつぶしと、薄い枠線（Excel の目盛線）
        fill = _gray_of(cell.fill.fgColor) if cell.fill is not None and cell.fill.patternType == "solid" else None
        draw.rectangle((x0, y0, x1, y1), fill=fill if fill is not None else 255, outline=225)
    for _, (x0, y0, x1, y1), cell in cells:  # 罫線
        b = cell.border
        for side, line in (
            (b.left, (x0, y0, x0, y1)), (b.right, (x1, y0, x1, y1)),
            (b.top, (x0, y0, x1, y0)), (b.bottom, (x0, y1, x1, y1)),
        ):
            if side is not None and side.style:
                draw.line(line, fill=0, width=_BORDER_PX.get(side.style, 1) * XL_SCALE)

    words: list[Word] = []
    for _, (x0, y0, x1, y1), cell in cells:  # 値
        text = format_value(cell.value, cell.number_format)
        if not text:
            continue
        size = cell.font.sz if cell.font is not None and cell.font.sz else 11
        font = _font(max(round(size * 96 / 72 * XL_SCALE), 6))
        lines = text.splitlines() or [text]
        body = "\n".join(lines)
        l, t, r, b = draw.multiline_textbbox((0, 0), body, font=font)
        tw, th = r - l, b - t
        al = cell.alignment
        horiz = al.horizontal if al is not None else None
        if horiz in (None, "general"):
            horiz = "right" if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool) else "left"
        pad = 3 * XL_SCALE
        tx = {"center": (x0 + x1 - tw) / 2, "centerContinuous": (x0 + x1 - tw) / 2, "right": x1 - pad - tw}.get(horiz, x0 + pad)
        vert = al.vertical if al is not None else None
        ty = {"top": y0 + pad, "center": (y0 + y1 - th) / 2}.get(vert, y1 - pad - th)
        draw.multiline_text((tx - l, ty - t), body, fill=0, font=font)
        box = (max(tx, x0), max(ty, y0), min(tx + tw, x1), min(ty + th, y1))  # 枠からはみ出た部分は含めない
        words.append((tuple(float(v) for v in box), "".join(lines), 100.0))
    return Page(np.array(img), words or None)


# ---------------------------------------------------------------- Excel の表示形式

_SECTION_SPLIT = re.compile(r';(?=(?:[^"]*"[^"]*")*[^"]*$)')  # 引用符の外の ; で区切る


def _general(v: float) -> str:
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    s = repr(round(float(v), 10))
    return s


def format_value(value, fmt: str | None) -> str:
    """セル値を、Excel の表示形式（よく使うもの）に従って画面に表示される文字列にする。

    小数桁数の判定は表示された文字列で行うため、"0.000" の書式なら 1 は "1.000" になる。
    対応しない書式（指数・分数など）は General（そのままの値）で表示する。
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        return value
    if isinstance(value, dt.datetime):
        return value.strftime("%Y/%m/%d %H:%M") if (value.hour, value.minute) != (0, 0) else value.strftime("%Y/%m/%d")
    if isinstance(value, dt.date):
        return value.strftime("%Y/%m/%d")
    if isinstance(value, dt.time):
        return value.strftime("%H:%M")
    if not isinstance(value, (int, float)):
        return str(value)

    fmt = fmt or "General"
    sections = _SECTION_SPLIT.split(fmt)
    if value < 0 and len(sections) >= 2:
        sec, v, sign = sections[1], -value, ""  # 負の書式には符号を自分で書く（例: +0;-0;0）
    elif value == 0 and len(sections) >= 3:
        sec, v, sign = sections[2], value, ""
    else:
        sec, v, sign = sections[0], abs(value), "-" if value < 0 else ""
    return _format_section(sec, v, sign)


def _format_section(sec: str, v: float, sign: str) -> str:
    # [Red] や [>=100] などの角かっこ、_x（空白）、*x（繰り返し）を除き、"…" と \x は文字としてそのまま出す
    tokens: list[tuple[str, str]] = []  # ("lit", 文字) / ("ph", 書式記号)
    i = 0
    while i < len(sec):
        ch = sec[i]
        if ch == "[":
            j = sec.find("]", i)
            i = len(sec) if j < 0 else j + 1
        elif ch == '"':
            j = sec.find('"', i + 1)
            j = len(sec) if j < 0 else j
            tokens.append(("lit", sec[i + 1 : j]))
            i = j + 1
        elif ch == "\\" and i + 1 < len(sec):
            tokens.append(("lit", sec[i + 1]))
            i += 2
        elif ch in "_*" and i + 1 < len(sec):
            i += 2
        elif ch in "0#?.,%":
            tokens.append(("ph", ch))
            i += 1
        else:
            tokens.append(("lit", ch))
            i += 1

    if "general" in sec.lower() or sec.strip() == "@":
        return sign + re.sub("general|@", _general(v), sec, flags=re.IGNORECASE).replace('"', "")
    if not any(k == "ph" and t in "0#?" for k, t in tokens):
        return "".join(t for k, t in tokens if k == "lit")  # 数値の記号が無い（文字だけ・@）の書式

    first = next(i for i, (k, t) in enumerate(tokens) if k == "ph" and t in "0#?")
    last = max(i for i, (k, t) in enumerate(tokens) if k == "ph")
    prefix = "".join(t for k, t in tokens[:first] if k == "lit")
    suffix = "".join(t for k, t in tokens[last + 1 :] if k == "lit")
    pattern = "".join(t for k, t in tokens[first : last + 1] if k == "ph")

    if "%" in pattern:
        v *= 100
        pattern = pattern.replace("%", "")
        suffix = "%" + suffix
    int_part, _, dec_part = pattern.partition(".")
    max_dec = sum(1 for c in dec_part if c in "0#?")
    min_dec = dec_part.count("0")  # 0 は必ず表示する桁、# と ? は 0 なら省く桁
    min_int = int_part.count("0")

    q = Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-max_dec), rounding=ROUND_HALF_UP)
    ip, _, dp = f"{q:f}".partition(".")
    dp = dp[:min_dec] + dp[min_dec:].rstrip("0")
    ip = ip.lstrip("0").rjust(min_int, "0")
    if "," in int_part and ip:
        ip = f"{int(ip):,}"
    num = ip + ("." + dp if dp else "")
    if not any(ch in "123456789" for ch in num):
        sign = ""  # 丸めて 0 になった負の値は "-0" と表示しない
    return prefix + sign + num + suffix


if __name__ == "__main__":  # 動作確認用: uv run document_input.py ファイル
    for a in sys.argv[1:]:
        for pno in range(page_count(Path(a))):
            pg = load_source(Path(a), pno)
            print(page_label(Path(a), pno, page_count(Path(a))), pg.gray.shape, len(pg.words or []), "words")
            for wd in (pg.words or [])[:20]:
                print("   ", [round(v) for v in wd[0]], wd[1])
