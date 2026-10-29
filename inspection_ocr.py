"""検査成績書の計測値をローカル OCR で読み取り、良否判定する。

OCR エンジンは Tesseract か indra-ocr（ONNX Runtime、CPU）を選べる。
どちらも画像は外部に送信せず、すべてこの PC 上で処理する
（indra-ocr は初回のみモデルを Hugging Face からダウンロードする）。

使い方:
    uv run inspection_ocr.py sample/inspection_sample.gif [--engine indra] [--config config.yaml] [--debug]

終了コード: 0=合格, 1=不合格, 2=要目視確認, 3=読み取りエラー
"""

from __future__ import annotations

import argparse
import functools
import os
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import yaml
from PIL import Image

SCALE = 4  # 低解像度の帳票を拡大してから OCR する倍率
N_DATA_ROWS = 20

COLUMNS = ["length", "width", "height", "gauss"]
COLUMN_LABELS = {
    "length": "長さ(mm)",
    "width": "幅(mm)",
    "height": "高さ(mm)",
    "gauss": "Gauss",
}
MAGNETIC = ["Br", "Hcb", "Hcj", "BHmax"]
SUMMARY_ROWS = ["max", "min", "ave", "cn"]
SUMMARY_LABELS = {"max": "最大値", "min": "最小値", "ave": "平均値", "cn": "c/n"}

OK, NG, CHECK = "OK", "NG", "要確認"


class LayoutError(Exception):
    pass


# ---------------------------------------------------------------- 設定

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_tesseract(cfg: dict) -> str:
    candidates = [
        cfg.get("tesseract_cmd"),
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return str(c)
    raise FileNotFoundError(
        "Tesseract が見つかりません。config.yaml の tesseract_cmd にパスを設定してください。"
    )


# ---------------------------------------------------------------- 表構造の検出

def load_gray(path: Path) -> np.ndarray:
    # cv2.imread は GIF 非対応のため Pillow で読む
    gray = np.array(Image.open(path).convert("L"))
    return cv2.resize(gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)


def _cluster(indices: np.ndarray, gap: int) -> list[int]:
    """隣接するインデックスをまとめ、各まとまりの中心を返す。"""
    groups: list[list[int]] = []
    for i in indices:
        if groups and i - groups[-1][-1] <= gap:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return [int(np.mean(g)) for g in groups]


def line_masks(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gray.shape[1] // 8, 1))
    # 文字の縦画（約 10px）より長く、1 行の高さ（約 13px）より短いカーネル
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11 * SCALE))
    h_mask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    v_mask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
    return h_mask, v_mask


def horizontal_lines(h_mask: np.ndarray) -> list[int]:
    coverage = h_mask.sum(axis=1) / 255
    return _cluster(np.where(coverage > h_mask.shape[1] * 0.3)[0], 2 * SCALE)


def vertical_lines(v_mask: np.ndarray, y0: int, y1: int) -> list[int]:
    """y0〜y1 の帯をほぼ貫通している縦罫線の x 座標。"""
    pad = 3 * SCALE
    band = v_mask[y0 + pad : y1 - pad]
    coverage = band.mean(axis=0) / 255
    return _cluster(np.where(coverage > 0.8)[0], 2 * SCALE)


@dataclass
class Layout:
    lot_row: tuple[int, int]
    magnetic_row: tuple[int, int]
    spec_row: tuple[int, int]
    data_rows: list[tuple[int, int]]
    summary_rows: list[tuple[int, int]]


def locate_layout(h_lines: list[int]) -> Layout:
    """二重罫線を基準に各行の位置を特定する（同一様式前提）。

    二重罫線は 3 か所: 「測定値」行の下 / No.20 の下 / c/n の下。
    """
    doubles = [i for i in range(len(h_lines) - 1) if h_lines[i + 1] - h_lines[i] < 10 * SCALE]
    if len(doubles) < 3:
        raise LayoutError(f"二重罫線が 3 本見つかりません（検出 {len(doubles)} 本）")
    m, d = doubles[0], doubles[1]
    if d - N_DATA_ROWS - 1 < m + 1 or d + 5 >= len(h_lines) or m < 5:
        raise LayoutError("罫線の本数が想定と異なります")

    def row(i: int) -> tuple[int, int]:
        return h_lines[i], h_lines[i + 1]

    data_rows = [row(i) for i in range(d - N_DATA_ROWS, d)]
    heights = np.array([b - a for a, b in data_rows])
    if np.any(np.abs(heights - np.median(heights)) > np.median(heights) * 0.3):
        raise LayoutError("計測データ行の高さが揃っていません")
    return Layout(
        lot_row=row(m - 5),
        magnetic_row=row(m - 1),
        spec_row=row(d - N_DATA_ROWS - 1),
        data_rows=data_rows,
        summary_rows=[row(d + 1 + i) for i in range(len(SUMMARY_ROWS))],
    )


# ---------------------------------------------------------------- OCR

@dataclass
class Read:
    text: str
    conf: float  # 0〜100、文字が無ければ -1
    note: str = ""  # 読み取り時点で分かった問題（あれば要確認にする）


def ocr_cell(gray: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> Read:
    m = 3 * SCALE  # 罫線を含めないよう内側に切り出す
    crop = gray[y0 + m : y1 - m, x0 + m : x1 - m]
    crop = cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    # 二値化や文字種ホワイトリストは使わない。低解像度の帳票では二値化で字形が崩れ、
    # ホワイトリストは誤読を数字に当てはめて隠してしまう（信頼度も無意味になる）。
    # 数字以外が読めた場合は後段の書式チェックで「要確認」にする。
    data = pytesseract.image_to_data(
        crop, config="--psm 7 --oem 1", output_type=pytesseract.Output.DICT
    )
    words = [(t.strip(), float(c)) for t, c in zip(data["text"], data["conf"]) if t.strip()]
    if not words:
        return Read("", -1)
    return Read("".join(t for t, _ in words), min(c for _, c in words))


@functools.cache
def _indra_converter(model_dir: str | None):
    try:
        import indra_ocr  # Tesseract 使用時は読み込まない
    except ImportError as e:
        print(
            "indra-ocr が見つかりません。uv の環境で実行してください:\n"
            "    uv run inspection_ocr.py <画像> --engine indra",
            file=sys.stderr,
        )
        raise SystemExit(3) from e  # 1 は「不合格」と紛らわしいので読み取りエラーの 3

    return indra_ocr.load(path=model_dir, mode="ocr")


def indra_lines(path: Path, model_dir: str | None) -> list[tuple[tuple[float, ...], str, float]]:
    """ページ全体を indra-ocr で読み、(拡大後座標の box, 文字列, 信頼度 0〜100) を返す。"""
    gray = np.array(Image.open(path).convert("L"))
    page = _indra_converter(model_dir).ocr(gray)
    return [(tuple(v * SCALE for v in ln.box), ln.text, float(ln.conf) * 100) for ln in page.lines]


def lines_in_cell(lines, x0: int, x1: int, y0: int, y1: int) -> Read:
    """中心がセル内にある行を集めて 1 セル分の読み取り結果にする。"""
    hits = sorted(
        (ln for ln in lines if x0 <= (ln[0][0] + ln[0][2]) / 2 < x1 and y0 <= (ln[0][1] + ln[0][3]) / 2 < y1),
        key=lambda ln: ln[0][0],
    )
    if not hits:
        return Read("", -1)
    # indra-ocr は "1. 99" のように数字の間へ空白を入れることがあるので除く
    text = "".join(t for _, t, _ in hits).replace(" ", "")
    read = Read(text, min(c for _, _, c in hits))
    # 1 行が縦罫線をまたいでいる = 隣のセルの文字と結合されている
    m = 3 * SCALE
    if any(b[0] < x0 - m or b[2] > x1 + m for b, _, _ in hits):
        read.note = f"隣のセルと結合して読まれました ('{' / '.join(t for _, t, _ in hits)}')"
    return read


def _columns(v_mask: np.ndarray, y0: int, y1: int, expected: int) -> list[tuple[int, int]]:
    xs = vertical_lines(v_mask, y0, y1)
    cols = list(zip(xs, xs[1:]))
    if len(cols) < expected:
        raise LayoutError(f"列数が不足しています（検出 {len(cols)} 列、必要 {expected} 列）")
    return cols


def read_document(
    path: Path,
    engine: str = "tesseract",
    debug_dir: Path | None = None,
    indra_model_dir: str | None = None,
) -> dict:
    """帳票画像から OCR 文字列（と信頼度）を取り出す。判定はしない。

    表構造（セル位置）は罫線から求め、セル内の文字を engine で読む。
    tesseract はセルごとに OCR し、indra はページ全体を 1 回 OCR して行をセルに割り当てる。
    """
    gray = load_gray(path)
    h_mask, v_mask = line_masks(gray)
    layout = locate_layout(horizontal_lines(h_mask))

    # 計測表: ラベル列 + 4 列。表全体の帯で縦罫線を取る
    table_cols = _columns(v_mask, layout.spec_row[0], layout.summary_rows[-1][1], 5)[-4:]
    mag_cols = _columns(v_mask, *layout.magnetic_row, 5)[-4:]
    lot_col = _columns(v_mask, *layout.lot_row, 2)[1]

    boxes: list[tuple[int, int, int, int]] = []
    lines = indra_lines(path, indra_model_dir) if engine == "indra" else None

    def read(x: tuple[int, int], y: tuple[int, int]) -> Read:
        boxes.append((x[0], x[1], y[0], y[1]))
        if lines is not None:
            return lines_in_cell(lines, x[0], x[1], y[0], y[1])
        return ocr_cell(gray, x[0], x[1], y[0], y[1])

    raw = {
        "engine": engine,
        "lot": read(lot_col, layout.lot_row),
        "magnetic": {k: read(c, layout.magnetic_row) for k, c in zip(MAGNETIC, mag_cols)},
        "spec": {k: read(c, layout.spec_row) for k, c in zip(COLUMNS, table_cols) if k != "gauss"},
        "data": {k: [read(c, r) for r in layout.data_rows] for k, c in zip(COLUMNS, table_cols)},
        "summary": {
            k: {s: read(c, r) for s, r in zip(SUMMARY_ROWS, layout.summary_rows)}
            for k, c in zip(COLUMNS, table_cols)
        },
    }

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x0, x1, y0, y1 in boxes:
            m = 3 * SCALE
            cv2.rectangle(vis, (x0 + m, y0 + m), (x1 - m, y1 - m), (0, 0, 255), 2)
        cv2.imwrite(str(debug_dir / f"{path.stem}_cells.png"), vis)
    return raw


# ---------------------------------------------------------------- 判定

@dataclass
class Cell:
    text: str
    value: float | None = None
    status: str = OK
    notes: list[str] = field(default_factory=list)

    def flag(self, note: str) -> None:
        self.status = CHECK
        self.notes.append(note)


@dataclass
class Report:
    engine: str
    lot: str
    magnetic: dict[str, Cell]
    specs: dict[str, tuple[float, float] | None]
    data: dict[str, list[Cell]]
    summary: dict[str, dict[str, Cell]]
    issues: list[str]

    @property
    def verdict(self) -> str:
        cells = [c for col in self.data.values() for c in col]
        cells += [c for col in self.summary.values() for c in col.values()]
        cells += list(self.magnetic.values())
        if self.issues or any(c.status == CHECK for c in cells):
            return "要目視確認"
        if any(c.status == NG for c in cells):
            return "不合格"
        return "合格"


def _decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def parse_number(r: Read, decimals: int | None, min_conf: float) -> Cell:
    cell = Cell(r.text)
    if r.note:
        cell.flag(r.note)
    if not re.fullmatch(r"\d+(\.\d+)?", r.text):
        cell.flag(f"数値として読めません ('{r.text}')")
        return cell
    cell.value = float(r.text)
    if decimals is not None and _decimals(r.text) != decimals:
        cell.flag(f"小数桁数が {decimals} 桁ではありません ('{r.text}')")
    if r.conf < min_conf:
        cell.flag(f"OCR 信頼度が低い ({r.conf:.0f})")
    return cell


def judge(cell: Cell, spec: tuple[float, float] | None) -> None:
    if spec is None or cell.value is None or cell.status == CHECK:
        return
    lo, hi = spec
    if not lo <= cell.value <= hi:
        cell.status = NG
        cell.notes.append(f"規格外 ({lo:g}〜{hi:g})")


def resolve_spec(col: str, raw: dict, cfg: dict, issues: list[str]) -> tuple[float, float] | None:
    spec_cfg = cfg.get("columns", {}).get(col, {}).get("spec", "from_document")
    if isinstance(spec_cfg, dict):
        return float(spec_cfg["min"]), float(spec_cfg["max"])
    r = raw["spec"].get(col)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", r.text) if r else None
    if not m or float(m[1]) >= float(m[2]):
        issues.append(
            f"{COLUMN_LABELS[col]}: 検査規格欄を読めません ('{r.text if r else ''}')。"
            "config.yaml に spec を設定してください"
        )
        return None
    if r.conf < cfg.get("min_confidence", 60):
        issues.append(f"{COLUMN_LABELS[col]}: 検査規格欄の OCR 信頼度が低い ({r.conf:.0f})")
    return float(m[1]), float(m[2])


def cross_check(col: str, cells: list[Cell], summary: dict[str, Cell], issues: list[str]) -> None:
    """帳票の Max/Min/Ave/c/n と、読み取った 20 値からの再計算値を照合する。

    1 文字でも読み誤ると不一致になりやすいため、OCR 誤読の検出に使う。
    """
    label = COLUMN_LABELS[col]
    values = [c.value for c in cells]
    if any(v is None for v in values):
        return
    calc = {"max": max(values), "min": min(values), "ave": float(np.mean(values))}
    for key, calc_v in calc.items():
        s = summary[key]
        if s.value is None:
            continue
        # 帳票の表示桁で丸めた誤差まで許容（平均値は四捨五入表示のため）
        tol = 0.5 * 10 ** -_decimals(s.text) + 1e-9 if key == "ave" else 1e-9
        if abs(s.value - calc_v) > tol:
            s.flag(f"再計算値 {calc_v:g} と不一致")
            rows = [str(i + 1) for i, v in enumerate(values) if v == calc_v] if key != "ave" else []
            hint = f"（再計算値は No.{','.join(rows)} の値）" if rows else ""
            issues.append(f"{label}: 帳票の{SUMMARY_LABELS[key]} {s.text} ≠ 再計算値 {calc_v:g}{hint}")

    cn = summary["cn"]
    m = re.fullmatch(r"(\d+)/(\d+)", cn.text)
    n_ng = sum(c.status == NG for c in cells)
    if not m:
        cn.flag(f"c/n を読めません ('{cn.text}')")
    elif int(m[2]) != len(cells) or int(m[1]) != n_ng:
        cn.flag(f"判定結果 {n_ng}/{len(cells)} と不一致")
        issues.append(f"{label}: 帳票の c/n {cn.text} ≠ 判定結果 {n_ng}/{len(cells)}")


def evaluate(raw: dict, cfg: dict) -> Report:
    min_conf = cfg.get("min_confidence", 60)
    issues: list[str] = []
    specs, data, summary = {}, {}, {}

    for col in COLUMNS:
        decimals = cfg.get("columns", {}).get(col, {}).get("decimals")
        specs[col] = resolve_spec(col, raw, cfg, issues)
        data[col] = [parse_number(r, decimals, min_conf) for r in raw["data"][col]]
        for c in data[col]:
            judge(c, specs[col])
        summary[col] = {}
        for key in ("max", "min", "ave"):
            summary[col][key] = parse_number(
                raw["summary"][col][key], decimals if key != "ave" else None, min_conf
            )
        cn = raw["summary"][col]["cn"]
        summary[col]["cn"] = Cell(cn.text)
        cross_check(col, data[col], summary[col], issues)

    magnetic = {}
    for name, r in raw["magnetic"].items():
        cell = parse_number(r, None, min_conf)
        spec = (cfg.get("magnetic") or {}).get(name)
        judge(cell, (float(spec["min"]), float(spec["max"])) if spec else None)
        magnetic[name] = cell

    return Report(raw["engine"], raw["lot"].text, magnetic, specs, data, summary, issues)


# ---------------------------------------------------------------- 出力

def _pad(s: str, width: int) -> str:
    """全角文字を 2 桁として右寄せ。"""
    w = sum(2 if unicodedata.east_asian_width(ch) in "FWA" else 1 for ch in s)
    return " " * max(width - w, 0) + s


def _fmt(cell: Cell) -> str:
    mark = {OK: "", NG: " NG", CHECK: " ?"}[cell.status]
    return (cell.text or "(空)") + mark


def print_report(path: Path, rep: Report) -> None:
    w = 14
    print(f"ファイル : {path}  (OCR: {rep.engine})")
    print(f"ロットNo.: {rep.lot or '(読み取れず)'}")
    print()
    print("磁気特性値: " + "  ".join(f"{k}={_fmt(c)}" for k, c in rep.magnetic.items()))
    print()
    print(_pad("No.", 8) + "".join(_pad(COLUMN_LABELS[c], w) for c in COLUMNS))
    spec_txt = [f"{s[0]:g}-{s[1]:g}" if s else "(不明)" for s in rep.specs.values()]
    print(_pad("規格", 8) + "".join(_pad(s, w) for s in spec_txt))
    print("-" * (8 + w * len(COLUMNS)))
    for i in range(N_DATA_ROWS):
        print(_pad(str(i + 1), 8) + "".join(_pad(_fmt(rep.data[c][i]), w) for c in COLUMNS))
    print("-" * (8 + w * len(COLUMNS)))
    for key in SUMMARY_ROWS:
        print(_pad(SUMMARY_LABELS[key], 8) + "".join(_pad(_fmt(rep.summary[c][key]), w) for c in COLUMNS))
    print("(NG=規格外, ?=要確認)")

    details = []
    for col in COLUMNS:
        for i, c in enumerate(rep.data[col]):
            if c.status != OK:
                details.append(f"[{c.status}] {COLUMN_LABELS[col]} No.{i + 1}: {c.text} - {'; '.join(c.notes)}")
    for col in COLUMNS:
        for key, c in rep.summary[col].items():
            if c.status == CHECK and not any(n.startswith(("再計算", "判定結果")) for n in c.notes):
                details.append(f"[{c.status}] {COLUMN_LABELS[col]} {SUMMARY_LABELS[key]}: {'; '.join(c.notes)}")
    for name, c in rep.magnetic.items():
        if c.status != OK:
            details.append(f"[{c.status}] 磁気特性 {name}: {c.text} - {'; '.join(c.notes)}")
    details += [f"[{CHECK}] {s}" for s in rep.issues]

    print()
    if details:
        print("■ 指摘事項")
        for d in details:
            print("  " + d)
        print()
    print(f"■ 総合判定: {rep.verdict}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="検査成績書の OCR 良否判定（ローカル処理）")
    ap.add_argument("images", nargs="+", type=Path, help="検査成績書の画像ファイル")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--engine", choices=["tesseract", "indra"], help="OCR エンジン（既定は config.yaml の engine）")
    ap.add_argument("--debug", action="store_true", help="セル切り出し位置を debug/ に画像出力")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    engine = args.engine or cfg.get("engine", "tesseract")
    if engine == "tesseract":
        pytesseract.pytesseract.tesseract_cmd = find_tesseract(cfg)

    codes = {"合格": 0, "不合格": 1, "要目視確認": 2}
    worst = 0
    for i, path in enumerate(args.images):
        if i:
            print("\n" + "=" * 64 + "\n")
        try:
            raw = read_document(
                path, engine, Path("debug") if args.debug else None, cfg.get("indra_model_dir")
            )
        except LayoutError as e:
            print(f"{path}: 表の構造を検出できません - {e}", file=sys.stderr)
            worst = max(worst, 3)
            continue
        rep = evaluate(raw, cfg)
        print_report(path, rep)
        worst = max(worst, codes[rep.verdict])
    return worst


if __name__ == "__main__":
    sys.exit(main())
