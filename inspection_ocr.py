"""検査成績書の計測値をローカル OCR で読み取り、良否判定する。

OCR エンジンは Tesseract か RapidOCR（PP-OCRv6、ONNX Runtime、CPU）を選べる。
どちらも画像は外部に送信せず、すべてこの PC 上で処理する
（RapidOCR のモデルはパッケージに同梱されていて、通信しない）。

帳票の様式は設定ファイル（configs/*.yaml）で切り替える。
type: neomag はこのファイルの NeoMag 専用ロジック、type: template は template_ocr.py の汎用処理で読む。

入力は画像・PDF・Excel（document_input.py）。PDF はページごとに 1 件として判定する。
電子 PDF・Excel の文字（テキスト層）は OCR せずにそのまま読む。

使い方:
    uv run inspection_ocr.py ファイル [--config configs/neomag.yaml] [--engine rapidocr] [--debug]

終了コード: 0=合格, 1=不合格, 2=要目視確認, 3=読み取りエラー
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from document_input import load_source, page_count
from ocr_common import (  # noqa: F401  (Read 等は tests から io.X として参照される)
    CHECK,
    CONFIG_DIR,
    NG,
    OK,
    Cell,
    LayoutError,
    Read,
    Row,
    decimals_of,
    find_tesseract,
    judge,
    lines_in_cell,
    load_config,
    ocr_cell,
    page_lines,
    pad,
    parse_number,
    setup_engine,
    verdict_of,
)

SCALE = 4  # 低解像度の帳票を拡大してから OCR する倍率
NEOMAG_WIDTH = 500  # 罫線検出の閾値はサンプル（幅 500px の GIF）の解像度が前提。PDF はこの幅で画像にする
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


# ---------------------------------------------------------------- 表構造の検出

def load_gray(path: Path, page: int = 0) -> np.ndarray:
    return load_source(path, page, width=NEOMAG_WIDTH).scaled(SCALE).gray


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
    page: int = 0,
) -> dict:
    """帳票画像から OCR 文字列（と信頼度）を取り出す。判定はしない。

    表構造（セル位置）は罫線から求め、セル内の文字を engine で読む。
    tesseract はセルごとに OCR し、rapidocr はページ全体を 1 回 OCR して行をセルに割り当てる。
    テキスト層（電子 PDF の文字）がセル内にあれば、OCR の代わりにそれを読む。
    """
    doc = load_source(path, page, width=NEOMAG_WIDTH)
    big = doc.scaled(SCALE)
    gray, words = big.gray, big.words
    h_mask, v_mask = line_masks(gray)
    layout = locate_layout(horizontal_lines(h_mask))

    # 計測表: ラベル列 + 4 列。表全体の帯で縦罫線を取る
    table_cols = _columns(v_mask, layout.spec_row[0], layout.summary_rows[-1][1], 5)[-4:]
    mag_cols = _columns(v_mask, *layout.magnetic_row, 5)[-4:]
    lot_col = _columns(v_mask, *layout.lot_row, 2)[1]

    boxes: list[tuple[int, int, int, int]] = []
    lines = None  # RapidOCR のページ全体の結果。OCR が必要になったときに 1 回だけ読む

    def read(x: tuple[int, int], y: tuple[int, int]) -> Read:
        nonlocal lines
        boxes.append((x[0], x[1], y[0], y[1]))
        if words:
            r = lines_in_cell(words, x[0], x[1], y[0], y[1], 3 * SCALE)
            if r.text:
                return r
        if engine == "rapidocr":
            if lines is None:
                lines = page_lines(doc.gray, SCALE)
            return lines_in_cell(lines, x[0], x[1], y[0], y[1], 3 * SCALE)
        return ocr_cell(gray, x[0], x[1], y[0], y[1], 3 * SCALE)

    raw = {
        "engine": f"テキスト層 + {engine}" if words else engine,
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
        name = f"{Path(path).stem}{f'_p{page + 1}' if page else ''}_cells.png"
        cv2.imencode(".png", vis)[1].tofile(debug_dir / name)  # 日本語パス対応
    return raw


# ---------------------------------------------------------------- 判定

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
        return verdict_of((c.status for c in cells), bool(self.issues))

    def rows(self) -> list[Row]:
        def row(item: str, c: Cell) -> Row:
            return Row(item, c.text, c.status, "; ".join(c.notes))

        out = [Row("ロットNo.", self.lot, OK)]
        out += [row(f"磁気特性 {k}", c) for k, c in self.magnetic.items()]
        for col in COLUMNS:
            s = self.specs[col]
            out.append(Row(f"{COLUMN_LABELS[col]} 規格", f"{s[0]:g}-{s[1]:g}" if s else "(不明)", OK if s else CHECK))
        for col in COLUMNS:
            out += [row(f"{COLUMN_LABELS[col]} No.{i + 1}", c) for i, c in enumerate(self.data[col])]
        for col in COLUMNS:
            out += [row(f"{COLUMN_LABELS[col]} {SUMMARY_LABELS[k]}", c) for k, c in self.summary[col].items()]
        out += [Row("（全体）", "", CHECK, s) for s in self.issues]
        return out


def resolve_spec(col: str, raw: dict, cfg: dict, issues: list[str]) -> tuple[float, float] | None:
    spec_cfg = cfg.get("columns", {}).get(col, {}).get("spec", "from_document")
    if isinstance(spec_cfg, dict):
        return float(spec_cfg["min"]), float(spec_cfg["max"])
    r = raw["spec"].get(col)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", r.text) if r else None
    if not m or float(m[1]) >= float(m[2]):
        issues.append(
            f"{COLUMN_LABELS[col]}: 検査規格欄を読めません ('{r.text if r else ''}')。"
            "設定ファイル（GUI の様式設定）で規格を指定してください"
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
        tol = 0.5 * 10 ** -decimals_of(s.text) + 1e-9 if key == "ave" else 1e-9
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
    print(pad("No.", 8) + "".join(pad(COLUMN_LABELS[c], w) for c in COLUMNS))
    spec_txt = [f"{s[0]:g}-{s[1]:g}" if s else "(不明)" for s in rep.specs.values()]
    print(pad("規格", 8) + "".join(pad(s, w) for s in spec_txt))
    print("-" * (8 + w * len(COLUMNS)))
    for i in range(N_DATA_ROWS):
        print(pad(str(i + 1), 8) + "".join(pad(_fmt(rep.data[c][i]), w) for c in COLUMNS))
    print("-" * (8 + w * len(COLUMNS)))
    for key in SUMMARY_ROWS:
        print(pad(SUMMARY_LABELS[key], 8) + "".join(pad(_fmt(rep.summary[c][key]), w) for c in COLUMNS))
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


def analyze(path: Path, cfg: dict, engine: str, debug_dir: Path | None = None, page: int = 0):
    """設定の type に応じて読み取り・判定し、Report（verdict と rows() を持つ）を返す。

    page は PDF のページ番号（0 始まり）。構造を検出できなければ LayoutError。
    """
    if cfg.get("type", "neomag") == "template":
        import template_ocr  # NeoMag だけ使う場合は読み込まない

        return template_ocr.analyze(path, cfg, engine, debug_dir, page)
    raw = read_document(path, engine, debug_dir, page)
    return evaluate(raw, cfg)


def report(path: Path | str, rep) -> None:
    if isinstance(rep, Report):
        print_report(path, rep)
    else:
        import template_ocr

        template_ocr.print_report(path, rep)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="検査成績書の OCR 良否判定（ローカル処理）")
    ap.add_argument("images", nargs="+", type=Path, help="検査成績書のファイル（画像・PDF・Excel）")
    ap.add_argument(
        "--config", type=Path, default=CONFIG_DIR / "neomag.yaml",
        help="様式の設定ファイル（省略時は configs/neomag.yaml）",
    )
    ap.add_argument("--engine", choices=["tesseract", "rapidocr"], help="OCR エンジン（既定は設定ファイルの engine）")
    ap.add_argument("--debug", action="store_true", help="セル切り出し位置を debug/ に画像出力")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    engine = args.engine or cfg.get("engine", "tesseract")
    setup_engine(engine, cfg)

    codes = {"合格": 0, "不合格": 1, "要目視確認": 2}
    worst = 0
    docs = [(path, page, n) for path in args.images for n in [page_count(path)] for page in range(n)]
    for i, (path, page, n) in enumerate(docs):
        if i:
            print("\n" + "=" * 64 + "\n")
        label = f"{path} ({page + 1}/{n} ページ)" if n > 1 else str(path)
        try:
            rep = analyze(path, cfg, engine, Path("debug") if args.debug else None, page)
        except LayoutError as e:
            print(f"{label}: 表の構造を検出できません - {e}", file=sys.stderr)
            worst = max(worst, 3)
            continue
        report(label, rep)
        worst = max(worst, codes[rep.verdict])
    return worst


if __name__ == "__main__":
    sys.exit(main())
