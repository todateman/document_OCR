"""汎用様式（type: template）の読み取りと良否判定。

GUI（ocr_gui.py）で基準画像に枠を描いて作った設定ファイルを使う。
入力画像を基準画像に位置合わせ（ORB 特徴点 + ホモグラフィ）してから、
枠の座標で各項目を切り出して読む。

項目の種類 (kind):
    text   … 文字。expect に一致→OK、ng_values に一致→NG、それ以外→要確認
    number … 数値。min/max で OK/NG、書式・信頼度の問題→要確認
    table  … 枠内を rows×cols のセルに分け、各セルを text / number として判定
    mark   … 手書きの○・チェックで選ぶ欄。選ばれた選択肢が expect と違えば NG
"""

from __future__ import annotations

import functools
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ocr_common import (
    CHECK,
    NG,
    OK,
    Cell,
    LayoutError,
    Read,
    Row,
    lines_in_cell,
    load_image_gray,
    ocr_cell,
    page_lines,
    pad,
    parse_number,
    verdict_of,
    width,
)

KINDS = {"text": "文字", "number": "数値", "table": "表", "mark": "マーク"}

MIN_INLIERS = 25  # 位置合わせに必要な特徴点の一致数
MAX_SIDE = 2000  # 位置合わせ・OCR の前にこれより大きい画像は縮小する
MARK_THRESHOLD = 0.3  # マークのスコア（手書き線の面積 / 文字 1 字分の面積）の既定の閾値
DITTO = {"↑", "〃", "同上", "″", "''"}
_DASHES = "−－ー‐―–—"


# ---------------------------------------------------------------- 画像

def load_page(path: Path) -> np.ndarray:
    gray = load_image_gray(path)
    s = MAX_SIDE / max(gray.shape)
    if s < 1:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return gray


def reference_path(cfg: dict) -> Path:
    p = Path(cfg.get("reference_image", ""))
    return p if p.is_absolute() else Path(cfg.get("_dir", ".")) / p


@functools.lru_cache(maxsize=8)
def _reference(path: str, mtime: float):
    gray = load_page(Path(path))
    orb = cv2.ORB_create(nfeatures=5000)
    kp, des = orb.detectAndCompute(gray, None)
    return gray, kp, des


def load_reference(cfg: dict):
    """基準画像（グレースケール）と特徴点。ファイルが変わるまでキャッシュする。"""
    p = reference_path(cfg)
    if not p.is_file():
        raise LayoutError(f"基準画像が見つかりません: {p}")
    return _reference(str(p), p.stat().st_mtime)


def align(gray: np.ndarray, cfg: dict) -> np.ndarray:
    """入力画像を基準画像の座標系に合わせて変形した画像を返す。"""
    ref, kp_ref, des_ref = load_reference(cfg)
    kp, des = cv2.ORB_create(nfeatures=5000).detectAndCompute(gray, None)
    if des is None or des_ref is None or len(kp) < MIN_INLIERS:
        raise LayoutError("画像から特徴点を検出できません")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des, des_ref, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < MIN_INLIERS:
        raise LayoutError(f"基準画像と一致する箇所が少なすぎます（{len(good)} 点）。様式が違う可能性があります")
    src = np.float32([kp[m.queryIdx].pt for m in good])
    dst = np.float32([kp_ref[m.trainIdx].pt for m in good])
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    n_in = int(inliers.sum()) if inliers is not None else 0
    if H is None or n_in < MIN_INLIERS:
        raise LayoutError(f"基準画像と位置合わせできません（一致 {n_in} 点）。様式が違う可能性があります")

    # 変形後の入力画像の四隅が、基準画像と同程度の大きさの凸四角形になっているか
    h, w = gray.shape
    corners = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H)
    area = cv2.contourArea(corners)
    ref_area = ref.shape[0] * ref.shape[1]
    if not cv2.isContourConvex(corners.astype(np.int32)) or not 0.3 < area / ref_area < 3:
        raise LayoutError("位置合わせの結果が不自然です。様式が違う可能性があります")
    return cv2.warpPerspective(gray, H, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR, borderValue=255)


@dataclass
class Ink:
    """手書きマークの検出用。罫線を除いた黒画素と、1 文字の高さの目安。"""
    clean: np.ndarray
    char_h: int


def ink_of(gray: np.ndarray) -> Ink:
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ボールペンの細い線は中間調になるので、Otsu より明るい側まで黒とみなす
    bw = ((gray < min(otsu + 50, 210)) * 255).astype(np.uint8)
    _, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    hs = st[1:, cv2.CC_STAT_HEIGHT]
    hs = hs[(hs >= 5) & (hs <= gray.shape[0] // 20)]
    char_h = int(np.percentile(hs, 90)) if len(hs) else 20
    # 罫線（文字より長い直線）を除く。手書きの曲線は残る
    k = int(char_h * 1.5)
    h = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k, 1)))
    v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    clean = bw & ~cv2.dilate(h | v, np.ones((3, 3), np.uint8))
    # かすれた手書き線や、罫線と重なって切れた○をつなぐ
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return Ink(clean, char_h)


def mark_score(ink: Ink, box) -> float:
    """枠内の手書き線（縦横とも文字より大きい塊）の量。

    印刷された文字だけなら 0 に近く、○で囲まれていれば文字と○がつながって大きな塊になる。
    横に並んだ文字がつながった塊は高さが文字 1 字分しかないので数えない。
    """
    x0, y0, x1, y1 = box
    # ○は枠からはみ出して書かれることが多いので、周囲も含めて大きさを測り、中心が枠内のものを数える
    p = ink.char_h
    H, W = ink.clean.shape
    ex0, ey0, ex1, ey1 = max(x0 - p, 0), max(y0 - p, 0), min(x1 + p, W), min(y1 + p, H)
    crop = ink.clean[ey0:ey1, ex0:ex1]
    if crop.size == 0 or x1 <= x0 or y1 <= y0:
        return 0.0
    _, _, st, cent = cv2.connectedComponentsWithStats(crop, 8)
    big = 1.3 * ink.char_h
    area = 0
    for s, (cx, cy) in zip(st[1:], cent[1:]):
        if not (x0 <= cx + ex0 < x1 and y0 <= cy + ey0 < y1):
            continue
        if s[cv2.CC_STAT_WIDTH] >= big and s[cv2.CC_STAT_HEIGHT] >= big:
            area += int(s[cv2.CC_STAT_AREA])
    return area / ink.char_h**2


# ---------------------------------------------------------------- 表の分割

def _cluster(indices: np.ndarray, gap: int) -> list[int]:
    groups: list[list[int]] = []
    for i in indices:
        if groups and i - groups[-1][-1] <= gap:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return [int(np.mean(g)) for g in groups]


def _split(lines: list[int], length: int, n: int) -> list[int]:
    """罫線位置から n 区間の境界（n+1 個）を作る。合わなければ等分。"""
    edges = [0] + _inner(lines, length) + [length]
    if len(edges) == n + 1:
        return edges
    return [round(length * i / n) for i in range(n + 1)]


def _inner(lines: list[int], length: int) -> list[int]:
    """枠の端に近い罫線（枠そのもの）を除いた内側の罫線。"""
    edge = max(length * 0.05, 6)
    return [p for p in lines if edge < p < length - edge]


def grid_lines(ref: np.ndarray, box) -> tuple[list[int], list[int]]:
    """枠内の横罫線の y と縦罫線の x（枠の左上からの相対位置）。"""
    x0, y0, x1, y1 = box
    crop = ref[y0:y1, x0:x1]
    if crop.size == 0:
        return [], []
    _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = bw.shape
    # わずかに傾いた罫線も 1 本につながるよう、線と直交する方向に太らせてから長い直線を取り出す
    h_mask = cv2.morphologyEx(
        cv2.dilate(bw, np.ones((5, 1), np.uint8)), cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 2, 10), 1)),
    )
    v_mask = cv2.morphologyEx(
        cv2.dilate(bw, np.ones((1, 5), np.uint8)), cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 2, 10))),
    )
    ys = _cluster(np.where(h_mask.mean(axis=1) / 255 > 0.7)[0], 4)
    xs = _cluster(np.where(v_mask.mean(axis=0) / 255 > 0.7)[0], 4)
    return ys, xs


def detect_grid(ref: np.ndarray, box) -> tuple[int, int]:
    """枠内の罫線から表の行数・列数を推定する（GUI で表を追加したときの初期値）。"""
    x0, y0, x1, y1 = box
    ys, xs = grid_lines(ref, box)
    return len(_inner(ys, y1 - y0)) + 1, len(_inner(xs, x1 - x0)) + 1


def table_edges(ref: np.ndarray, box, rows: int, cols: int) -> tuple[list[int], list[int]]:
    """表の枠内の行・列の境界（基準画像の座標）。基準画像の罫線に合わせ、合わなければ等分。"""
    x0, y0, x1, y1 = box
    rows, cols = max(int(rows or 1), 1), max(int(cols or 1), 1)
    ys, xs = grid_lines(ref, box)
    return [y0 + y for y in _split(ys, y1 - y0, rows)], [x0 + x for x in _split(xs, x1 - x0, cols)]


def _labels(given, n: int) -> list[str]:
    """行名・列名。指定が足りない分は 1, 2, ... の番号。1 行（1 列）だけなら指定が無ければ名前なし。"""
    given = ["" if s is None else str(s) for s in (given or [])]
    if n == 1 and not given:
        return [""]
    return [given[i] if i < len(given) else str(i + 1) for i in range(n)]


def _skips(spec) -> set[tuple[int, int]]:
    """"5-3, 5-6" や ["5-3"] 形式の除外セル（行-列、1 始まり）。"""
    if isinstance(spec, str):
        spec = spec.split(",")
    out = set()
    for s in spec or []:
        m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(s))
        if m:
            out.add((int(m[1]), int(m[2])))
    return out


@dataclass
class Target:
    """読み取り対象 1 つ（text / number の単独項目、または表の 1 セル）。"""
    name: str
    box: tuple[int, int, int, int]
    rule: dict
    above: int | None = None  # 表で真上のセルの Target 番号（↑ の解決用）


def expand(cfg: dict, ref: np.ndarray) -> list[tuple[dict, list[Target]]]:
    """設定の各項目を、読み取り単位（表はセルごと）に展開する。"""
    out = []
    n = 0
    for f in cfg.get("fields") or []:
        kind = f.get("kind", "text")
        box = tuple(int(v) for v in f["box"]) if f.get("box") else None
        targets: list[Target] = []
        if kind == "table" and box:
            ys, xs = table_edges(ref, box, f.get("rows", 1), f.get("cols", 1))
            rows, cols = len(ys) - 1, len(xs) - 1
            rl, cl = _labels(f.get("row_labels"), rows), _labels(f.get("col_labels"), cols)
            skip = _skips(f.get("skip"))
            rule = {**f, "kind": f.get("cell_kind", "number")}
            index: dict[tuple[int, int], int] = {}
            for r in range(rows):
                for c in range(cols):
                    if (r + 1, c + 1) in skip:
                        continue
                    parts = [f.get("name", ""), rl[r], cl[c]]
                    name = " ".join(p for p in parts if p)
                    index[r, c] = n
                    targets.append(Target(name, (xs[c], ys[r], xs[c + 1], ys[r + 1]), rule, index.get((r - 1, c))))
                    n += 1
        elif kind in ("text", "number") and box:
            targets.append(Target(f.get("name", ""), box, f))
            n += 1
        out.append((f, targets))
    return out


# ---------------------------------------------------------------- 判定

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", "", text)


def _number_text(text: str, unit: str | None) -> str:
    t = normalize(text)
    for d in _DASHES:
        t = t.replace(d, "-")
    if unit:
        u = normalize(unit)
        if t.lower().endswith(u.lower()):
            t = t[: -len(u)]
    return t


def _as_list(v) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, str):
        return [s for s in (x.strip() for x in v.split(",")) if s]
    return [str(x) for x in v]


def _num(v) -> float | None:
    return None if v is None or v == "" else float(v)


def judge_text(r: Read, rule: dict, min_conf: float) -> Cell:
    cell = Cell(normalize(r.text))
    if r.note:
        cell.flag(r.note)
    if not cell.text:
        if not rule.get("allow_blank"):
            cell.flag("読み取れません（空欄）")
        return cell
    key = cell.text.casefold()
    expect = [normalize(s).casefold() for s in _as_list(rule.get("expect"))]
    ng = [normalize(s).casefold() for s in _as_list(rule.get("ng_values"))]
    if key in ng:
        cell.status = NG
        cell.notes.append("NG の値です")
    elif expect and key not in expect:
        cell.flag(f"期待値（{' / '.join(_as_list(rule.get('expect')))}）と違います")
    elif r.conf < min_conf and (expect or ng):
        cell.flag(f"OCR 信頼度が低い ({r.conf:.0f})")
    return cell


def judge_number(r: Read, rule: dict, min_conf: float, above: Cell | None = None) -> Cell:
    text = _number_text(r.text, rule.get("unit"))
    if text in DITTO and above is not None:
        cell = Cell(text, above.value, above.status, [f"「{text}」を上のセルと同じ値として判定"])
        return cell
    if not text and rule.get("allow_blank") and not r.note:
        return Cell("")
    dec = rule.get("decimals")
    cell = parse_number(Read(text, r.conf, r.note), None if dec in (None, "") else int(dec), min_conf, signed=True)
    lo, hi = _num(rule.get("min")), _num(rule.get("max"))
    if cell.value is not None and cell.status != CHECK:
        if lo is not None and cell.value < lo or hi is not None and cell.value > hi:
            cell.status = NG
            cell.notes.append(f"規格外 ({'' if lo is None else f'{lo:g}'}〜{'' if hi is None else f'{hi:g}'})")
    return cell


def judge_mark(scores: dict[str, float], rule: dict) -> Cell:
    thr = _num(rule.get("mark_threshold")) or MARK_THRESHOLD
    detail = "スコア " + " ".join(f"{k}={v:.2f}" for k, v in scores.items())
    if not scores:
        cell = Cell("")
        cell.flag("選択肢の枠がありません")
        return cell
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    marked = [k for k, v in ranked if v >= thr]
    if not marked:
        cell = Cell("")
        cell.flag(f"マークが見つかりません（{detail}）")
        return cell
    if len(marked) > 1:
        cell = Cell(" / ".join(marked))
        cell.flag(f"複数の選択肢にマークがあります（{detail}）")
        return cell
    cell = Cell(marked[0], notes=[detail])
    expect = _as_list(rule.get("expect"))
    if expect and marked[0] not in expect:
        cell.status = NG
        cell.notes.insert(0, f"期待（{' / '.join(expect)}）と違います")
    return cell


@dataclass
class Item:
    name: str
    cell: Cell
    box: tuple[int, int, int, int] | None
    index: int  # 設定の fields 内の位置（表のセルなら表の位置）


@dataclass
class TemplateReport:
    name: str
    engine: str
    items: list[Item]
    issues: list[str]
    image: np.ndarray | None = field(default=None, repr=False)  # 基準画像に合わせた入力画像

    @property
    def verdict(self) -> str:
        return verdict_of((it.cell.status for it in self.items), bool(self.issues))

    def rows(self) -> list[Row]:
        out = [Row(it.name, it.cell.text, it.cell.status, "; ".join(it.cell.notes), it.box, it.index) for it in self.items]
        out += [Row("（全体）", "", CHECK, s) for s in self.issues]
        return out


def read_targets(
    gray: np.ndarray, cfg: dict, engine: str, ref: np.ndarray, char_h: int = 20
) -> list[tuple[dict, list[tuple[Target, Read]]]]:
    """位置合わせ済みの画像から各読み取り単位の OCR 結果を取り出す（判定はしない）。"""
    fields = expand(cfg, ref)
    lines = page_lines(gray) if engine == "rapidocr" else None
    margin = max(2, char_h // 6)
    out = []
    for f, targets in fields:
        reads = []
        for t in targets:
            x0, y0, x1, y1 = t.box
            if lines is not None:
                r = lines_in_cell(lines, x0, x1, y0, y1, margin)
            else:
                lang = cfg.get("tesseract_lang", "jpn+eng") if t.rule.get("kind") == "text" else None
                r = ocr_cell(gray, x0, x1, y0, y1, margin, lang, scale=min(max(40 / char_h, 1), 4))
            reads.append((t, r))
        out.append((f, reads))
    return out


def evaluate(gray: np.ndarray, cfg: dict, engine: str) -> TemplateReport:
    ref, _, _ = load_reference(cfg)
    min_conf = cfg.get("min_confidence", 60)
    ink = ink_of(gray)
    items: list[Item] = []
    cells: list[Cell] = []
    for i, (f, reads) in enumerate(read_targets(gray, cfg, engine, ref, ink.char_h)):
        name = f.get("name", "")
        if f.get("kind") == "mark":
            opts = [o for o in f.get("options") or [] if o.get("box")]
            scores = {str(o.get("label", "")): mark_score(ink, [int(v) for v in o["box"]]) for o in opts}
            boxes = [[int(v) for v in o["box"]] for o in opts]
            hull = (
                (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
                if boxes else None
            )
            items.append(Item(name, judge_mark(scores, f), hull, i))
            continue
        for t, r in reads:
            if t.rule.get("kind") == "number":
                above = cells[t.above] if t.above is not None else None
                cell = judge_number(r, t.rule, min_conf, above)
            else:
                cell = judge_text(r, t.rule, min_conf)
            cells.append(cell)  # Target の通し番号（expand の n）と同じ順
            items.append(Item(t.name, cell, t.box, i))
    return TemplateReport(cfg.get("name", ""), engine, items, [], gray)


def analyze(path: Path, cfg: dict, engine: str, debug_dir: Path | None = None) -> TemplateReport:
    gray = align(load_page(path), cfg)
    rep = evaluate(gray, cfg, engine)
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        out = debug_dir / f"{Path(path).stem}_cells.png"
        cv2.imencode(".png", draw_boxes(gray, rep.rows()))[1].tofile(out)
    return rep


# ---------------------------------------------------------------- 出力

STATUS_BGR = {OK: (60, 160, 60), NG: (40, 40, 220), CHECK: (0, 150, 240)}


def draw_boxes(gray: np.ndarray, rows: list[Row]) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for r in rows:
        if r.box:
            cv2.rectangle(vis, r.box[:2], r.box[2:], STATUS_BGR.get(r.status, (128, 128, 128)), 2)
    return vis


def print_report(path: Path, rep: TemplateReport) -> None:
    print(f"ファイル : {path}  (OCR: {rep.engine})")
    print(f"様式     : {rep.name}")
    print()
    w = max([20] + [width(it.name) for it in rep.items]) + 2
    v = max([12] + [width(it.cell.text) for it in rep.items]) + 2
    print(pad("項目", w, left=True) + pad("読取値", v, left=True) + "判定")
    print("-" * (w + v + 6))
    for it in rep.items:
        print(pad(it.name, w, left=True) + pad(it.cell.text or "(空)", v, left=True) + it.cell.status)
    details = [f"[{it.cell.status}] {it.name}: {it.cell.text} - {'; '.join(it.cell.notes)}"
               for it in rep.items if it.cell.status != OK]
    details += [f"[{CHECK}] {s}" for s in rep.issues]
    print()
    if details:
        print("■ 指摘事項")
        for d in details:
            print("  " + d)
        print()
    print(f"■ 総合判定: {rep.verdict}")
