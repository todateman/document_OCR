"""NeoMag 専用ロジック（inspection_ocr.py）と汎用様式（template_ocr.py）で共通の部品。

設定の読み込み、OCR エンジンの呼び出し、数値の書式チェックと規格判定。
"""

from __future__ import annotations

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

ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = ROOT / "settings.yaml"
CONFIG_DIR = ROOT / "configs"

OK, NG, CHECK = "OK", "NG", "要確認"


class LayoutError(Exception):
    """帳票の構造（罫線・位置合わせ）を検出できない。"""


# ---------------------------------------------------------------- 設定

def read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class _Dumper(yaml.SafeDumper):
    pass


def _represent_list(dumper: yaml.SafeDumper, data: list):
    # 枠の座標や選択肢の名前など、値だけの短いリストは 1 行で書く（人が読みやすいように）
    flow = all(isinstance(v, (int, float, str)) for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


def _represent_dict(dumper: yaml.SafeDumper, data: dict):
    flow = "box" in data and "label" in data  # マークの選択肢 {label, box}
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items(), flow_style=flow)


_Dumper.add_representer(list, _represent_list)
_Dumper.add_representer(dict, _represent_dict)


def write_yaml(path: Path, data: dict, header: str = "") -> None:
    """GUI からの保存用。"_" で始まるキー（実行時の補助情報）は書き出さない。"""
    clean = {k: v for k, v in data.items() if not k.startswith("_")}
    text = yaml.dump(clean, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=120)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + text)


def load_settings() -> dict:
    return read_yaml(SETTINGS_PATH) if SETTINGS_PATH.is_file() else {}


def load_config(path: Path) -> dict:
    """様式の設定に PC 共通設定（settings.yaml）を合わせて返す。

    様式側に値があればそちらを優先する（null は未設定扱い）。
    """
    path = Path(path)
    cfg = read_yaml(path)
    merged = {**load_settings(), **{k: v for k, v in cfg.items() if v is not None}}
    merged["_dir"] = str(path.resolve().parent)  # 基準画像など相対パスの起点
    return merged


def config_name(path: Path) -> str:
    try:
        return read_yaml(path).get("name") or path.stem
    except (OSError, yaml.YAMLError):
        return path.stem


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
        "Tesseract が見つかりません。settings.yaml の tesseract_cmd（GUI の「共通設定」）にパスを設定してください。"
    )


def setup_engine(engine: str, cfg: dict) -> None:
    if engine == "tesseract":
        pytesseract.pytesseract.tesseract_cmd = find_tesseract(cfg)


# ---------------------------------------------------------------- OCR

@dataclass
class Read:
    text: str
    conf: float  # 0〜100、文字が無ければ -1
    note: str = ""  # 読み取り時点で分かった問題（あれば要確認にする）


def ocr_cell(
    gray: np.ndarray, x0: int, x1: int, y0: int, y1: int,
    margin: int = 12, lang: str | None = None, scale: float = 1,
) -> Read:
    m = margin  # 罫線を含めないよう内側に切り出す
    crop = gray[y0 + m : y1 - m, x0 + m : x1 - m]
    if crop.size == 0:
        return Read("", -1)
    if scale != 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    crop = cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    # 二値化や文字種ホワイトリストは使わない。低解像度の帳票では二値化で字形が崩れ、
    # ホワイトリストは誤読を数字に当てはめて隠してしまう（信頼度も無意味になる）。
    # 数字以外が読めた場合は後段の書式チェックで「要確認」にする。
    try:
        data = pytesseract.image_to_data(
            crop, lang=lang, config="--psm 7 --oem 1", output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractError as e:
        return Read("", -1, f"Tesseract のエラー（言語データ {lang} が無い可能性）: {e}")
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


def indra_page_lines(
    gray: np.ndarray, model_dir: str | None, scale: float = 1
) -> list[tuple[tuple[float, ...], str, float]]:
    """ページ全体を indra-ocr で読み、(scale 倍した座標の box, 文字列, 信頼度 0〜100) を返す。"""
    page = _indra_converter(model_dir).ocr(gray)
    return [(tuple(v * scale for v in ln.box), ln.text, float(ln.conf) * 100) for ln in page.lines]


def lines_in_cell(lines, x0: int, x1: int, y0: int, y1: int, margin: int = 12) -> Read:
    """中心がセル内にある行を集めて 1 セル分の読み取り結果にする。"""
    m = margin

    def inside(ln) -> bool:
        cx, cy = (ln[0][0] + ln[0][2]) / 2, (ln[0][1] + ln[0][3]) / 2
        if not (x0 <= cx < x1 and y0 <= cy < y1):
            return False
        # 縦罫線の上で検出された文字なしの行は罫線のかすれなので無視する
        return bool(ln[1].strip()) or x0 + m <= cx < x1 - m

    hits = sorted((ln for ln in lines if inside(ln)), key=lambda ln: ln[0][0])
    if not hits:
        return Read("", -1)
    # indra-ocr は "1. 99" のように数字の間へ空白を入れることがあるので除く
    text = "".join(t for _, t, _ in hits).replace(" ", "")
    read = Read(text, min(c for _, _, c in hits))
    # 1 行が縦罫線をまたいでいる = 隣のセルの文字と結合されている
    if any(b[0] < x0 - m or b[2] > x1 + m for b, _, _ in hits):
        read.note = f"隣のセルと結合して読まれました ('{' / '.join(t for _, t, _ in hits)}')"
    return read


def load_image_gray(path: Path) -> np.ndarray:
    # cv2.imread は GIF 非対応・日本語パス非対応のため Pillow で読む
    return np.array(Image.open(path).convert("L"))


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
class Row:
    """GUI・CSV 等に表示する 1 項目分の結果。"""
    item: str
    text: str
    status: str
    note: str = ""
    box: tuple[int, int, int, int] | None = None  # 基準画像上の (x0, y0, x1, y1)。汎用様式のみ
    index: int | None = None  # 汎用様式の設定 fields 内の位置


def verdict_of(statuses, has_issues: bool = False) -> str:
    statuses = list(statuses)
    if has_issues or CHECK in statuses:
        return "要目視確認"
    if NG in statuses:
        return "不合格"
    return "合格"


def decimals_of(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def parse_number(r: Read, decimals: int | None, min_conf: float, signed: bool = False) -> Cell:
    cell = Cell(r.text)
    if r.note:
        cell.flag(r.note)
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?" if signed else r"\d+(\.\d+)?", r.text):
        cell.flag(f"数値として読めません ('{r.text}')")
        return cell
    cell.value = float(r.text)
    if decimals is not None and decimals_of(r.text) != decimals:
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


# ---------------------------------------------------------------- 出力

def width(s: str) -> int:
    """全角文字を 2 桁とした表示幅。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "FWA" else 1 for ch in s)


def pad(s: str, n: int, left: bool = False) -> str:
    """表示幅 n に揃える（既定は右寄せ）。"""
    sp = " " * max(n - width(s), 0)
    return s + sp if left else sp + s
