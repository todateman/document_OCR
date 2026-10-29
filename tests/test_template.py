import copy
import io as _io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import template_ocr as t  # noqa: E402
from ocr_common import CHECK, NG, OK, LayoutError, Read, load_config, read_yaml, write_yaml  # noqa: E402

SAMPLE = ROOT / "sample" / "denshitenbin_kensa_seisekisho01.jpg"
CONFIG = ROOT / "configs" / "denshitenbin.yaml"

# サンプル画像を目視で確認した正解値（項目名 → 読取値）
EXPECTED = {
    "基本情報 機種名": "JL-180",
    "基本情報 器物番号": "ABCDEF",
    "総合判定": "合格",
    "外観・機能 表記・記号": "OK",
    "外観・機能 設置台": "OK",
    "外観・機能 外乱等": "無",
    "CAL後SPAN値 1": "180.0001",
    "CAL後SPAN値 2": "-1",
    "CAL後SPAN値 4": "0",
    "再現性 1回目 調整前フル点": "100.0003",
    "再現性 調整前 2 フル点": "-1",
    "再現性 調整後 4 フル点": "-1",
    "四隅 中心 調整前": "50.0001",
    "四隅 調整前 左後": "+4",
    "四隅 調整後 右後": "-1",
    "直線性 調整前 25% 荷重量": "50",
    "直線性 調整前 100% 増": "+4",
    "直線性 調整後 75% 減": "+1",
    "感じ 下限": "合",
    "感じ 秤量": "合",
}


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def page(cfg):
    return t.align(t.load_page(SAMPLE), cfg)


@pytest.fixture(scope="module")
def report(page, cfg):
    return t.evaluate(page, cfg, "indra")


def items(rep):
    return {it.name: it.cell for it in rep.items}


def test_sample_reads_expected(report):
    got = items(report)
    for name, text in EXPECTED.items():
        assert got[name].text == text, name
    # 斜線・矢印の欄（読まないセル）は項目に含まれない
    assert "直線性 調整前 100% 減" not in got
    assert len(report.items) == 87


def test_sample_passes(report):
    assert [(it.name, it.cell.notes) for it in report.items if it.cell.status != OK] == []
    assert report.verdict == "合格"


def _distort(gray: np.ndarray, kind: str) -> np.ndarray:
    h, w = gray.shape
    if kind == "rotate_shift":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -2, 1.0)
        M[:, 2] += (30, -20)
        return cv2.warpAffine(gray, M, (w, h), borderValue=255)
    if kind == "scale":
        return cv2.resize(gray, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    if kind == "perspective":
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[30, 10], [w - 10, 40], [w - 40, h - 5], [5, h - 30]])
        return cv2.warpPerspective(gray, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=255)
    if kind == "jpeg":
        buf = _io.BytesIO()
        Image.fromarray(gray).save(buf, "JPEG", quality=20)
        return np.array(Image.open(buf))
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["rotate_shift", "scale", "perspective", "jpeg"])
def test_alignment_handles_distortion(kind, cfg, tmp_path):
    gray = np.array(Image.open(SAMPLE).convert("L"))
    p = tmp_path / f"{kind}.png"
    Image.fromarray(_distort(gray, kind)).save(p)
    rep = t.analyze(p, cfg, "indra")
    got = items(rep)
    # 判定に使う値（数値・OK 欄・マーク）は元画像と同じに読めるか、読めなければ要確認になること
    # （劣化で信頼度が下がって要確認になるのは正しい動作。誤読のまま OK になってはいけない）
    for name, text in EXPECTED.items():
        if name.startswith("基本情報"):  # 表示のみの文字欄は劣化で崩れても判定に影響しない
            continue
        cell = got[name]
        assert cell.text == text or cell.status == CHECK, (kind, name, cell.text)
        assert cell.status in (OK, CHECK), (kind, name, cell.notes)
    for name in ("総合判定", "感じ 下限", "感じ 秤量"):
        assert (got[name].text, got[name].status) == (EXPECTED[name], OK), (kind, name)
    assert rep.verdict in ("合格", "要目視確認")


def test_other_form_is_rejected(cfg):
    with pytest.raises(LayoutError):
        t.analyze(ROOT / "sample" / "inspection_sample.gif", cfg, "indra")


def _with_marks(tmp_path, name, edit):
    gray = np.array(Image.open(SAMPLE).convert("L"))
    edit(gray)
    p = tmp_path / f"{name}.png"
    Image.fromarray(gray).save(p)
    return p


def _circle_fail(g):
    cv2.ellipse(g, (1037, 390), (45, 22), -10, 20, 380, 50, 2)  # 「不合格」を○で囲む


def test_mark_other_option_is_ng(cfg, tmp_path):
    def edit(g):
        g[374:407, 740:840] = 255  # 「合格」の○を消す
        _circle_fail(g)

    rep = t.analyze(_with_marks(tmp_path, "fail", edit), cfg, "indra")
    cell = items(rep)["総合判定"]
    assert (cell.text, cell.status) == ("不合格", NG)
    assert rep.verdict == "不合格"


def test_mark_on_both_options_needs_check(cfg, tmp_path):
    rep = t.analyze(_with_marks(tmp_path, "both", _circle_fail), cfg, "indra")
    assert items(rep)["総合判定"].status == CHECK
    assert rep.verdict == "要目視確認"


def test_no_mark_needs_check(cfg, tmp_path):
    def edit(g):
        g[374:407, 740:840] = 255

    rep = t.analyze(_with_marks(tmp_path, "none", edit), cfg, "indra")
    assert items(rep)["総合判定"].status == CHECK


def test_narrow_spec_detects_ng(page, cfg):
    cfg = copy.deepcopy(cfg)
    f = next(f for f in cfg["fields"] if f["name"] == "四隅 調整後")
    f["min"], f["max"] = -0.5, 0.5
    rep = t.evaluate(page, cfg, "indra")
    ng = [it.name for it in rep.items if it.cell.status == NG]
    assert ng == ["四隅 調整後 右後"]  # -1
    assert rep.verdict == "不合格"


def test_text_rules():
    rule = {"expect": ["OK"], "ng_values": ["NG"]}
    assert t.judge_text(Read("OK", 95), rule, 60).status == OK
    assert t.judge_text(Read("ｏｋ", 95), rule, 60).status == OK  # 全角・小文字も同じとみなす
    assert t.judge_text(Read("NG", 95), rule, 60).status == NG
    assert t.judge_text(Read("0K", 95), rule, 60).status == CHECK  # 読み誤りは NG ではなく要確認
    assert t.judge_text(Read("", -1), rule, 60).status == CHECK
    assert t.judge_text(Read("", -1), {"allow_blank": True}, 60).status == OK


@pytest.mark.parametrize(
    "text, unit, value",
    [("+4", None, 4), ("−1", None, -1), ("ー2", None, -2), ("50g", "g", 50), ("2 5%", "%", 25),
     ("100.0003g", "g", 100.0003), ("0. 1 m g", "mg", 0.1)],
)
def test_number_parsing(text, unit, value):
    cell = t.judge_number(Read(text, 95), {"unit": unit}, 60)
    assert cell.status == OK, cell.notes
    assert cell.value == pytest.approx(value)


def test_number_rules():
    rule = {"min": -2, "max": 2, "decimals": 0}
    assert t.judge_number(Read("+2", 95), rule, 60).status == OK  # 境界値は合格
    assert t.judge_number(Read("+3", 95), rule, 60).status == NG
    assert t.judge_number(Read("1.0", 95), rule, 60).status == CHECK  # 小数桁数が違う
    assert t.judge_number(Read("1O", 95), rule, 60).status == CHECK  # 数字以外
    assert t.judge_number(Read("1", 40), rule, 60).status == CHECK  # 信頼度不足
    above = t.judge_number(Read("+1", 95), rule, 60)
    ditto = t.judge_number(Read("↑", 95), rule, 60, above)
    assert (ditto.value, ditto.status) == (1, OK)


def test_mark_rules():
    rule = {"expect": ["合格"]}
    assert t.judge_mark({"合格": 1.2, "不合格": 0.0}, rule).status == OK
    assert t.judge_mark({"合格": 0.0, "不合格": 1.2}, rule).status == NG
    assert t.judge_mark({"合格": 0.1, "不合格": 0.0}, rule).status == CHECK
    assert t.judge_mark({"合格": 0.1, "不合格": 0.0}, {**rule, "mark_threshold": 0.05}).status == OK


def test_config_roundtrip(tmp_path):
    cfg = read_yaml(CONFIG)
    p = tmp_path / "copy.yaml"
    write_yaml(p, {**cfg, "_dir": "ignored"})
    assert read_yaml(p) == cfg
