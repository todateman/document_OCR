import copy
import sys
from pathlib import Path

import pytest
import pytesseract

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import inspection_ocr as io  # noqa: E402

SAMPLE = ROOT / "sample" / "inspection_sample.gif"

# サンプル画像を目視で確認した正解値
EXPECTED_DATA = {
    "length": "20.02 20.02 20.00 20.01 20.02 20.03 20.03 20.02 20.01 20.03 "
              "20.03 20.07 20.03 20.02 20.01 20.03 20.02 20.04 20.02 20.03",
    "width": "10.03 9.99 10.00 10.03 10.02 9.99 9.97 10.02 10.03 10.00 "
             "10.02 10.03 10.00 10.02 10.01 10.02 10.00 10.03 10.02 10.00",
    "height": "1.99 1.99 2.00 1.99 1.96 2.00 2.02 2.00 2.00 2.03 "
              "2.02 2.00 2.04 1.99 1.97 1.99 1.98 2.00 2.04 2.00",
    "gauss": "1310 1300 1300 1310 1300 1270 1310 1330 1330 1300 "
             "1350 1340 1340 1330 1280 1360 1300 1280 1310 1300",
}
EXPECTED_SPEC = {"length": "19.9-20.1", "width": "9.9-10.1", "height": "1.95-2.05"}
EXPECTED_SUMMARY = {
    "length": ["20.07", "20.00", "20.02", "0/20"],
    "width": ["10.03", "9.97", "10.01", "0/20"],
    "height": ["2.04", "1.96", "2.00", "0/20"],
    "gauss": ["1360", "1270", "1312.5", "0/20"],
}
EXPECTED_MAGNETIC = {"Br": "10.44", "Hcb": "9.92", "Hcj": "22.49", "BHmax": "26.05"}


@pytest.fixture(scope="module")
def cfg():
    return io.load_config(ROOT / "configs" / "neomag.yaml")


@pytest.fixture(scope="module", params=["tesseract", "indra"])
def raw(request, cfg):
    if request.param == "tesseract":
        try:
            pytesseract.pytesseract.tesseract_cmd = io.find_tesseract(cfg)
        except FileNotFoundError:
            pytest.skip("Tesseract が未インストール")
    return io.read_document(SAMPLE, request.param)


def test_ocr_matches_expected(raw):
    assert raw["lot"].text == "NN14092301"
    for col, values in EXPECTED_DATA.items():
        assert [r.text for r in raw["data"][col]] == values.split(), col
    for col, spec in EXPECTED_SPEC.items():
        assert raw["spec"][col].text == spec
    for col, values in EXPECTED_SUMMARY.items():
        assert [raw["summary"][col][k].text for k in io.SUMMARY_ROWS] == values, col
    assert {k: r.text for k, r in raw["magnetic"].items()} == EXPECTED_MAGNETIC


def test_sample_passes(raw, cfg):
    rep = io.evaluate(raw, cfg)
    assert rep.issues == []
    assert rep.verdict == "合格"


def test_narrow_spec_detects_ng(raw, cfg):
    cfg = copy.deepcopy(cfg)
    cfg["columns"]["height"]["spec"] = {"min": 1.97, "max": 2.03}
    rep = io.evaluate(raw, cfg)
    ng_rows = [i + 1 for i, c in enumerate(rep.data["height"]) if c.status == io.NG]
    assert ng_rows == [5, 13, 19]  # 1.96, 2.04, 2.04
    # 帳票の c/n (0/20) と判定結果 (3/20) が食い違うので要確認になる
    assert rep.summary["height"]["cn"].status == io.CHECK
    assert rep.verdict == "要目視確認"


def test_misread_caught_by_cross_check(raw, cfg):
    raw = copy.deepcopy(raw)
    raw["data"]["length"][4] = io.Read("20.92", 95)  # 20.02 を 20.92 と誤読した想定
    rep = io.evaluate(raw, cfg)
    assert rep.data["length"][4].status == io.NG
    assert any("最大値" in s and "No.5" in s for s in rep.issues)
    assert rep.verdict == "要目視確認"


def test_line_spanning_cells_flagged():
    s = io.SCALE
    lines = [((100 * s, 0, 260 * s, 10 * s), "1300 1310", 99.0)]  # 2 セルにまたがる 1 行
    read = io.lines_in_cell(lines, 100 * s, 200 * s, 0, 12 * s)
    assert read.note
    assert io.parse_number(read, 0, 60).status == io.CHECK


def test_spaces_in_indra_line_removed():
    s = io.SCALE
    lines = [((10 * s, 0, 30 * s, 10 * s), "1. 99", 93.0)]
    read = io.lines_in_cell(lines, 0, 40 * s, 0, 12 * s)
    assert (read.text, read.note) == ("1.99", "")


def test_non_numeric_read_flagged(raw, cfg):
    raw = copy.deepcopy(raw)
    raw["data"]["width"][0] = io.Read("1O.03", 80)
    rep = io.evaluate(raw, cfg)
    assert rep.data["width"][0].status == io.CHECK
    assert rep.verdict == "要目視確認"
