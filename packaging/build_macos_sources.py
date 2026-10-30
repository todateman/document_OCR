"""macOS 版の実行ファイルに対応するソース一式を作る。

    uv run python packaging/build_macos_sources.py --output dist/sources.tar.gz

同梱するライブラリの一覧と版は packaging/macos_sources.yaml にある。OpenCV の wheel に含まれる dylib と
一覧が一致しないとき、取得したファイルの sha256 が一致しないときは失敗する。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import platform
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "packaging" / "macos_sources.yaml"
HOMEBREW_RAW = "https://raw.githubusercontent.com/Homebrew/homebrew-core"

EVIDENCE = {
    "binary": "バイナリに含まれる版の文字列と一致",
    "filename": "dylib のファイル名の版と一致",
    "inferred": "soname と Homebrew の履歴からの推定",
}


class SourceError(Exception):
    pass


def check_dylibs(manifest: dict, dylib_dir: Path) -> None:
    """wheel の dylib と一覧を照合する。OpenCV の更新で古いソースのまま公開しないため。"""
    wheel = metadata.version("opencv-python-headless")
    if wheel != manifest["opencv_wheel"]:
        raise SourceError(f"opencv-python-headless が {wheel} ですが、一覧は {manifest['opencv_wheel']} 用です")
    actual = {p.name for p in dylib_dir.glob("*.dylib")}
    listed = {d for lib in manifest["libraries"] for d in lib["dylibs"]}
    if actual != listed:
        raise SourceError(f"dylib が一覧と一致しません。一覧にない: {sorted(actual - listed)} / wheel にない: {sorted(listed - actual)}")


def download(url: str, dest: Path, sha256: str | None = None) -> str:
    """dest に保存して sha256 を返す。sha256 を指定したときは検証する。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-fsSL", "--retry", "4", "--retry-all-errors", "--connect-timeout", "30", "-o", str(dest), url],
        check=True,
    )
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    actual = h.hexdigest()
    if sha256 and actual != sha256:
        raise SourceError(f"sha256 が一致しません: {url}\n  期待: {sha256}\n  実際: {actual}")
    return actual


def file_name(url: str) -> str:
    return url.split("?")[0].rstrip("/").rsplit("/", 1)[1]


def formula_path(name: str) -> str:
    return f"Formula/{'lib' if name.startswith('lib') else name[0]}/{name}.rb"


def fetch_library(lib: dict, out: Path) -> None:
    d = out / "libraries" / f"{lib['name']}-{lib['version']}"
    download(lib["url"], d / file_name(lib["url"]), lib["sha256"])
    for i, patch in enumerate(lib.get("patches", []), 1):
        download(patch["url"], d / "patches" / f"{i:02}-{file_name(patch['url'])}", patch["sha256"])
    # ビルド手順として Homebrew の式も同梱する（コミットで固定されているので内容は変わらない）
    download(f"{HOMEBREW_RAW}/{lib['homebrew_commit']}/{formula_path(lib['name'])}", d / f"{lib['name']}.rb")


def runtime_packages() -> list[str]:
    """実行ファイルに含まれる Python パッケージの名前（この環境に入っているもののみ）。"""
    text = subprocess.run(
        ["uv", "export", "--no-dev", "--no-hashes", "--no-emit-project", "--no-header", "--no-annotate", "-q"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    names = {ln.split("==")[0] for ln in text.splitlines() if "==" in ln}
    names.add("pyinstaller")  # 実行ファイルの起動部分（bootloader）
    installed = set()
    for name in names:
        try:
            metadata.version(name)
            installed.add(name)
        except metadata.PackageNotFoundError:
            pass  # 他の OS でのみ使うパッケージ
    return sorted(installed)


def fetch_python_packages(manifest: dict, out: Path) -> list[tuple[str, str, str]]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = {(p["name"], p["version"]): p for p in lock["package"]}
    overrides = manifest["python_packages_without_sdist"]
    rows = []
    for name in runtime_packages():
        version = metadata.version(name)
        sdist = locked[(name, version)].get("sdist")
        if sdist:
            url, sha = sdist["url"], sdist["hash"].removeprefix("sha256:")
        elif name in overrides and overrides[name]["version"] == version:
            url, sha = overrides[name]["url"], overrides[name]["sha256"]
        else:
            raise SourceError(f"{name} {version} のソースの場所が不明です（macos_sources.yaml に追加してください）")
        download(url, out / "python" / f"{name}-{version}" / file_name(url), sha)
        rows.append((name, version, url))
    return rows


def write_index(manifest: dict, packages: list[tuple[str, str, str]], python_url: str, out: Path) -> None:
    lines = [
        "# 対応するソースの一覧",
        "",
        "macOS 版の実行ファイルに含まれるものの対応するソースです。",
        "",
        f"OpenCV の wheel（opencv-python-headless {manifest['opencv_wheel']}）に含まれるライブラリは、wheel の作成者が"
        "ビルドしたものです。作成時の環境の記録がないため、版は次の根拠で特定しました。推定のものは実際の版と異なる可能性があります。",
        "",
        "- " + "\n- ".join(f"`{k}`: {v}" for k, v in EVIDENCE.items()),
        "",
        "`libraries/` の各フォルダには、ソース、Homebrew が適用するパッチ（`patches/`）、ビルド手順となる Homebrew の式（`*.rb`）があります。",
        "`app/` はこのアプリのソース、`python/` は Python パッケージ（OpenCV を含む）のソース、`cpython/` は Python 本体のソースです。",
        "",
        "## ライブラリ",
        "",
        "| 名前 | 版 | ライセンス | 根拠 | dylib | ソース | sha256 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for lib in manifest["libraries"]:
        lines.append(
            f"| {lib['name']} | {lib['version']} | {lib['license']} | {lib['evidence']} | {' '.join(lib['dylibs'])} "
            f"| {lib['url']} | `{lib['sha256']}` |"
        )
    lines += ["", "## Python パッケージ", "", "| 名前 | 版 | ソース |", "| --- | --- | --- |"]
    lines += [f"| {n} | {v} | {u} |" for n, v, u in packages]
    lines += ["", "## Python 本体", "", python_url, ""]
    (out / "SOURCES.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True, help="作成する .tar.gz")
    ap.add_argument("--check-only", action="store_true", help="dylib と一覧の照合だけを行う")
    args = ap.parse_args()

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    cv2_dir = Path(importlib.util.find_spec("cv2").origin).parent
    try:
        check_dylibs(manifest, cv2_dir / ".dylibs")
        if args.check_only:
            print(f"{sum(len(x['dylibs']) for x in manifest['libraries'])} 個の dylib が一覧と一致しました")
            return 0

        top = args.output.name.removesuffix(".tar.gz")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / top
            for lib in manifest["libraries"]:
                fetch_library(lib, out)
            packages = fetch_python_packages(manifest, out)

            python_url = "https://www.python.org/ftp/python/{0}/Python-{0}.tgz".format(platform.python_version())
            download(python_url, out / "cpython" / file_name(python_url))

            (out / "app").mkdir()
            archive = out / "app" / "document_OCR.tar.gz"
            subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(archive), "HEAD"], cwd=ROOT, check=True)

            write_index(manifest, packages, python_url, out)

            args.output.parent.mkdir(parents=True, exist_ok=True)
            # 中身はほとんど圧縮済みなので、圧縮率より速度を優先する
            with tarfile.open(args.output, "w:gz", compresslevel=1) as tar:
                tar.add(out, arcname=top)
    except (SourceError, subprocess.CalledProcessError) as e:
        print(f"ソース一式を作成できません: {e}", file=sys.stderr)
        return 1
    print(f"{args.output}: {args.output.stat().st_size / 2**20:.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
