# PyInstaller の設定（全 OS 共通）: uv run pyinstaller document_ocr.spec
# 出力は dist/document_OCR/。configs/ と settings.yaml はビルド後に実行ファイルの隣へコピーする
# （.github/workflows/build.yml を参照）。
from PyInstaller.utils.hooks import collect_data_files


def analysis(script):
    a = Analysis(
        [script],
        # RapidOCR の設定とモデルは __file__ 基準で読まれるので、パッケージ内の相対位置のまま同梱する
        datas=collect_data_files("rapidocr"),
        excludes=["pytest", "reportlab"],
    )
    # Windows の FFmpeg は実行時に読み込まれる別 DLL で、動画機能を使わないので同梱しない
    a.binaries = [b for b in a.binaries if "opencv_videoio_ffmpeg" not in b[0]]
    return a


def exe(a, name, console):
    # CLI の日本語の出力が、リダイレクト先の文字コードで落ちないようにする
    options = [("X utf8_mode=1", None, "OPTION")] if console else []
    return EXE(PYZ(a.pure), a.scripts, options, exclude_binaries=True, name=name, console=console)


gui = analysis("ocr_gui.py")
cli = analysis("inspection_ocr.py")

COLLECT(
    exe(gui, "document_OCR", console=False),
    gui.binaries,
    gui.datas,
    exe(cli, "document_OCR_cli", console=True),
    cli.binaries,
    cli.datas,
    name="document_OCR",
)
