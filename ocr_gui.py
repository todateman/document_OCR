"""検査成績書 OCR 良否判定の GUI。

- 判定タブ  : 様式と画像を選んで判定し、結果を一覧と画像で確認する
- 様式設定タブ: 様式の設定ファイル（configs/*.yaml）を作成・編集する。
               汎用様式は基準画像の上に枠を描いて読み取り項目を指定し、その場で試し読みできる

起動:
    uv run ocr_gui.py      （または OCR判定.bat をダブルクリック）
"""

from __future__ import annotations

import copy
import queue
import shutil
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter import font as tkfont

from PIL import Image, ImageTk

import inspection_ocr
import template_ocr
from ocr_common import (
    CHECK,
    CONFIG_DIR,
    NG,
    OK,
    SETTINGS_PATH,
    LayoutError,
    Row,
    config_name,
    load_config,
    load_image_gray,
    load_settings,
    read_yaml,
    setup_engine,
    write_yaml,
)

IMAGE_TYPES = [("画像", "*.png *.jpg *.jpeg *.gif *.bmp *.tif *.tiff"), ("すべて", "*.*")]
ENGINES = {"設定ファイルに従う": None, "indra-ocr": "indra", "Tesseract": "tesseract"}
ENGINE_LABELS = {"indra": "indra-ocr", "tesseract": "Tesseract"}

STATUS_COLOR = {OK: "#2e7d32", NG: "#c62828", CHECK: "#e65100"}
STATUS_BG = {OK: "", NG: "#ffcdd2", CHECK: "#ffe0b2"}
VERDICT_BG = {"合格": "#c8e6c9", "不合格": "#ffcdd2", "要目視確認": "#ffe0b2", "読み取りエラー": "#e0e0e0"}
KIND_COLOR = {"text": "#1565c0", "number": "#6a1b9a", "table": "#00838f", "mark": "#ef6c00"}
KIND_BY_LABEL = {v: k for k, v in template_ocr.KINDS.items()}

NEOMAG_COLUMNS = {"length": "長さ(mm)", "width": "幅(mm)", "height": "高さ(mm)", "gauss": "表面磁束密度(Gauss)"}
NEOMAG_MAGNETIC = ["Br", "Hcb", "Hcj", "BHmax"]


# ---------------------------------------------------------------- 共通部品

def run_in_background(widget: tk.Misc, func, on_done, on_error=None) -> None:
    """func を別スレッドで実行し、終わったら画面のスレッドで on_done(結果) を呼ぶ。"""
    q: queue.Queue = queue.Queue()

    def work():
        try:
            q.put((True, func()))
        except BaseException as e:  # SystemExit（indra-ocr 未導入）も画面に出す
            q.put((False, e))

    def poll():
        try:
            ok, value = q.get_nowait()
        except queue.Empty:
            widget.after(100, poll)
            return
        if ok:
            on_done(value)
        elif on_error:
            on_error(value)
        else:
            show_error(value)

    threading.Thread(target=work, daemon=True).start()
    widget.after(100, poll)


def show_error(e: BaseException) -> None:
    if isinstance(e, (LayoutError, FileNotFoundError)):
        messagebox.showerror("エラー", str(e))
    elif isinstance(e, SystemExit):
        messagebox.showerror("エラー", "indra-ocr が見つかりません。uv の環境（OCR判定.bat）から起動してください。")
    else:
        messagebox.showerror("エラー", "".join(traceback.format_exception_only(type(e), e)))


def config_paths() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.yaml"))


def merged_config(cfg: dict, path: Path | None) -> dict:
    """編集中の設定に共通設定を合わせる（load_config と同じ規則）。"""
    merged = {**load_settings(), **{k: v for k, v in cfg.items() if v is not None}}
    merged["_dir"] = str((path.parent if path else CONFIG_DIR).resolve())
    return merged


def to_float(s: str) -> float | None:
    """入力欄の数値。整数なら int にする（設定ファイルに 2.0 ではなく 2 と書くため）。"""
    s = s.strip()
    if not s:
        return None
    v = float(s)
    return int(v) if v.is_integer() else v


def fmt_num(v) -> str:
    if v is None or v == "":
        return ""
    v = float(v)
    return str(int(v)) if v.is_integer() else f"{v:g}"


def split_list(s: str) -> list[str]:
    return [x.strip() for x in s.replace("、", ",").split(",") if x.strip()]


class ImageView(ttk.Frame):
    """スクロール・拡大縮小できる画像表示。座標変換と枠の描画を受け持つ。"""

    def __init__(self, master, on_redraw=None):
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg="#9e9e9e", highlightthickness=0, cursor="crosshair")
        xs = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        ys = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xs.set, yscrollcommand=ys.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.image: Image.Image | None = None
        self.zoom = 1.0
        self.photo = None
        self.on_redraw = on_redraw
        self._fit_pending = False
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-e.delta // 120 * 3, "units"))
        self.canvas.bind("<Shift-MouseWheel>", lambda e: self.canvas.xview_scroll(-e.delta // 120 * 3, "units"))
        self.canvas.bind("<Control-MouseWheel>", lambda e: self.set_zoom(self.zoom * (1.2 if e.delta > 0 else 1 / 1.2)))
        self.canvas.bind("<Configure>", self._on_configure)

    def set_image(self, image: Image.Image | None, fit: bool = True) -> None:
        self.image = image
        if fit:
            self._fit_pending = True
            self.fit()
        else:
            self.render()

    def _on_configure(self, _e):
        if self._fit_pending:
            self.fit()

    def fit(self) -> None:
        if not self.image:
            self.render()
            return
        w = self.canvas.winfo_width()
        if w <= 1:
            return  # まだ表示前。<Configure> で改めて合わせる
        self._fit_pending = False
        self.zoom = max(min(w / self.image.width, 4), 0.05)
        self.render()

    def set_zoom(self, z: float) -> None:
        self.zoom = max(min(z, 4), 0.05)
        self.render()

    def render(self) -> None:
        self.canvas.delete("all")
        if not self.image:
            self.canvas.configure(scrollregion=(0, 0, 0, 0))
            return
        w, h = round(self.image.width * self.zoom), round(self.image.height * self.zoom)
        resample = Image.LANCZOS if self.zoom < 1 else Image.BILINEAR
        self.photo = ImageTk.PhotoImage(self.image.resize((max(w, 1), max(h, 1)), resample))
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo, tags="img")
        self.canvas.configure(scrollregion=(0, 0, w, h))
        if self.on_redraw:
            self.on_redraw()

    def to_image(self, event) -> tuple[int, int]:
        return (round(self.canvas.canvasx(event.x) / self.zoom), round(self.canvas.canvasy(event.y) / self.zoom))

    def to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return x * self.zoom, y * self.zoom

    def rect(self, box, **kw) -> int:
        x0, y0 = self.to_canvas(box[0], box[1])
        x1, y1 = self.to_canvas(box[2], box[3])
        return self.canvas.create_rectangle(x0, y0, x1, y1, **kw)

    def see(self, box) -> None:
        """枠が見える位置までスクロールする。"""
        if not self.image:
            return
        w, h = self.image.width * self.zoom, self.image.height * self.zoom
        x0, y0 = self.to_canvas(box[0], box[1])
        x1, y1 = self.to_canvas(box[2], box[3])
        vx0, vx1 = (f * w for f in self.canvas.xview())
        vy0, vy1 = (f * h for f in self.canvas.yview())
        if not (vx0 <= x0 and x1 <= vx1):
            self.canvas.xview_moveto(max((x0 + x1) / 2 - (vx1 - vx0) / 2, 0) / w)
        if not (vy0 <= y0 and y1 <= vy1):
            self.canvas.yview_moveto(max((y0 + y1) / 2 - (vy1 - vy0) / 2, 0) / h)


class ScrollFrame(ttk.Frame):
    """縦スクロールできる入れ物。中身は self.inner に置く。"""

    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        win = self.canvas.create_window(0, 0, anchor="nw", window=self.inner)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(win, width=e.width))
        # 入力欄の上でもホイールでスクロールできるよう、マウスがこの中にある間だけ全体に割り当てる
        self.bind("<Enter>", lambda e: self.bind_all("<MouseWheel>", self._wheel))
        self.bind("<Leave>", lambda e: self.unbind_all("<MouseWheel>"))

    def _wheel(self, e) -> None:
        if self.winfo_exists() and self.inner.winfo_height() > self.canvas.winfo_height():
            self.canvas.yview_scroll(-e.delta // 120 * 2, "units")


class ResultTable(ttk.Frame):
    """読み取り結果の一覧（項目 / 読取値 / 判定 / 理由）。NG・要確認は色付け。"""

    def __init__(self, master, on_select=None):
        super().__init__(master)
        top = ttk.Frame(self)
        top.pack(fill="x")
        self.only_problems = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="NG・要確認だけ表示", variable=self.only_problems, command=self._fill).pack(side="left")
        self.count = ttk.Label(top, text="")
        self.count.pack(side="right")
        self.tree = ttk.Treeview(self, columns=("text", "status", "note"), show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="項目")
        self.tree.heading("text", text="読取値")
        self.tree.heading("status", text="判定")
        self.tree.heading("note", text="理由・メモ")
        self.tree.column("#0", width=220, stretch=False)
        self.tree.column("text", width=150, stretch=False)
        self.tree.column("status", width=70, stretch=False, anchor="center")
        self.tree.column("note", width=300)
        for st, bg in STATUS_BG.items():
            if bg:
                self.tree.tag_configure(st, background=bg)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.rows: list[Row] = []
        self.on_select = on_select
        self.tree.bind("<<TreeviewSelect>>", self._selected)

    def set_rows(self, rows: list[Row]) -> None:
        self.rows = rows
        self._fill()

    def _fill(self) -> None:
        self.tree.delete(*self.tree.get_children())
        n_ng = sum(r.status == NG for r in self.rows)
        n_chk = sum(r.status == CHECK for r in self.rows)
        self.count.configure(text=f"全 {len(self.rows)} 項目　NG {n_ng}　要確認 {n_chk}" if self.rows else "")
        for i, r in enumerate(self.rows):
            if self.only_problems.get() and r.status == OK:
                continue
            self.tree.insert("", "end", iid=str(i), text=r.item, values=(r.text or "(空)", r.status, r.note), tags=(r.status,))

    def _selected(self, _e=None) -> None:
        sel = self.tree.selection()
        if sel and self.on_select:
            self.on_select(self.rows[int(sel[0])])


# ---------------------------------------------------------------- 判定タブ

class RunTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=6)
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Label(bar, text="様式:").pack(side="left")
        self.config_var = tk.StringVar()
        self.config_box = ttk.Combobox(bar, textvariable=self.config_var, state="readonly", width=32)
        self.config_box.pack(side="left", padx=(2, 12))
        ttk.Label(bar, text="OCR エンジン:").pack(side="left")
        self.engine_var = tk.StringVar(value=next(iter(ENGINES)))
        ttk.Combobox(bar, textvariable=self.engine_var, values=list(ENGINES), state="readonly", width=16).pack(
            side="left", padx=(2, 12)
        )
        ttk.Button(bar, text="画像を追加…", command=self.add_images).pack(side="left")
        ttk.Button(bar, text="一覧を空にする", command=self.clear).pack(side="left", padx=4)
        self.run_button = ttk.Button(bar, text="▶ 判定実行", command=self.run)
        self.run_button.pack(side="left", padx=12)
        self.progress = ttk.Label(bar, text="")
        self.progress.pack(side="left")

        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, pady=(6, 0))
        left = ttk.Frame(pane)
        pane.add(left, weight=3)
        self.files = ttk.Treeview(left, columns=("verdict",), height=6, selectmode="browse")
        self.files.heading("#0", text="画像ファイル")
        self.files.heading("verdict", text="総合判定")
        self.files.column("#0", width=380)
        self.files.column("verdict", width=110, anchor="center", stretch=False)
        for v, bg in VERDICT_BG.items():
            self.files.tag_configure(v, background=bg)
        self.files.pack(fill="x")
        self.files.bind("<<TreeviewSelect>>", lambda e: self.show_selected())
        self.verdict = tk.Label(left, text="画像を追加して「判定実行」を押してください", font=("", 18, "bold"), pady=8)
        self.verdict.pack(fill="x", pady=6)
        self.verdict_bg = self.verdict.cget("bg")
        self.table = ResultTable(left, on_select=self.highlight)
        self.table.pack(fill="both", expand=True)
        self.view = ImageView(pane, on_redraw=self.draw_boxes)
        pane.add(self.view, weight=2)

        self.paths: list[Path] = []
        self.results: dict[str, tuple] = {}  # パス → (report または例外, 表示画像)
        self.current: str | None = None
        self.selected_row: Row | None = None
        self.refresh_configs()

    def refresh_configs(self, select: Path | None = None) -> None:
        self.config_map = {f"{config_name(p)}  ({p.name})": p for p in config_paths()}
        names = list(self.config_map)
        self.config_box.configure(values=names)
        if select:
            for k, p in self.config_map.items():
                if p.resolve() == select.resolve():
                    self.config_var.set(k)
        if self.config_var.get() not in self.config_map and names:
            self.config_var.set(names[0])

    def add_images(self) -> None:
        for p in filedialog.askopenfilenames(title="検査成績書の画像を選択", filetypes=IMAGE_TYPES):
            p = Path(p)
            if str(p) not in {str(x) for x in self.paths}:
                self.paths.append(p)
                self.files.insert("", "end", iid=str(p), text=p.name, values=("未判定",))
        if self.paths and not self.files.selection():
            self.files.selection_set(str(self.paths[0]))

    def clear(self) -> None:
        self.paths.clear()
        self.results.clear()
        self.files.delete(*self.files.get_children())
        self.table.set_rows([])
        self.view.set_image(None)
        self.verdict.configure(text="", bg=self.verdict_bg)

    def run(self) -> None:
        if not self.paths:
            messagebox.showinfo("判定", "先に「画像を追加…」で検査成績書の画像を選んでください。")
            return
        cfg_path = self.config_map.get(self.config_var.get())
        if not cfg_path:
            messagebox.showinfo("判定", "様式を選んでください。")
            return
        try:
            cfg = load_config(cfg_path)
            engine = ENGINES[self.engine_var.get()] or cfg.get("engine", "tesseract")
            setup_engine(engine, cfg)
        except Exception as e:
            show_error(e)
            return
        paths = list(self.paths)
        self.run_button.configure(state="disabled")
        self.progress.configure(text=f"判定中… (0/{len(paths)})")

        def step(i: int) -> None:
            if i >= len(paths):
                self.run_button.configure(state="normal")
                self.progress.configure(text=f"完了 ({len(paths)} 件)")
                return
            path = paths[i]

            def work():
                try:
                    rep = inspection_ocr.analyze(path, cfg, engine)
                except LayoutError as e:
                    rep = e
                img = rep.image if getattr(rep, "image", None) is not None else load_image_gray(path)
                return rep, Image.fromarray(img)

            def done(result):
                self.results[str(path)] = result
                rep = result[0]
                verdict = "読み取りエラー" if isinstance(rep, Exception) else rep.verdict
                if self.files.exists(str(path)):
                    self.files.item(str(path), values=(verdict,), tags=(verdict,))
                if i == 0 or self.current == str(path) or not self.files.selection():
                    self.files.selection_set(str(path))
                self.show_selected()
                self.progress.configure(text=f"判定中… ({i + 1}/{len(paths)})")
                step(i + 1)

            def failed(e):
                self.run_button.configure(state="normal")
                self.progress.configure(text="中断しました")
                show_error(e)

            run_in_background(self, work, done, failed)

        step(0)

    def show_selected(self) -> None:
        sel = self.files.selection()
        if not sel:
            return
        self.current = sel[0]
        self.selected_row = None
        result = self.results.get(sel[0])
        if not result:
            self.verdict.configure(text="未判定", bg=self.verdict_bg)
            self.table.set_rows([])
            try:
                self.view.set_image(Image.open(sel[0]).convert("L"))
            except OSError:
                self.view.set_image(None)
            return
        rep, img = result
        if isinstance(rep, Exception):
            self.verdict.configure(text=f"読み取りエラー: {rep}", bg=VERDICT_BG["読み取りエラー"], wraplength=600)
            self.table.set_rows([])
        else:
            self.verdict.configure(text=f"総合判定: {rep.verdict}", bg=VERDICT_BG[rep.verdict])
            self.table.set_rows(rep.rows())
        self.view.set_image(img)

    def draw_boxes(self) -> None:
        for r in self.table.rows:
            if r.box:
                selected = r is self.selected_row
                self.view.rect(r.box, outline=STATUS_COLOR.get(r.status, "gray"), width=4 if selected else 2)

    def highlight(self, row: Row) -> None:
        self.selected_row = row
        self.view.render()
        if row.box:
            self.view.see(row.box)


# ---------------------------------------------------------------- 様式設定タブ

class Form(ttk.Frame):
    """ラベル + 入力欄を縦に並べる簡易フォーム。値が変わると on_change を呼ぶ。"""

    def __init__(self, master, on_change):
        super().__init__(master)
        self.on_change = on_change
        self.row = 0
        self.columnconfigure(1, weight=1)

    def _place(self, label: str, widget, hint: str = "") -> None:
        ttk.Label(self, text=label).grid(row=self.row, column=0, sticky="w", padx=(0, 6), pady=2)
        widget.grid(row=self.row, column=1, sticky="ew", pady=2)
        if hint:
            self.row += 1
            ttk.Label(self, text=hint, foreground="#666", wraplength=420).grid(row=self.row, column=1, sticky="w")
        self.row += 1

    def entry(self, label: str, value, setter, hint: str = "") -> tk.StringVar:
        var = tk.StringVar(value="" if value is None else str(value))
        e = ttk.Entry(self, textvariable=var)
        self._place(label, e, hint)
        self._watch(var, setter, e)
        return var

    def combo(self, label: str, value: str, values: list[str], setter, hint: str = "") -> tk.StringVar:
        var = tk.StringVar(value=value)
        c = ttk.Combobox(self, textvariable=var, values=values, state="readonly")
        self._place(label, c, hint)
        self._watch(var, setter, c)
        return var

    def check(self, label: str, value: bool, setter) -> tk.BooleanVar:
        var = tk.BooleanVar(value=bool(value))
        c = ttk.Checkbutton(self, text=label, variable=var)
        c.grid(row=self.row, column=1, sticky="w", pady=2)
        self.row += 1
        var.trace_add("write", lambda *_: (setter(var.get()), self.on_change()))
        return var

    def _watch(self, var: tk.StringVar, setter, widget) -> None:
        def changed(*_):
            try:
                setter(var.get())
                widget.configure(style="TEntry" if isinstance(widget, ttk.Entry) else "TCombobox")
            except ValueError:
                if isinstance(widget, ttk.Entry):
                    widget.configure(style="Bad.TEntry")  # 数値でない入力は反映せず赤枠で示す
                return
            self.on_change()

        var.trace_add("write", changed)


def _setter(d: dict, key: str, conv=lambda s: s.strip() or None):
    def set_(s):
        v = conv(s)
        if v is None or v == []:
            d.pop(key, None)
        else:
            d[key] = v
    return set_


def _int_or_none(s: str) -> int | None:
    s = s.strip()
    return int(s) if s else None


class FieldPanel(ttk.Frame):
    """選択中の読み取り項目のプロパティ欄。"""

    def __init__(self, master, editor: EditorTab):
        super().__init__(master)
        self.editor = editor
        self.form: Form | None = None
        self.show(None)

    def show(self, index: int | None) -> None:
        for w in self.winfo_children():
            w.destroy()
        self.form = None
        if index is None:
            ttk.Label(self, text="項目を選ぶと、ここで名前や判定条件を設定できます。", foreground="#666").pack(anchor="w")
            return
        f = self.editor.fields[index]
        kind = f.get("kind", "text")
        changed = lambda: self.editor.field_changed(index)
        form = Form(self, changed)
        form.pack(fill="x")
        self.form = form
        self.name_var = form.entry("項目名", f.get("name", ""), lambda s: f.__setitem__("name", s.strip()))
        form.combo("種類", template_ocr.KINDS[kind], list(template_ocr.KINDS.values()),
                   lambda s: self.editor.change_kind(index, KIND_BY_LABEL[s]))

        if kind == "table":
            form.entry("行数", f.get("rows", 1), _setter(f, "rows", lambda s: max(int(s), 1)))
            form.entry("列数", f.get("cols", 1), _setter(f, "cols", lambda s: max(int(s), 1)))
            form.entry("行の名前", ", ".join(map(str, f.get("row_labels") or [])), _setter(f, "row_labels", split_list),
                       "カンマ区切り。省略すると 1, 2, 3 …")
            form.entry("列の名前", ", ".join(map(str, f.get("col_labels") or [])), _setter(f, "col_labels", split_list))
            skip = f.get("skip")
            form.entry("読まないセル", ", ".join(skip) if isinstance(skip, list) else (skip or ""),
                       _setter(f, "skip"), "行-列 で指定。例: 5-3, 5-6（斜線や矢印の欄など）")
            form.combo("セルの種類", "文字" if f.get("cell_kind") == "text" else "数値", ["数値", "文字"],
                       lambda s: self.editor.change_cell_kind(index, "text" if s == "文字" else "number"))
            rule_kind = f.get("cell_kind", "number")
        else:
            rule_kind = kind

        if rule_kind == "number":
            ttk.Separator(form).grid(row=form.row, column=0, columnspan=2, sticky="ew", pady=4)
            form.row += 1
            form.entry("下限", fmt_num(f.get("min")), _setter(f, "min", to_float), "空欄なら下限なし（境界値は合格）")
            form.entry("上限", fmt_num(f.get("max")), _setter(f, "max", to_float), "空欄なら上限なし。下限・上限とも空欄なら読み取りのみ")
            form.entry("小数桁数", f.get("decimals", ""), _setter(f, "decimals", _int_or_none),
                       "桁数が違えば読み誤りとみなし要確認。空欄なら問わない")
            form.entry("単位", f.get("unit", ""), _setter(f, "unit"), "値の後ろに付く単位（例: g, mm, %）。取り除いて数値として読む")
            form.check("空欄でもよい", f.get("allow_blank"), lambda v: f.__setitem__("allow_blank", v) if v else f.pop("allow_blank", None))
        elif rule_kind == "text":
            ttk.Separator(form).grid(row=form.row, column=0, columnspan=2, sticky="ew", pady=4)
            form.row += 1
            form.entry("期待する値", ", ".join(map(str, f.get("expect") or [])), _setter(f, "expect", split_list),
                       "一致すれば OK。カンマ区切りで複数可（例: OK）。空欄なら読み取りのみ")
            form.entry("NG とする値", ", ".join(map(str, f.get("ng_values") or [])), _setter(f, "ng_values", split_list),
                       "一致すれば NG（例: NG, 有）。どちらにも当たらなければ要確認")
            form.check("空欄でもよい", f.get("allow_blank"), lambda v: f.__setitem__("allow_blank", v) if v else f.pop("allow_blank", None))
        elif rule_kind == "mark":
            self._mark_panel(form, index, f)

    def _mark_panel(self, form: Form, index: int, f: dict) -> None:
        ttk.Separator(form).grid(row=form.row, column=0, columnspan=2, sticky="ew", pady=4)
        form.row += 1
        ttk.Label(form, text="選択肢").grid(row=form.row, column=0, sticky="nw")
        box = ttk.Frame(form)
        box.grid(row=form.row, column=1, sticky="ew")
        form.row += 1
        lb = tk.Listbox(box, height=4, exportselection=False)
        lb.pack(fill="x")
        for o in f.get("options") or []:
            lb.insert("end", o.get("label", ""))
        if self.editor.sel_option is not None and self.editor.sel_option < lb.size():
            lb.selection_set(self.editor.sel_option)
        lb.bind("<<ListboxSelect>>", lambda e: lb.curselection() and self.editor.select(index, lb.curselection()[0]))
        btns = ttk.Frame(box)
        btns.pack(fill="x", pady=2)
        ttk.Button(btns, text="＋選択肢（枠を描く）", command=lambda: self.editor.start_add_option(index)).pack(side="left")
        ttk.Button(btns, text="名前変更", command=lambda: self.editor.rename_option(index, lb.curselection())).pack(side="left", padx=2)
        ttk.Button(btns, text="削除", command=lambda: self.editor.delete_option(index, lb.curselection())).pack(side="left")
        labels = [str(o.get("label", "")) for o in f.get("options") or []]
        expect = (f.get("expect") or [""])[0]
        form.combo("期待する選択肢", expect if expect in labels else "(指定なし)", ["(指定なし)", *labels],
                   lambda s: f.__setitem__("expect", [s]) if s != "(指定なし)" else f.pop("expect", None),
                   "マークされた選択肢がこれと違えば NG")
        form.entry("マークの閾値", fmt_num(f.get("mark_threshold")), _setter(f, "mark_threshold", to_float),
                   f"スコアがこれ以上の選択肢をマークありとみなす（空欄なら {template_ocr.MARK_THRESHOLD}）。"
                   "試し読みで表示されるスコアを見て調整")

    def focus_name(self) -> None:
        if self.form:
            for w in self.form.grid_slaves(row=0, column=1):
                w.focus_set()
                if isinstance(w, ttk.Entry):
                    w.select_range(0, "end")


class NeoMagForm(ttk.Frame):
    """NeoMag 専用様式の設定（規格値など）。"""

    def __init__(self, master, editor: EditorTab):
        super().__init__(master, padding=8)
        self.editor = editor

    def load(self, cfg: dict) -> None:
        for w in self.winfo_children():
            w.destroy()
        cols = cfg.setdefault("columns", {})
        ttk.Label(self, text="計測表の各列", font=("", 11, "bold")).grid(row=0, column=0, columnspan=6, sticky="w")
        for j, h in enumerate(["列", "小数桁数", "規格", "", "下限", "上限"]):
            ttk.Label(self, text=h).grid(row=1, column=j, sticky="w", padx=4)
        for i, (key, label) in enumerate(NEOMAG_COLUMNS.items(), start=2):
            c = cols.setdefault(key, {})
            ttk.Label(self, text=label).grid(row=i, column=0, sticky="w", padx=4, pady=2)
            dec = tk.StringVar(value=str(c.get("decimals", "")))
            ttk.Spinbox(self, from_=0, to=6, width=5, textvariable=dec).grid(row=i, column=1, padx=4)
            spec = c.get("spec", "from_document")
            mode = tk.StringVar(value="doc" if not isinstance(spec, dict) else "fixed")
            lo = tk.StringVar(value=fmt_num(spec.get("min")) if isinstance(spec, dict) else "")
            hi = tk.StringVar(value=fmt_num(spec.get("max")) if isinstance(spec, dict) else "")
            ttk.Radiobutton(self, text="帳票の規格欄を読む", value="doc", variable=mode).grid(row=i, column=2, sticky="w")
            ttk.Radiobutton(self, text="数値で指定 →", value="fixed", variable=mode).grid(row=i, column=3, sticky="w")
            ttk.Entry(self, textvariable=lo, width=10).grid(row=i, column=4, padx=4)
            ttk.Entry(self, textvariable=hi, width=10).grid(row=i, column=5, padx=4)

            def apply(*_, c=c, dec=dec, mode=mode, lo=lo, hi=hi):
                try:
                    if dec.get().strip():
                        c["decimals"] = int(dec.get())
                    else:
                        c.pop("decimals", None)
                    if mode.get() == "doc":
                        c["spec"] = "from_document"
                    elif lo.get().strip() and hi.get().strip():
                        c["spec"] = {"min": float(lo.get()), "max": float(hi.get())}
                except ValueError:
                    return
                self.editor.mark_dirty()

            for v in (dec, mode, lo, hi):
                v.trace_add("write", apply)

        r = len(NEOMAG_COLUMNS) + 3
        ttk.Label(self, text="磁気特性値の規格（空欄の項目は読み取り表示のみ）", font=("", 11, "bold")).grid(
            row=r, column=0, columnspan=6, sticky="w", pady=(12, 0)
        )
        mag = cfg.get("magnetic") or {}
        cfg["magnetic"] = mag
        ttk.Label(self, text="下限").grid(row=r + 1, column=4, sticky="w", padx=4)
        ttk.Label(self, text="上限").grid(row=r + 1, column=5, sticky="w", padx=4)
        for i, key in enumerate(NEOMAG_MAGNETIC, start=r + 2):
            m = mag.get(key) or {}
            ttk.Label(self, text=key).grid(row=i, column=0, sticky="w", padx=4, pady=2)
            lo = tk.StringVar(value=fmt_num(m.get("min")))
            hi = tk.StringVar(value=fmt_num(m.get("max")))
            ttk.Entry(self, textvariable=lo, width=10).grid(row=i, column=4, padx=4)
            ttk.Entry(self, textvariable=hi, width=10).grid(row=i, column=5, padx=4)

            def apply_mag(*_, key=key, lo=lo, hi=hi):
                try:
                    if lo.get().strip() and hi.get().strip():
                        mag[key] = {"min": float(lo.get()), "max": float(hi.get())}
                    elif not lo.get().strip() and not hi.get().strip():
                        mag.pop(key, None)
                except ValueError:
                    return
                self.editor.mark_dirty()

            lo.trace_add("write", apply_mag)
            hi.trace_add("write", apply_mag)
        ttk.Label(
            self, foreground="#666", wraplength=800, justify="left",
            text="NeoMag 様式は罫線から表の位置を自動で検出するため、枠の指定はありません。"
                 "最大値・最小値・平均値の再計算照合と c/n の照合も自動で行います。",
        ).grid(row=r + 7, column=0, columnspan=6, sticky="w", pady=(12, 0))


class EditorTab(ttk.Frame):
    HANDLE = 7  # 角のつまみの大きさ（画面上の px）

    def __init__(self, master, app: App):
        super().__init__(master, padding=6)
        self.app = app
        self.path: Path | None = None
        self.cfg: dict = {}
        self.dirty = False
        self.ref_gray = None
        self.sel: int | None = None
        self.sel_option: int | None = None
        self.mode: tuple | None = None  # ("add", kind) / ("add_option", index)
        self.drag: dict | None = None
        self.results: list[Row] = []

        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Button(bar, text="新規作成…", command=self.new).pack(side="left")
        ttk.Button(bar, text="開く…", command=self.open).pack(side="left", padx=2)
        ttk.Button(bar, text="保存", command=self.save).pack(side="left")
        ttk.Button(bar, text="名前を付けて保存…", command=self.save_as).pack(side="left", padx=2)
        self.title = ttk.Label(bar, text="", font=("", 10, "bold"))
        self.title.pack(side="left", padx=12)

        self.body = ttk.Frame(self)
        self.body.pack(fill="both", expand=True, pady=(6, 0))
        self.status = ttk.Label(self, text="", foreground="#1565c0")
        self.status.pack(fill="x", pady=(4, 0))
        self.show_empty()

    # ---- 表示の切り替え

    def show_empty(self) -> None:
        for w in self.body.winfo_children():
            w.destroy()
        ttk.Label(
            self.body, justify="left",
            text="「開く…」で既存の様式（configs フォルダの .yaml）を開くか、\n"
                 "「新規作成…」で新しい様式を作ってください。\n\n"
                 "新規作成では、まず記入済みの検査成績書の画像（基準画像）を選び、\n"
                 "その上に読み取りたい欄の枠を描いていきます。",
        ).pack(anchor="nw", pady=20, padx=20)

    def build(self) -> None:
        for w in self.body.winfo_children():
            w.destroy()
        self.sel = self.sel_option = None
        self.results = []
        self.mode = None
        common = ttk.LabelFrame(self.body, text="様式の基本設定", padding=6)
        common.pack(fill="x")
        self.base_settings(common)

        if self.cfg.get("type", "neomag") != "template":
            nf = NeoMagForm(self.body, self)
            nf.pack(fill="both", expand=True)
            nf.load(self.cfg)
            self.set_status("")
            return

        ref = template_ocr.reference_path(merged_config(self.cfg, self.path))
        try:
            self.ref_gray = template_ocr.load_page(ref)
        except OSError:
            self.ref_gray = None
            messagebox.showwarning("基準画像", f"基準画像を開けません: {ref}")

        pane = ttk.PanedWindow(self.body, orient="horizontal")
        pane.pack(fill="both", expand=True, pady=(6, 0))
        self.view = ImageView(pane, on_redraw=self.draw)
        pane.add(self.view, weight=3)
        if self.ref_gray is not None:
            self.view.set_image(Image.fromarray(self.ref_gray))
        c = self.view.canvas
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Motion>", self.on_motion)
        self.winfo_toplevel().bind("<Escape>", lambda e: self.cancel_mode())
        c.bind("<Delete>", lambda e: self.delete_field())
        c.bind("<Enter>", lambda e: c.focus_set())

        side_pane = ttk.PanedWindow(pane, orient="vertical")
        pane.add(side_pane, weight=2)
        side = ttk.Frame(side_pane, padding=(6, 0, 0, 0))
        side_pane.add(side, weight=1)
        adds = ttk.LabelFrame(side, text="項目を追加（ボタンを押してから画像上をドラッグ）", padding=4)
        adds.pack(fill="x")
        for kind, label in template_ocr.KINDS.items():
            ttk.Button(adds, text=f"＋{label}", command=lambda k=kind: self.start_add(k)).pack(side="left", padx=2)
        ttk.Button(adds, text="－", width=3, command=lambda: self.view.set_zoom(self.view.zoom / 1.25)).pack(side="right")
        ttk.Button(adds, text="＋", width=3, command=lambda: self.view.set_zoom(self.view.zoom * 1.25)).pack(side="right")
        ttk.Button(adds, text="全体", width=5, command=self.view.fit).pack(side="right")

        lst = ttk.Frame(side)
        lst.pack(fill="both", expand=True, pady=4)
        self.tree = ttk.Treeview(lst, columns=("kind", "rule"), height=6, selectmode="browse")
        self.tree.heading("#0", text="項目名")
        self.tree.heading("kind", text="種類")
        self.tree.heading("rule", text="判定条件 / 試し読み")
        self.tree.column("#0", width=170)
        self.tree.column("kind", width=60, stretch=False)
        self.tree.column("rule", width=200)
        for st, bg in STATUS_BG.items():
            if bg:
                self.tree.tag_configure(st, background=bg)
        sb = ttk.Scrollbar(lst, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<Delete>", lambda e: self.delete_field())
        ops = ttk.Frame(side)
        ops.pack(fill="x", side="bottom", before=lst)
        ttk.Button(ops, text="↑", width=3, command=lambda: self.move_field(-1)).pack(side="left")
        ttk.Button(ops, text="↓", width=3, command=lambda: self.move_field(1)).pack(side="left", padx=2)
        ttk.Button(ops, text="複製", command=self.duplicate_field).pack(side="left")
        ttk.Button(ops, text="削除", command=self.delete_field).pack(side="left", padx=2)
        ttk.Button(ops, text="試し読み（画像を選ぶ）…", command=lambda: self.test_read(ask=True)).pack(side="right")
        ttk.Button(ops, text="試し読み（基準画像）", command=self.test_read).pack(side="right", padx=2)

        nb = ttk.Notebook(side_pane)
        side_pane.add(nb, weight=1)
        prop_wrap = ScrollFrame(nb)
        nb.add(prop_wrap, text="選択中の項目")
        self.panel = FieldPanel(prop_wrap.inner, self)
        self.panel.pack(fill="both", expand=True, padx=6, pady=6)
        self.result_table = ResultTable(nb, on_select=self.on_result_select)
        nb.add(self.result_table, text="試し読みの結果")
        self.notebook = nb
        self.refresh_tree()
        self.set_status("「＋文字」などのボタンを押してから、基準画像の上をドラッグして枠を描きます。枠はドラッグで移動、角のつまみで大きさを変えられます。")

    def base_settings(self, frame: ttk.Frame) -> None:
        """様式名・エンジン・最小信頼度（・説明）を 1〜2 行で並べる。"""
        cfg = self.cfg
        name = tk.StringVar(value=cfg.get("name", ""))
        eng = tk.StringVar(value=ENGINE_LABELS.get(cfg.get("engine", "indra"), cfg.get("engine", "")))
        conf = tk.StringVar(value=fmt_num(cfg.get("min_confidence", 60)))
        ttk.Label(frame, text="様式名").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=name, width=36).grid(row=0, column=1, sticky="ew", padx=(4, 16))
        ttk.Label(frame, text="OCR エンジン").grid(row=0, column=2, sticky="w")
        ttk.Combobox(frame, textvariable=eng, values=list(ENGINE_LABELS.values()), state="readonly", width=12).grid(
            row=0, column=3, padx=(4, 16))
        ttk.Label(frame, text="最小信頼度").grid(row=0, column=4, sticky="w")
        conf_entry = ttk.Entry(frame, textvariable=conf, width=6)
        conf_entry.grid(row=0, column=5, padx=4)
        ttk.Label(frame, foreground="#666",
                  text="日本語の欄があれば indra-ocr を推奨。OCR の信頼度 (0〜100) が最小信頼度未満の値は要確認").grid(
            row=1, column=1, columnspan=5, sticky="w")
        frame.columnconfigure(1, weight=1)

        def apply(*_):
            cfg["name"] = name.get().strip()
            cfg["engine"] = {v: k for k, v in ENGINE_LABELS.items()}.get(eng.get(), cfg.get("engine"))
            try:
                cfg["min_confidence"] = float(conf.get())
                conf_entry.configure(style="TEntry")
            except ValueError:
                conf_entry.configure(style="Bad.TEntry")
            self.mark_dirty()

        for v in (name, eng, conf):
            v.trace_add("write", apply)
        if cfg.get("type") == "template":
            ttk.Label(frame, text="説明").grid(row=2, column=0, sticky="nw", pady=(4, 0))
            t = tk.Text(frame, height=2, wrap="word", font=tkfont.nametofont("TkDefaultFont"))
            t.insert("1.0", cfg.get("description", ""))
            t.grid(row=2, column=1, columnspan=5, sticky="ew", pady=(4, 0))

            def apply_desc(_e=None):
                d = t.get("1.0", "end").strip()
                if d != (cfg.get("description") or ""):
                    if d:
                        cfg["description"] = d
                    else:
                        cfg.pop("description", None)
                    self.mark_dirty()

            t.bind("<KeyRelease>", apply_desc)

    @property
    def fields(self) -> list[dict]:
        return self.cfg.setdefault("fields", [])

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def mark_dirty(self) -> None:
        self.dirty = True
        name = self.path.name if self.path else "(未保存)"
        self.title.configure(text=f"{name} *")

    def set_clean(self) -> None:
        self.dirty = False
        self.title.configure(text=self.path.name if self.path else "")

    # ---- ファイル操作

    def confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        ans = messagebox.askyesnocancel("様式設定", "変更が保存されていません。保存しますか？")
        if ans is None:
            return False
        if ans:
            return self.save()
        return True

    def open(self, path: Path | None = None) -> None:
        if not self.confirm_discard():
            return
        if path is None:
            p = filedialog.askopenfilename(title="様式の設定ファイルを開く", initialdir=CONFIG_DIR,
                                           filetypes=[("設定ファイル", "*.yaml"), ("すべて", "*.*")])
            if not p:
                return
            path = Path(p)
        try:
            self.cfg = read_yaml(path)
        except Exception as e:
            show_error(e)
            return
        self.cfg.setdefault("type", "neomag")
        self.path = path
        self.build()
        self.set_clean()

    def new(self) -> None:
        if not self.confirm_discard():
            return
        img = filedialog.askopenfilename(title="基準画像（記入済みの検査成績書）を選択", filetypes=IMAGE_TYPES)
        if not img:
            return
        name = simpledialog.askstring("新規作成", "様式名（例: 電子天秤検査成績書）", parent=self)
        if not name:
            return
        p = filedialog.asksaveasfilename(title="設定ファイルの保存先", initialdir=CONFIG_DIR, initialfile=f"{name}.yaml",
                                         defaultextension=".yaml", filetypes=[("設定ファイル", "*.yaml")])
        if not p:
            return
        path = Path(p)
        img = Path(img)
        ref = path.with_suffix(img.suffix.lower())
        if img.resolve() != ref.resolve():
            shutil.copyfile(img, ref)  # 設定ファイルと同じフォルダに基準画像を置く
        self.cfg = {"type": "template", "name": name, "reference_image": ref.name,
                    "engine": "indra", "min_confidence": 60, "fields": []}
        self.path = path
        self.build()
        self.save()

    def save(self) -> bool:
        if self.path is None:
            return self.save_as()
        if self.cfg.get("type") == "template" and not self.check_fields():
            return False
        try:
            write_yaml(self.path, self.cfg)
        except OSError as e:
            show_error(e)
            return False
        self.set_clean()
        self.set_status(f"保存しました: {self.path}")
        self.app.run_tab.refresh_configs()
        return True

    def save_as(self) -> bool:
        if not self.cfg:
            return False
        p = filedialog.asksaveasfilename(title="名前を付けて保存", initialdir=CONFIG_DIR, defaultextension=".yaml",
                                         initialfile=self.path.name if self.path else "", filetypes=[("設定ファイル", "*.yaml")])
        if not p:
            return False
        new = Path(p)
        if self.cfg.get("type") == "template" and self.path and new.parent.resolve() != self.path.parent.resolve():
            # 基準画像は設定ファイルからの相対パスなので、別フォルダへ保存するときは画像もコピーする
            src = template_ocr.reference_path(merged_config(self.cfg, self.path))
            if src.is_file():
                shutil.copyfile(src, new.parent / src.name)
        self.path = new
        return self.save()

    def check_fields(self) -> bool:
        names = [f.get("name", "") for f in self.fields]
        empty = [i + 1 for i, n in enumerate(names) if not n]
        dup = sorted({n for n in names if n and names.count(n) > 1})
        msgs = []
        if empty:
            msgs.append(f"名前のない項目があります（{', '.join(map(str, empty))} 番目）")
        if dup:
            msgs.append(f"同じ名前の項目があります（{', '.join(dup)}）")
        for f in self.fields:
            if f.get("kind") == "mark" and len(f.get("options") or []) < 2:
                msgs.append(f"マーク「{f.get('name')}」の選択肢が 2 つ未満です")
        if msgs:
            return messagebox.askyesno("確認", "\n".join(msgs) + "\n\nこのまま保存しますか？")
        return True

    # ---- 項目一覧

    def summary(self, f: dict) -> str:
        kind = f.get("kind", "text")
        rk = f.get("cell_kind", "number") if kind == "table" else kind
        parts = []
        if kind == "table":
            parts.append(f"{f.get('rows', 1)}行×{f.get('cols', 1)}列")
        if rk == "number":
            lo, hi = fmt_num(f.get("min")), fmt_num(f.get("max"))
            parts.append(f"{lo}〜{hi}" if lo or hi else "読み取りのみ")
        elif rk == "text":
            parts.append("期待: " + "/".join(map(str, f["expect"])) if f.get("expect") else "読み取りのみ")
        elif rk == "mark":
            labels = [o.get("label", "") for o in f.get("options") or []]
            parts.append(" / ".join(labels) + (f"（期待: {f['expect'][0]}）" if f.get("expect") else ""))
        return "  ".join(parts)

    def refresh_tree(self) -> None:
        if not hasattr(self, "tree"):
            return
        self.tree.delete(*self.tree.get_children())
        for i, f in enumerate(self.fields):
            rows = [r for r in self.results if r.index == i]
            tag, rule = (), self.summary(f)
            if rows:
                st = CHECK if any(r.status == CHECK for r in rows) else NG if any(r.status == NG for r in rows) else OK
                tag = (st,)
                if len(rows) == 1:
                    rule = f"→ {rows[0].text or '(空)'} [{rows[0].status}]"
                else:
                    rule = f"→ OK {sum(r.status == OK for r in rows)} / NG {sum(r.status == NG for r in rows)} / 要確認 {sum(r.status == CHECK for r in rows)}"
            self.tree.insert("", "end", iid=str(i), text=f.get("name", ""),
                             values=(template_ocr.KINDS.get(f.get("kind", "text"), ""), rule), tags=tag)
        if self.sel is not None and self.sel < len(self.fields):
            self.tree.selection_set(str(self.sel))
            self.tree.see(str(self.sel))

    def on_tree_select(self, _e=None) -> None:
        s = self.tree.selection()
        if s and int(s[0]) != self.sel:
            self.select(int(s[0]), None, from_tree=True)
            box = self.fields[self.sel].get("box") or self._option_hull(self.fields[self.sel])
            if box:
                self.view.see(box)

    def select(self, index: int | None, option: int | None = None, from_tree: bool = False) -> None:
        changed = index != self.sel
        self.sel, self.sel_option = index, option
        if changed or option is not None:
            self.panel.show(index)
        if not from_tree and index is not None:
            self.tree.selection_set(str(index))
            self.tree.see(str(index))
        if index is not None:
            self.notebook.select(0)
        self.view.render()

    def field_changed(self, index: int) -> None:
        self.mark_dirty()
        f = self.fields[index]
        if self.tree.exists(str(index)):
            self.tree.item(str(index), text=f.get("name", ""),
                           values=(template_ocr.KINDS.get(f.get("kind", "text"), ""), self.summary(f)))
        self.view.render()

    def change_kind(self, index: int, kind: str) -> None:
        f = self.fields[index]
        if f.get("kind") == kind:
            return
        box = f.get("box") or self._option_hull(f)
        new = {"name": f.get("name", ""), "kind": kind}
        if kind == "mark":
            new["options"] = [{"label": "選択肢1", "box": list(box)}] if box else []
        else:
            new["box"] = list(box) if box else None
        if kind == "table" and box and self.ref_gray is not None:
            new["rows"], new["cols"] = template_ocr.detect_grid(self.ref_gray, box)
        self.fields[index] = new
        self.mark_dirty()
        self.after_idle(lambda: (self.refresh_tree(), self.panel.show(index), self.view.render()))

    def change_cell_kind(self, index: int, kind: str) -> None:
        f = self.fields[index]
        if f.get("cell_kind", "number") == kind:
            return
        for k in ("min", "max", "decimals", "unit", "expect", "ng_values"):
            f.pop(k, None)
        f["cell_kind"] = kind
        self.mark_dirty()
        self.after_idle(lambda: (self.refresh_tree(), self.panel.show(index)))

    def move_field(self, d: int) -> None:
        i = self.sel
        if i is None or not 0 <= i + d < len(self.fields):
            return
        self.fields[i], self.fields[i + d] = self.fields[i + d], self.fields[i]
        self.sel = i + d
        self.results = []
        self.mark_dirty()
        self.refresh_tree()
        self.panel.show(self.sel)

    def duplicate_field(self) -> None:
        if self.sel is None:
            return
        f = copy.deepcopy(self.fields[self.sel])
        f["name"] = f.get("name", "") + "（コピー）"
        off = 15
        if f.get("box"):
            f["box"] = [v + off for v in f["box"]]
        for o in f.get("options") or []:
            o["box"] = [v + off for v in o["box"]]
        self.fields.insert(self.sel + 1, f)
        self.results = []
        self.mark_dirty()
        self.select(self.sel + 1)
        self.refresh_tree()

    def delete_field(self) -> None:
        if self.sel is None:
            return
        f = self.fields[self.sel]
        if not messagebox.askyesno("削除", f"項目「{f.get('name', '')}」を削除しますか？"):
            return
        del self.fields[self.sel]
        self.results = []
        self.sel = None
        self.mark_dirty()
        self.refresh_tree()
        self.panel.show(None)
        self.view.render()

    # ---- マークの選択肢

    def start_add_option(self, index: int) -> None:
        self.mode = ("add_option", index)
        self.set_status("選択肢の枠を画像上でドラッグしてください（Esc で終了）")

    def rename_option(self, index: int, cur) -> None:
        if not cur:
            return
        o = self.fields[index]["options"][cur[0]]
        s = simpledialog.askstring("選択肢の名前", "選択肢の名前（例: 合格）", initialvalue=o.get("label", ""), parent=self)
        if s:
            old = o.get("label")
            o["label"] = s.strip()
            f = self.fields[index]
            if f.get("expect") == [old]:
                f["expect"] = [o["label"]]
            self.field_changed(index)
            self.panel.show(index)

    def delete_option(self, index: int, cur) -> None:
        if not cur:
            return
        f = self.fields[index]
        o = f["options"].pop(cur[0])
        if f.get("expect") == [o.get("label")]:
            f.pop("expect", None)
        self.sel_option = None
        self.field_changed(index)
        self.panel.show(index)

    @staticmethod
    def _option_hull(f: dict):
        boxes = [o["box"] for o in f.get("options") or [] if o.get("box")]
        if not boxes:
            return None
        return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]

    # ---- 描画

    def boxes(self):
        """(項目番号, 選択肢番号 or None, 枠) の一覧。"""
        for i, f in enumerate(self.fields):
            if f.get("kind") == "mark":
                for j, o in enumerate(f.get("options") or []):
                    if o.get("box"):
                        yield i, j, o["box"]
            elif f.get("box"):
                yield i, None, f["box"]

    def draw(self) -> None:
        v, c = self.view, self.view.canvas
        title = None
        for i, j, box in self.boxes():
            f = self.fields[i]
            selected = i == self.sel
            color = KIND_COLOR.get(f.get("kind", "text"), "black")
            v.rect(box, outline=color, width=3 if selected else 1.5, dash=() if selected else (4, 2))
            if f.get("kind") == "table" and self.ref_gray is not None:
                ys, xs = template_ocr.table_edges(self.ref_gray, box, f.get("rows", 1), f.get("cols", 1))
                for y in ys[1:-1]:
                    c.create_line(*v.to_canvas(box[0], y), *v.to_canvas(box[2], y), fill=color, dash=(2, 2))
                for x in xs[1:-1]:
                    c.create_line(*v.to_canvas(x, box[1]), *v.to_canvas(x, box[3]), fill=color, dash=(2, 2))
                for r, cc in template_ocr._skips(f.get("skip")):
                    if r < len(ys) and cc < len(xs):
                        c.create_line(*v.to_canvas(xs[cc - 1], ys[r - 1]), *v.to_canvas(xs[cc], ys[r]), fill="#c62828", width=2)
                        c.create_line(*v.to_canvas(xs[cc], ys[r - 1]), *v.to_canvas(xs[cc - 1], ys[r]), fill="#c62828", width=2)
            if j is not None:
                label = f["options"][j].get("label", "")
                x, y = v.to_canvas(box[0], box[1])
                self._label(x + 3, y + 3, label, color, "nw", 9)
            if selected and (self.sel_option is None or self.sel_option == j):
                for (hx, hy) in self.corners(box):
                    cx, cy = v.to_canvas(hx, hy)
                    h = self.HANDLE / 2
                    c.create_rectangle(cx - h, cy - h, cx + h, cy + h, fill="white", outline=color)
            if selected and j in (None, 0):
                title = (*v.to_canvas(box[0], box[1]), f.get("name", ""), color)
        # 試し読みの結果（セルごとの判定）
        for r in self.results:
            if r.box and r.status != OK:
                v.rect(r.box, outline=STATUS_COLOR[r.status], width=3)
        if self.drag and self.drag.get("rubber"):
            v.rect(self.drag["rubber"], outline="#d50000", width=2, dash=(3, 3))
        if title:  # 選択中の項目名は他の枠より手前に出す
            x, y, name, color = title
            self._label(x, y - 4, name, color, "sw", 10)

    def _label(self, x: float, y: float, text: str, color: str, anchor: str, size: int) -> None:
        """帳票の文字と重なっても読めるよう、白い背景付きで文字を書く。"""
        c = self.view.canvas
        t = c.create_text(x, y, text=text, anchor=anchor, fill=color, font=("", size, "bold"))
        x0, y0, x1, y1 = c.bbox(t)
        bg = c.create_rectangle(x0 - 2, y0 - 1, x1 + 2, y1 + 1, fill="white", outline=color)
        c.tag_raise(t, bg)

    @staticmethod
    def corners(box):
        x0, y0, x1, y1 = box
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    # ---- マウス操作

    def start_add(self, kind: str) -> None:
        if self.ref_gray is None:
            return
        self.mode = ("add", kind)
        self.set_status(f"「{template_ocr.KINDS[kind]}」の枠を画像上でドラッグしてください（Esc で中止）"
                        + ("。表は表全体を囲むと、罫線から行数・列数を自動で設定します" if kind == "table" else "")
                        + ("。まず 1 つ目の選択肢（例: 合格）の欄を囲みます" if kind == "mark" else ""))
        self.view.canvas.configure(cursor="tcross")

    def cancel_mode(self) -> None:
        self.mode = None
        self.drag = None
        self.view.canvas.configure(cursor="crosshair")
        self.set_status("")
        self.view.render()

    def hit(self, x: int, y: int):
        """(項目番号, 選択肢番号, 角の番号 or None)。角は選択中の枠だけ判定する。"""
        tol = self.HANDLE / self.view.zoom
        for i, j, box in self.boxes():
            if i == self.sel and (self.sel_option is None or self.sel_option == j):
                for k, (hx, hy) in enumerate(self.corners(box)):
                    if abs(x - hx) <= tol and abs(y - hy) <= tol:
                        return i, j, k
        best = None
        for i, j, box in self.boxes():
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                area = (box[2] - box[0]) * (box[3] - box[1])
                if best is None or area < best[0]:  # 重なっていれば小さい枠を優先
                    best = (area, i, j)
        return (best[1], best[2], None) if best else None

    def on_press(self, e) -> None:
        if self.ref_gray is None:
            return
        x, y = self.view.to_image(e)
        if self.mode:
            self.drag = {"start": (x, y), "rubber": [x, y, x, y]}
            return
        h = self.hit(x, y)
        if not h:
            self.select(None)
            self.panel.show(None)
            return
        i, j, corner = h
        if i != self.sel or j != self.sel_option:
            self.select(i, j)
        box = self.fields[i]["options"][j]["box"] if j is not None else self.fields[i]["box"]
        self.drag = {"start": (x, y), "orig": list(box), "corner": corner, "target": (i, j)}

    def on_drag(self, e) -> None:
        if not self.drag:
            return
        x, y = self.view.to_image(e)
        sx, sy = self.drag["start"]
        if "rubber" in self.drag:
            self.drag["rubber"] = [min(sx, x), min(sy, y), max(sx, x), max(sy, y)]
        else:
            i, j = self.drag["target"]
            o = self.drag["orig"]
            k = self.drag["corner"]
            if k is None:
                dx, dy = x - sx, y - sy
                nb = [o[0] + dx, o[1] + dy, o[2] + dx, o[3] + dy]
            else:
                xs = [o[0], o[2]]
                ys = [o[1], o[3]]
                xs[0 if k in (0, 3) else 1] = x
                ys[0 if k in (0, 1) else 1] = y
                nb = [min(xs), min(ys), max(xs), max(ys)]
            target = self.fields[i]["options"][j] if j is not None else self.fields[i]
            target["box"] = [int(v) for v in nb]
            self.drag["moved"] = True
        self.view.render()

    def on_release(self, e) -> None:
        d, self.drag = self.drag, None
        if not d:
            return
        if "rubber" in d:
            box = d["rubber"]
            if box[2] - box[0] < 5 or box[3] - box[1] < 5:
                self.view.render()
                return
            self.add_box([int(v) for v in box])
        elif d.get("moved"):
            self.results = []
            self.mark_dirty()
            self.refresh_tree()
            self.view.render()

    def on_motion(self, e) -> None:
        if self.mode or self.drag or self.ref_gray is None:
            return
        x, y = self.view.to_image(e)
        h = self.hit(x, y)
        cur = "crosshair"
        if h and h[2] is not None:
            cur = "size_nw_se" if h[2] in (0, 2) else "size_ne_sw"
        elif h:
            cur = "fleur"
        self.view.canvas.configure(cursor=cur)

    def add_box(self, box: list[int]) -> None:
        kind = self.mode[0]
        if kind == "add_option":
            i = self.mode[1]
            f = self.fields[i]
            n = len(f.get("options") or []) + 1
            label = simpledialog.askstring("選択肢の名前", f"{n} つ目の選択肢の名前（例: 不合格）", parent=self) or f"選択肢{n}"
            f.setdefault("options", []).append({"label": label.strip(), "box": box})
            self.sel_option = n - 1
            self.field_changed(i)
            self.panel.show(i)
            self.set_status("続けて次の選択肢の枠を描けます（Esc で終了）")
            return
        kind = self.mode[1]
        n = sum(f.get("kind") == kind for f in self.fields) + 1
        name = f"{template_ocr.KINDS[kind]}{n}"
        if kind == "mark":
            label = simpledialog.askstring("選択肢の名前", "この枠の選択肢の名前（例: 合格）", parent=self) or "選択肢1"
            f = {"name": name, "kind": "mark", "options": [{"label": label.strip(), "box": box}]}
        else:
            f = {"name": name, "kind": kind, "box": box}
        if kind == "table":
            f["rows"], f["cols"] = template_ocr.detect_grid(self.ref_gray, box)
            f["cell_kind"] = "number"
        self.fields.append(f)
        self.results = []
        self.mark_dirty()
        i = len(self.fields) - 1
        self.refresh_tree()
        self.select(i, 0 if kind == "mark" else None)
        if kind == "mark":
            self.mode = ("add_option", i)
            self.set_status("続けて 2 つ目の選択肢（例: 不合格）の欄を囲んでください（Esc で終了）")
        else:
            self.cancel_mode()
            self.set_status(f"「{name}」を追加しました。右の欄で項目名と判定条件を設定してください。")
            self.panel.focus_name()

    # ---- 試し読み

    def test_read(self, ask: bool = False) -> None:
        if self.cfg.get("type") != "template":
            return
        if ask:
            p = filedialog.askopenfilename(title="試し読みする画像", filetypes=IMAGE_TYPES)
            if not p:
                return
            img = Path(p)
        else:
            img = template_ocr.reference_path(merged_config(self.cfg, self.path))
        cfg = merged_config(copy.deepcopy(self.cfg), self.path)
        engine = cfg.get("engine", "indra")
        try:
            setup_engine(engine, cfg)
        except Exception as e:
            show_error(e)
            return
        self.set_status(f"試し読み中… ({img.name})")

        def done(rep):
            self.results = rep.rows()
            self.result_table.set_rows(self.results)
            self.notebook.select(1)
            self.refresh_tree()
            self.view.set_image(Image.fromarray(rep.image), fit=False)
            self.set_status(f"試し読み完了（{img.name}）: 総合判定 {rep.verdict}。"
                            "結果の行をクリックすると該当の枠に移動します。枠の色: 赤=NG, 橙=要確認")

        def failed(e):
            self.set_status("試し読みに失敗しました")
            show_error(e)

        run_in_background(self, lambda: template_ocr.analyze(img, cfg, engine), done, failed)

    def on_result_select(self, row: Row) -> None:
        if row.index is not None:
            self.sel = row.index
            self.sel_option = None
            self.tree.selection_set(str(row.index))
            self.tree.see(str(row.index))
            self.panel.show(row.index)
        self.view.render()
        if row.box:
            self.view.see(row.box)
            x0, y0 = self.view.to_canvas(row.box[0], row.box[1])
            x1, y1 = self.view.to_canvas(row.box[2], row.box[3])
            self.view.canvas.create_rectangle(x0 - 3, y0 - 3, x1 + 3, y1 + 3, outline="#d50000", width=3)


# ---------------------------------------------------------------- 共通設定

class SettingsDialog(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("共通設定（この PC で使う設定）")
        self.transient(master)
        self.resizable(True, False)
        self.data = load_settings()
        f = ttk.Frame(self, padding=10)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)
        self.vars = {}
        rows = [
            ("tesseract_cmd", "Tesseract の場所", "file", "空欄なら自動で探します"),
            ("tesseract_lang", "Tesseract の言語", None, "文字項目を読むときの言語（日本語は jpn の言語データが必要）"),
            ("indra_model_dir", "indra-ocr のモデル", "dir", "空欄なら初回にダウンロードして保存（ネットワークの無い PC ではフォルダを指定）"),
        ]
        for i, (key, label, browse, hint) in enumerate(rows):
            ttk.Label(f, text=label).grid(row=2 * i, column=0, sticky="w", padx=(0, 6), pady=(6, 0))
            v = tk.StringVar(value=self.data.get(key) or "")
            self.vars[key] = v
            ttk.Entry(f, textvariable=v, width=50).grid(row=2 * i, column=1, sticky="ew", pady=(6, 0))
            if browse:
                ttk.Button(f, text="参照…", command=lambda v=v, b=browse: self.browse(v, b)).grid(row=2 * i, column=2, padx=4, pady=(6, 0))
            ttk.Label(f, text=hint, foreground="#666").grid(row=2 * i + 1, column=1, sticky="w")
        btns = ttk.Frame(f)
        btns.grid(row=len(rows) * 2, column=0, columnspan=3, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="保存", command=self.save).pack(side="left")
        ttk.Button(btns, text="キャンセル", command=self.destroy).pack(side="left", padx=4)
        self.grab_set()

    def browse(self, var: tk.StringVar, kind: str) -> None:
        p = (filedialog.askdirectory(parent=self) if kind == "dir"
             else filedialog.askopenfilename(parent=self, filetypes=[("実行ファイル", "*.exe"), ("すべて", "*.*")]))
        if p:
            var.set(str(Path(p)))

    def save(self) -> None:
        for k, v in self.vars.items():
            self.data[k] = v.get().strip() or None
        write_yaml(SETTINGS_PATH, self.data, header="# この PC 共通の設定（GUI の「共通設定」で編集できます）\n")
        self.destroy()


# ---------------------------------------------------------------- アプリ

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("検査成績書 OCR 良否判定")
        self.geometry("1400x900")
        self.minsize(900, 600)
        default = tkfont.nametofont("TkDefaultFont")
        default.configure(family="Meiryo UI", size=10)
        for name in ("TkTextFont", "TkHeadingFont", "TkMenuFont"):
            tkfont.nametofont(name).configure(family="Meiryo UI")
        style = ttk.Style(self)
        style.configure("Treeview", rowheight=int(default.metrics("linespace") * 1.4))
        style.configure("Bad.TEntry", fieldbackground="#ffcdd2", foreground="#c62828")

        menu = tk.Menu(self)
        m = tk.Menu(menu, tearoff=False)
        m.add_command(label="共通設定…", command=lambda: SettingsDialog(self))
        m.add_separator()
        m.add_command(label="終了", command=self.on_close)
        menu.add_cascade(label="ファイル", menu=m)
        self.config(menu=menu)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.run_tab = RunTab(nb)
        self.editor = EditorTab(nb, self)
        nb.add(self.run_tab, text="  判定  ")
        nb.add(self.editor, text="  様式設定  ")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.report_callback_exception = lambda *exc: show_error(exc[1])

    def on_close(self) -> None:
        if self.editor.confirm_discard():
            self.destroy()


def main() -> None:
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # 高 DPI 画面で文字がぼやけないように
        except (AttributeError, OSError):
            pass
    App().mainloop()


if __name__ == "__main__":
    main()
