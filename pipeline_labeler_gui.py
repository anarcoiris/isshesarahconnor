#!/usr/bin/env python3
# pipeline_labeler_gui.py
"""
Interfaz de etiquetado para recortes de caras y manejo de atributos discriminativos.

Funcionalidades:
- Cargar carpeta con imágenes (recortes de cara).
- Definir esquema de atributos dinámicamente (single-choice o multi-label).
- Revisar imágenes una a una, ver miniatura y metadatos.
- Guardar anotaciones por imagen en annotations.json y annotations.csv.
- Filtrar imágenes por atributos/valores.
- Aplicar etiquetas en batch a selección.
- Exportar subconjunto (copiar archivos) en estructura organizada por etiquetas.
- Importar/exportar esquema de atributos.

Uso:
    python pipeline_labeler_gui.py --gui
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from PIL import Image, ImageTk, UnidentifiedImageError
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import pandas as pd

# ---------- Utils & Data stores ----------

DEFAULT_SCHEMA = {
    # attribute_name: {"type": "single" or "multi", "labels": [list of labels], "default": None or []}
    "pose": {"type": "single", "labels": ["front", "profile", "three_quarter"], "default": "front"},
    "age": {"type": "single", "labels": ["young", "adult", "old"], "default": "adult"},
    "hair_color": {"type": "single", "labels": ["black", "brown", "blonde", "red", "gray", "other"], "default": "brown"},
    "eye_color": {"type": "single", "labels": ["brown", "blue", "green", "other"], "default": "brown"},
    "accessories": {"type": "multi", "labels": ["glasses", "hat", "earrings", "mask"], "default": []},
    "smiling": {"type": "single", "labels": ["no", "yes"], "default": "no"}
}


class SchemaManager:
    """Gestiona el esquema de atributos (persistencia y edición)."""
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.schema: Dict[str, Dict] = DEFAULT_SCHEMA.copy()

    def load(self, path: Optional[Path] = None):
        p = Path(path) if path else self.path
        if p and p.exists():
            with open(p, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
        return self.schema

    def save(self, path: Optional[Path] = None):
        p = Path(path) if path else self.path
        if not p:
            raise ValueError("No path provided to save schema")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.schema, f, indent=2, ensure_ascii=False)

    def add_attribute(self, name: str, typ: str = "single", labels: Optional[List[str]] = None, default: Any = None):
        if labels is None:
            labels = []
        self.schema[name] = {"type": typ, "labels": labels, "default": default if default is not None else ([] if typ == "multi" else (labels[0] if labels else None))}

    def add_label(self, attr: str, label: str):
        if attr not in self.schema:
            raise KeyError(attr)
        if label not in self.schema[attr]["labels"]:
            self.schema[attr]["labels"].append(label)


class AnnotationStore:
    """Guarda y carga anotaciones para cada imagen y exporta CSV."""
    def __init__(self, annotations_path: Optional[Path] = None):
        self.path = Path(annotations_path) if annotations_path else None
        # mapping: rel_path_or_abs -> {attr: value or [values], ...}
        self.data: Dict[str, Dict] = {}

    def load(self, path: Optional[Path] = None):
        p = Path(path) if path else self.path
        if p and p.exists():
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
                if isinstance(raw, dict):
                    self.data = raw
                else:
                    # handle list of records
                    self.data = {r["image"]: r for r in raw}
        return self.data

    def save(self, path: Optional[Path] = None):
        p = Path(path) if path else self.path
        if not p:
            raise ValueError("No path to save annotations")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        # Also export CSV for convenience
        try:
            rows = []
            for img, ann in self.data.items():
                row = {"image": img}
                for k, v in ann.items():
                    if isinstance(v, list):
                        row[k] = ";".join([str(x) for x in v])
                    else:
                        row[k] = v
                rows.append(row)
            df = pd.DataFrame(rows)
            df.to_csv(p.with_suffix(".csv"), index=False)
        except Exception:
            pass

    def set_annotation(self, image_path: str, ann: Dict):
        self.data[image_path] = ann

    def get_annotation(self, image_path: str) -> Dict:
        return self.data.get(image_path, {})

    def bulk_apply(self, images: List[str], attr: str, value: Any, schema: SchemaManager):
        """Aplica la etiqueta value al atributo attr para las imágenes listadas.
        Si attribute es multi, 'value' puede ser list or single (we toggle / add)."""
        typ = schema.schema[attr]["type"]
        for img in images:
            ann = self.get_annotation(img).copy()
            if typ == "single":
                ann[attr] = value
            else:
                cur = set(ann.get(attr, []))
                # if value is list add all; if str toggle
                if isinstance(value, list):
                    cur.update(value)
                else:
                    if value in cur:
                        cur.remove(value)
                    else:
                        cur.add(value)
                ann[attr] = sorted(list(cur))
            self.set_annotation(img, ann)


# ---------- Thumbnail cache & image list ----------

def make_thumbnail(path: Path, size=(320, 320)):
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail(size, Image.LANCZOS)
            return im.copy()
    except UnidentifiedImageError:
        return None
    except Exception:
        return None


# ---------- GUI ----------

class LabelerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pipeline Labeler - Discriminative Attributes")
        self.schema_mgr = SchemaManager()
        self.ann_store = AnnotationStore()
        self.image_folder: Optional[Path] = None
        self.image_paths: List[Path] = []
        self.filtered_indices: List[int] = []
        self.current_index: int = 0
        self.thumbnail_cache: Dict[str, ImageTk.PhotoImage] = {}
        self.tk_image_ref = None  # keep reference for Tkinter

        self.setup_ui()

    def setup_ui(self):
        frm = ttk.Frame(self.root, padding=6)
        frm.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Top controls: folder & load
        top = ttk.Frame(frm)
        top.grid(column=0, row=0, sticky="ew")
        ttk.Label(top, text="Folder:").grid(column=0, row=0, sticky="w")
        self.folder_entry = ttk.Entry(top, width=60)
        self.folder_entry.grid(column=1, row=0, sticky="w")
        ttk.Button(top, text="Browse", command=self.browse_folder).grid(column=2, row=0, sticky="w")
        ttk.Button(top, text="Load images", command=self.load_images).grid(column=3, row=0, sticky="w")
        ttk.Button(top, text="Reload annotations", command=self.reload_annotations).grid(column=4, row=0, sticky="w")

        # Left: listbox of filenames + filter builder
        left = ttk.Frame(frm)
        left.grid(column=0, row=1, sticky="nsw", padx=(0,6))
        left.columnconfigure(0, weight=1)
        ttk.Label(left, text="Images").grid(column=0, row=0, sticky="w")
        self.listbox = tk.Listbox(left, width=40, height=25, selectmode=tk.EXTENDED)
        self.listbox.grid(column=0, row=1, sticky="ns")
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)
        self.listbox_scroll = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=self.listbox_scroll.set)
        self.listbox_scroll.grid(column=1, row=1, sticky="ns")
        ttk.Button(left, text="Apply Bulk", command=self.bulk_apply_dialog).grid(column=0, row=2, sticky="w", pady=(6,0))

        # Filter builder
        filt_frame = ttk.LabelFrame(left, text="Filter")
        filt_frame.grid(column=0, row=3, sticky="ew", pady=(8,0))
        self.filter_attr_cb = ttk.Combobox(filt_frame, values=list(self.schema_mgr.schema.keys()), state="readonly")
        self.filter_attr_cb.grid(column=0, row=0, sticky="w")
        self.filter_val_cb = ttk.Combobox(filt_frame, values=[], state="readonly")
        self.filter_val_cb.grid(column=1, row=0, sticky="w")
        ttk.Button(filt_frame, text="Apply", command=self.apply_filter).grid(column=0, row=1, sticky="w")
        ttk.Button(filt_frame, text="Clear", command=self.clear_filter).grid(column=1, row=1, sticky="w")

        # Center: image preview
        center = ttk.Frame(frm)
        center.grid(column=1, row=1, sticky="nsew")
        center.columnconfigure(0, weight=1)
        ttk.Label(center, text="Preview").grid(column=0, row=0, sticky="w")
        self.canvas = tk.Canvas(center, width=360, height=360, bg="#222")
        self.canvas.grid(column=0, row=1, sticky="n")
        nav = ttk.Frame(center)
        nav.grid(column=0, row=2, pady=(6,0))
        ttk.Button(nav, text="<< Prev", command=self.prev_image).grid(column=0, row=0)
        ttk.Button(nav, text="Next >>", command=self.next_image).grid(column=1, row=0)
        ttk.Button(nav, text="Save annotations", command=self.save_annotations).grid(column=2, row=0)
        ttk.Button(nav, text="Export subset", command=self.export_subset_dialog).grid(column=3, row=0)

        # Right: attribute controls and schema manager
        right = ttk.Frame(frm)
        right.grid(column=2, row=1, sticky="ns", padx=(6,0))
        ttk.Label(right, text="Attributes").grid(column=0, row=0, sticky="w")
        self.attr_frame = ttk.Frame(right)
        self.attr_frame.grid(column=0, row=1, sticky="n")
        # schema manager buttons
        sm = ttk.Frame(right)
        sm.grid(column=0, row=2, pady=(8,0), sticky="w")
        ttk.Button(sm, text="Attribute manager", command=self.open_schema_manager).grid(column=0, row=0, sticky="w")
        ttk.Button(sm, text="Create subset (by filter)", command=self.export_subset_dialog).grid(column=1, row=0, sticky="w")

        # Bottom: log
        log_frame = ttk.Frame(frm)
        log_frame.grid(column=0, row=4, columnspan=3, sticky="ew", pady=(8,0))
        ttk.Label(log_frame, text="Log").grid(column=0, row=0, sticky="w")
        self.log_text = tk.Text(log_frame, height=8)
        self.log_text.grid(column=0, row=1, columnspan=3, sticky="ew")
        self.log("Ready.")

        # set resizing weights
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(1, weight=1)

        # initial UI population
        self.rebuild_attribute_controls()

    # ---------- Logging ----------
    def log(self, *args):
        line = " ".join(str(a) for a in args) + "\n"
        try:
            self.log_text.insert("end", line)
            self.log_text.see("end")
        except Exception:
            print(line.strip())

    # ---------- Folder & images ----------
    def browse_folder(self):
        p = filedialog.askdirectory()
        if p:
            self.folder_entry.delete(0, "end")
            self.folder_entry.insert(0, p)

    def load_images(self):
        folder = Path(self.folder_entry.get().strip())
        if not folder.exists() or not folder.is_dir():
            messagebox.showerror("Error", "Folder inválido")
            return
        self.image_folder = folder
        # collect images
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in exts and p.is_file()]
        self.image_paths = paths
        self.filtered_indices = list(range(len(self.image_paths)))
        self.current_index = 0
        self.thumbnail_cache.clear()
        self.populate_listbox()
        # load annotations if exist
        ann_path = folder / "annotations.json"
        if ann_path.exists():
            try:
                self.ann_store.path = ann_path
                self.ann_store.load()
                self.log(f"Loaded annotations from {ann_path}")
            except Exception as e:
                self.log("Failed to load annotations:", e)
        else:
            # set default path
            self.ann_store.path = ann_path
        self.refresh_preview()

    def populate_listbox(self):
        self.listbox.delete(0, "end")
        for i, p in enumerate(self.image_paths):
            self.listbox.insert("end", p.name)

    def on_list_select(self, evt=None):
        sel = self.listbox.curselection()
        if sel:
            # if multiple selected, focus first
            idx = sel[0]
            # map shown index to absolute index
            self.current_index = idx
            self.refresh_preview()

    def index_to_path(self, idx: int) -> Optional[Path]:
        if 0 <= idx < len(self.image_paths):
            return self.image_paths[idx]
        return None

    # ---------- Preview & annotation UI ----------
    def refresh_preview(self):
        if not self.image_paths:
            self.canvas.delete("all")
            return
        p = self.index_to_path(self.current_index)
        if not p:
            return
        # load thumbnail (cache the PhotoImage)
        key = str(p)
        if key not in self.thumbnail_cache:
            pil = make_thumbnail(p, size=(340, 340))
            if pil is None:
                self.log("Cannot load", p)
                blank = Image.new("RGB", (340, 340), (50, 50, 50))
                pil = blank
            tkimg = ImageTk.PhotoImage(pil)
            self.thumbnail_cache[key] = tkimg
        else:
            tkimg = self.thumbnail_cache[key]
        self.canvas.delete("all")
        self.canvas.create_image(180, 180, image=tkimg)
        self.tk_image_ref = tkimg  # keep reference
        # load annotation values
        ann = self.ann_store.get_annotation(str(p))
        # update controls to reflect annotation
        self.update_attribute_controls(ann)

        # select in listbox
        try:
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(self.current_index)
            self.listbox.see(self.current_index)
        except Exception:
            pass

    def next_image(self):
        if not self.image_paths:
            return
        self.save_current_annotation()
        self.current_index = min(len(self.image_paths)-1, self.current_index+1)
        self.refresh_preview()

    def prev_image(self):
        if not self.image_paths:
            return
        self.save_current_annotation()
        self.current_index = max(0, self.current_index-1)
        self.refresh_preview()

    # ---------- Attribute controls dynamic ----------
    def rebuild_attribute_controls(self):
        # clear
        for w in self.attr_frame.winfo_children():
            w.destroy()
        # update combo in filter
        self.filter_attr_cb['values'] = list(self.schema_mgr.schema.keys())
        # create controls for each attribute
        self.attr_widgets = {}  # attr -> widget refs
        r = 0
        for attr, meta in self.schema_mgr.schema.items():
            ttk.Label(self.attr_frame, text=attr).grid(column=0, row=r, sticky="w")
            if meta["type"] == "single":
                var = tk.StringVar(value=meta.get("default", "") if meta.get("default") is not None else "")
                cb = ttk.Combobox(self.attr_frame, values=meta["labels"], textvariable=var, state="readonly", width=20)
                cb.grid(column=1, row=r, sticky="w")
                self.attr_widgets[attr] = {"type":"single","var":var,"widget":cb}
            else:
                # multi: list of checkbuttons
                box = ttk.Frame(self.attr_frame)
                box.grid(column=1, row=r, sticky="w")
                vars = {}
                col = 0
                for lab in meta["labels"]:
                    v = tk.BooleanVar(value=False)
                    cb = ttk.Checkbutton(box, text=lab, variable=v)
                    cb.grid(column=col, row=0, sticky="w")
                    vars[lab] = v
                    col += 1
                self.attr_widgets[attr] = {"type":"multi","vars":vars,"widget":box}
            r += 1
        # Add "Save current" button
        ttk.Button(self.attr_frame, text="Save current annotation", command=self.save_current_annotation).grid(column=0, row=r, columnspan=2, pady=(8,0))

    def update_attribute_controls(self, ann: Dict):
        # set widget values according to ann
        for attr, meta in self.schema_mgr.schema.items():
            w = self.attr_widgets.get(attr)
            if not w:
                continue
            if meta["type"] == "single":
                val = ann.get(attr, meta.get("default"))
                try:
                    w["var"].set(val if val is not None else "")
                except Exception:
                    pass
            else:
                cur = set(ann.get(attr, []))
                for lab, var in w["vars"].items():
                    var.set(lab in cur)

    def gather_current_annotation(self) -> Dict:
        ann = {}
        for attr, meta in self.schema_mgr.schema.items():
            w = self.attr_widgets.get(attr)
            if not w:
                continue
            if meta["type"] == "single":
                val = w["var"].get()
                ann[attr] = val
            else:
                vals = [lab for lab, var in w["vars"].items() if var.get()]
                ann[attr] = vals
        return ann

    def save_current_annotation(self):
        if not self.image_paths:
            return
        p = self.index_to_path(self.current_index)
        if not p:
            return
        ann = self.gather_current_annotation()
        self.ann_store.set_annotation(str(p), ann)
        try:
            self.ann_store.save(self.ann_store.path)
            self.log(f"Saved annotation for {p.name}")
        except Exception as e:
            self.log("Failed saving annotation:", e)

    # ---------- Schema manager dialog ----------
    def open_schema_manager(self):
        dlg = SchemaDialog(self.root, self.schema_mgr)
        self.root.wait_window(dlg.top)
        if dlg.modified:
            self.rebuild_attribute_controls()
            self.log("Schema updated")

    # ---------- Filter ----------
    def apply_filter(self):
        attr = self.filter_attr_cb.get()
        val = self.filter_val_cb.get()
        if not attr or not val:
            messagebox.showinfo("Filter", "Selecciona atributo y valor")
            return
        self.filtered_indices = []
        for i, p in enumerate(self.image_paths):
            ann = self.ann_store.get_annotation(str(p))
            meta = self.schema_mgr.schema.get(attr)
            if not meta:
                continue
            if meta["type"] == "single":
                if ann.get(attr) == val:
                    self.filtered_indices.append(i)
            else:
                if val in ann.get(attr, []):
                    self.filtered_indices.append(i)
        # update listbox to display only filtered names
        self.listbox.delete(0, "end")
        for i in self.filtered_indices:
            self.listbox.insert("end", self.image_paths[i].name)
        self.current_index = self.filtered_indices[0] if self.filtered_indices else 0
        self.log(f"Filter applied: {len(self.filtered_indices)} images")
        if self.filtered_indices:
            # show first filtered
            self.refresh_preview()

    def clear_filter(self):
        # restore full list
        self.filtered_indices = list(range(len(self.image_paths)))
        self.populate_listbox()
        self.log("Filter cleared")
        self.refresh_preview()

    # ---------- Bulk apply ----------
    def bulk_apply_dialog(self):
        if not self.image_paths:
            return
        # choose attr
        attrs = list(self.schema_mgr.schema.keys())
        attr = simpledialog.askstring("Bulk apply", f"Attr a modificar (available: {', '.join(attrs)})")
        if not attr:
            return
        if attr not in self.schema_mgr.schema:
            messagebox.showerror("Error", "Atributo no encontrado")
            return
        meta = self.schema_mgr.schema[attr]
        if meta["type"] == "single":
            # choose value
            val = simpledialog.askstring("Value", f"Valor (choices: {', '.join(meta['labels'])})")
            if val is None:
                return
            images = [str(self.image_paths[i]) for i in self.listbox.curselection()] or [str(self.image_paths[self.current_index])]
            self.ann_store.bulk_apply(images, attr, val, self.schema_mgr)
            self.log(f"Applied {attr}={val} to {len(images)} images")
        else:
            # multi -> toggle label
            val = simpledialog.askstring("Value", f"Label to toggle/add (choices: {', '.join(meta['labels'])})")
            if val is None:
                return
            images = [str(self.image_paths[i]) for i in self.listbox.curselection()] or [str(self.image_paths[self.current_index])]
            self.ann_store.bulk_apply(images, attr, val, self.schema_mgr)
            self.log(f"Toggled {val} in {attr} for {len(images)} images")
        # save
        self.ann_store.save(self.ann_store.path)

    # ---------- Export subset ----------
    def export_subset_dialog(self):
        # ask mode: by selection or by filter
        mode = simpledialog.askstring("Export subset", "Export mode: 'selection' or 'filter' (or 'all')", initialvalue="selection")
        if not mode:
            return
        if mode not in ("selection", "filter", "all"):
            messagebox.showerror("Invalid", "Choose selection/filter/all")
            return
        if mode == "selection":
            sel = self.listbox.curselection()
            if not sel:
                messagebox.showinfo("No selection", "Selecciona imágenes en la lista")
                return
            image_indices = [i for i in sel]
        elif mode == "filter":
            if not self.filtered_indices:
                messagebox.showinfo("No filter", "Aplica un filtro primero")
                return
            image_indices = self.filtered_indices
        else:
            image_indices = list(range(len(self.image_paths)))
        # choose target dir
        target = filedialog.askdirectory(title="Select export target folder")
        if not target:
            return
        target = Path(target)
        # ask organization: by attribute?
        org_attr = simpledialog.askstring("Organize by", "Organize by attribute (or leave blank for flat copy)", initialvalue="")
        # copy files
        copied = 0
        for idx in image_indices:
            p = self.image_paths[idx]
            ann = self.ann_store.get_annotation(str(p))
            if org_attr and org_attr in ann:
                val = ann[org_attr]
                # val could be list or single
                if isinstance(val, list):
                    # copy to each label
                    for v in val:
                        dest = target / org_attr / str(v)
                        dest.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, dest / p.name)
                        copied += 1
                else:
                    dest = target / org_attr / str(val)
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dest / p.name)
                    copied += 1
            else:
                # flat
                dest = target / "images"
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest / p.name)
                copied += 1
        messagebox.showinfo("Export done", f"Copied {copied} files to {target}")
        self.log(f"Exported {copied} files to {target}")

    # ---------- Annotations ----------
    def save_annotations(self):
        if not self.ann_store.path:
            # ask for path
            p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
            if not p:
                return
            self.ann_store.path = Path(p)
        self.ann_store.save(self.ann_store.path)
        self.log(f"Annotations saved to {self.ann_store.path}")

    def reload_annotations(self):
        if not self.ann_store.path:
            messagebox.showinfo("No annotations", "No annotations path set yet")
            return
        try:
            self.ann_store.load(self.ann_store.path)
            self.log("Annotations reloaded")
            self.refresh_preview()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load annotations: {e}")

# ---------- Schema Dialog ----------

class SchemaDialog:
    def __init__(self, parent, schema_mgr: SchemaManager):
        self.parent = parent
        self.schema_mgr = schema_mgr
        self.modified = False
        self.top = tk.Toplevel(parent)
        self.top.title("Attribute manager")
        self.build_ui()

    def build_ui(self):
        frm = ttk.Frame(self.top, padding=8)
        frm.grid()
        self.tree = ttk.Treeview(frm, columns=("type","labels"), show="headings", height=10)
        self.tree.heading("type", text="Type")
        self.tree.heading("labels", text="Labels")
        self.tree.grid(column=0, row=0, columnspan=3)
        self.refresh_tree()
        ttk.Button(frm, text="Add attribute", command=self.add_attribute).grid(column=0, row=1)
        ttk.Button(frm, text="Add label", command=self.add_label).grid(column=1, row=1)
        ttk.Button(frm, text="Save schema", command=self.save_schema).grid(column=2, row=1)
        ttk.Button(frm, text="Close", command=self.close).grid(column=1, row=2)

    def refresh_tree(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for k, v in self.schema_mgr.schema.items():
            typ = v.get("type","single")
            labs = ",".join(v.get("labels",[]))
            self.tree.insert("", "end", iid=k, values=(typ, labs))

    def add_attribute(self):
        name = simpledialog.askstring("Attribute name", "New attribute name", parent=self.top)
        if not name:
            return
        typ = simpledialog.askstring("Type", "Type: 'single' or 'multi'", parent=self.top, initialvalue="single")
        if typ not in ("single", "multi"):
            messagebox.showerror("Invalid", "Type must be single or multi")
            return
        labels = simpledialog.askstring("Labels", "Comma-separated labels (example: a,b,c)", parent=self.top)
        labs = [s.strip() for s in labels.split(",")] if labels else []
        default = labs[0] if typ=="single" and labs else ([] if typ=="multi" else None)
        self.schema_mgr.add_attribute(name, typ, labs, default)
        self.modified = True
        self.refresh_tree()

    def add_label(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Select", "Select attribute row first")
            return
        attr = sel[0]
        lab = simpledialog.askstring("New label", f"New label to add to {attr}", parent=self.top)
        if not lab:
            return
        self.schema_mgr.add_label(attr, lab)
        self.modified = True
        self.refresh_tree()

    def save_schema(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if not p:
            return
        self.schema_mgr.save(Path(p))
        messagebox.showinfo("Saved", f"Schema saved to {p}")

    def close(self):
        self.top.destroy()


# ---------- CLI / Entrypoint ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Start GUI")
    parser.add_argument("--folder", default=None, help="Optionally pre-load a folder of images")
    args = parser.parse_args()

    if args.gui:
        root = tk.Tk()
        app = LabelerGUI(root)
        if args.folder:
            root.after(100, lambda: (app.folder_entry.delete(0,"end"), app.folder_entry.insert(0,args.folder), app.load_images()))
        root.mainloop()
    else:
        print("Run with --gui to open the labeling interface.")

if __name__ == "__main__":
    main()
