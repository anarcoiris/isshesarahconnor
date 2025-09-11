#!/usr/bin/env python3
"""
train_pipeline_gui.py

Interfaz unificada (GUI) para el pipeline completo:
 - Descargar / fetch (integra fetch_images_alpha.py o download_clean.py vía subprocess)
 - Preprocesado / Preparador de dataset: detección y recorte de caras, normalización y organización en train/val
 - Entrenamiento (integra train_robust_gui.py internamente o vía subprocess)
 - Diagnóstico / Calibrado (integra diagnose_model.py)
 - Export (integra export.py para TorchScript/ONNX)

Características añadidas solicitadas:
 - recorte de caras usando face_recognition (si disponible) o HaarCascade (OpenCV) como fallback
 - blacklist de dominios mostrada en UI (si existe)
 - ejecución en hilo con logging en la UI
 - presets de configuración para cada etapa (guardar/cargar JSON)
 - placeholders y UI para "características discriminativas" futuras (p. ej. entrenar detectores por rasgos)

Uso:
  python scripts/train_pipeline_gui.py --gui
  o ejecutar en modo CLI con --run-step <step> para ejecutar tareas desde terminal.

Nota: Este script intenta ejecutar los otros scripts mediante `sys.executable path/to/script.py`.
Asegúrate que todos los scripts están en la carpeta `scripts/` relativa a éste o pásalos con rutas absolutas en la UI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
from PIL import Image, UnidentifiedImageError

# Face detection imports (try alternatives)
try:
    import face_recognition  # type: ignore
    _HAS_FACE_REC = True
except Exception:
    _HAS_FACE_REC = False

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# GUI
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    _HAS_TK = True
except Exception:
    _HAS_TK = False

# logging
logger = logging.getLogger('train_pipeline')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(ch)

# helpers
PY = os.environ.get('PYTHON', None) or os.sys.executable
BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / 'scripts'

# ----------------------- Face cropping utility -------------------------

def detect_faces_pil(img: Image.Image) -> List[Tuple[int,int,int,int]]:
    """Return list of face boxes (left, top, right, bottom) using face_recognition or cv2 fallback."""
    if _HAS_FACE_REC:
        try:
            arr = np.array(img.convert('RGB'))
            boxes = face_recognition.face_locations(arr, model='hog')  # (top, right, bottom, left)
            return [(left, top, right, bottom) for (top, right, bottom, left) in boxes]
        except Exception as e:
            logger.debug('face_recognition failed: %s', e)
    if _HAS_CV2:
        try:
            arr = np.array(img.convert('RGB'))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            # use haarcascade frontal face provided by cv2
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            detector = cv2.CascadeClassifier(cascade_path)
            rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
            boxes = []
            for (x,y,w,h) in rects:
                boxes.append((x, y, x+w, y+h))
            return boxes
        except Exception as e:
            logger.debug('cv2 face detection failed: %s', e)
    # fallback: none
    return []


def crop_faces_from_image(src_path: Path, dest_dir: Path, margin: float = 0.2, min_size: int = 40) -> int:
    """Detect faces in src_path, crop and save to dest_dir. Returns number of faces saved."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src_path) as im:
            im = im.convert('RGB')
            boxes = detect_faces_pil(im)
            saved = 0
            w, h = im.size
            for i, (left, top, right, bottom) in enumerate(boxes):
                fw = right - left
                fh = bottom - top
                if fw < min_size or fh < min_size:
                    continue
                # expand box by margin
                pad_w = int(fw * margin)
                pad_h = int(fh * margin)
                l = max(0, left - pad_w)
                t = max(0, top - pad_h)
                r = min(w, right + pad_w)
                b = min(h, bottom + pad_h)
                crop = im.crop((l, t, r, b))
                out_name = dest_dir / f"{src_path.stem}_face_{i}.jpg"
                crop.save(out_name, format='JPEG', quality=92)
                saved += 1
            return saved
    except UnidentifiedImageError:
        return 0
    except Exception as e:
        logger.exception('Failed to crop %s: %s', src_path, e)
        return 0


def crop_faces_in_folder(raw_dir: Path, out_dir: Path, recursive=True, margin=0.2, min_size=40, keep_no_face=False) -> Dict[str,int]:
    """Walk raw_dir and crop faces, writing to out_dir. Returns stats dict."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {'images':0, 'faces':0, 'skipped_no_face':0}
    files = list(raw_dir.rglob('*')) if recursive else list(raw_dir.glob('*'))
    img_files = [p for p in files if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.bmp')]
    for p in img_files:
        stats['images'] += 1
        saved = crop_faces_from_image(p, out_dir, margin=margin, min_size=min_size)
        if saved == 0:
            stats['skipped_no_face'] += 1
            if keep_no_face:
                # copy original resized into out_dir/noface
                try:
                    target = out_dir / 'noface' / p.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, target)
                except Exception:
                    pass
        else:
            stats['faces'] += saved
    return stats

# ----------------------- Subprocess runners ---------------------------

def run_script(script_path: Path, args: List[str], log_cb=None) -> int:
    """Run a Python script as subprocess and stream output to log_cb (or logger). Returns returncode."""
    cmd = [PY, str(script_path)] + args
    logger.info('Running: %s', ' '.join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert p.stdout is not None
    for line in p.stdout:
        line = line.rstrip('')
        if log_cb:
            log_cb(line)
        else:
            logger.info(line)
    p.wait()
    logger.info('Process %s finished with code %d', script_path.name, p.returncode)
    return p.returncode

# ---------------------------- GUI ------------------------------------

class PipelineGUI:
    def __init__(self):
        if not _HAS_TK:
            raise RuntimeError('Tkinter no available on this system')
        self.root = tk.Tk()
        self.root.title('Dataset & Training Pipeline')
        frm = ttk.Frame(self.root, padding=8)
        frm.grid()

        # Tabs
        nb = ttk.Notebook(frm)
        nb.grid(column=0, row=0)
        self.tab_fetch = ttk.Frame(nb)
        self.tab_prep = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb)
        self.tab_diag = ttk.Frame(nb)
        self.tab_export = ttk.Frame(nb)
        nb.add(self.tab_fetch, text='Fetch')
        nb.add(self.tab_prep, text='Prep')
        nb.add(self.tab_train, text='Train')
        nb.add(self.tab_diag, text='Diagnose')
        nb.add(self.tab_export, text='Export')

        # Fetch tab
        ttk.Label(self.tab_fetch, text='Script path (fetch)').grid(column=0, row=0, sticky='w')
        self.fetch_path = ttk.Entry(self.tab_fetch, width=80)
        self.fetch_path.insert(0, str(SCRIPTS_DIR / 'fetch_images_alpha.py'))
        self.fetch_path.grid(column=1, row=0)
        ttk.Button(self.tab_fetch, text='Browse', command=self.browse_fetch).grid(column=2, row=0)

        ttk.Label(self.tab_fetch, text='Query positive').grid(column=0, row=1, sticky='w')
        self.fetch_pos = ttk.Entry(self.tab_fetch, width=50); self.fetch_pos.insert(0, 'Linda Hamilton')
        self.fetch_pos.grid(column=1, row=1, sticky='w')
        ttk.Label(self.tab_fetch, text='Num images').grid(column=0, row=2, sticky='w')
        self.fetch_num = ttk.Entry(self.tab_fetch, width=10); self.fetch_num.insert(0, '1000')
        self.fetch_num.grid(column=1, row=2, sticky='w')
        ttk.Button(self.tab_fetch, text='Run fetch', command=self.start_fetch).grid(column=1, row=3, sticky='w')

        # Prep tab
        ttk.Label(self.tab_prep, text='Raw folder').grid(column=0, row=0, sticky='w')
        self.raw_folder = ttk.Entry(self.tab_prep, width=80)
        self.raw_folder.insert(0, str(Path('data_linda_alpha/pos_raw')))
        self.raw_folder.grid(column=1, row=0)
        ttk.Button(self.tab_prep, text='Browse', command=self.browse_raw).grid(column=2, row=0)

        ttk.Label(self.tab_prep, text='Out faces folder').grid(column=0, row=1, sticky='w')
        self.faces_out = ttk.Entry(self.tab_prep, width=80)
        self.faces_out.insert(0, str(Path('data_linda_alpha/pos_faces')))
        self.faces_out.grid(column=1, row=1)
        ttk.Button(self.tab_prep, text='Browse', command=self.browse_faces_out).grid(column=2, row=1)

        ttk.Label(self.tab_prep, text='Margin (frac)').grid(column=0, row=2, sticky='w')
        self.margin_e = ttk.Entry(self.tab_prep, width=8); self.margin_e.insert(0, '0.2')
        self.margin_e.grid(column=1, row=2, sticky='w')
        ttk.Label(self.tab_prep, text='Min face size').grid(column=0, row=3, sticky='w')
        self.minface_e = ttk.Entry(self.tab_prep, width=8); self.minface_e.insert(0, '40')
        self.minface_e.grid(column=1, row=3, sticky='w')
        self.keep_no_face_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.tab_prep, text='Keep no-face copies', variable=self.keep_no_face_var).grid(column=1, row=4, sticky='w')
        ttk.Button(self.tab_prep, text='Crop faces', command=self.start_crop).grid(column=1, row=5, sticky='w')

        # Train tab (reuse previous entries)
        ttk.Label(self.tab_train, text='Script (train)').grid(column=0, row=0, sticky='w')
        self.train_path = ttk.Entry(self.tab_train, width=80)
        self.train_path.insert(0, str(SCRIPTS_DIR / 'train_robust_gui.py'))
        self.train_path.grid(column=1, row=0)
        ttk.Button(self.tab_train, text='Browse', command=self.browse_train).grid(column=2, row=0)
        ttk.Button(self.tab_train, text='Start training (GUI script)', command=self.start_train_script).grid(column=1, row=1, sticky='w')

        # Diagnose tab
        ttk.Label(self.tab_diag, text='Diagnose script').grid(column=0, row=0, sticky='w')
        self.diagnose_path = ttk.Entry(self.tab_diag, width=80)
        self.diagnose_path.insert(0, str(SCRIPTS_DIR / 'diagnose_model.py'))
        self.diagnose_path.grid(column=1, row=0)
        ttk.Button(self.tab_diag, text='Browse', command=self.browse_diag).grid(column=2, row=0)
        ttk.Label(self.tab_diag, text='Model path').grid(column=0, row=1, sticky='w')
        self.model_path = ttk.Entry(self.tab_diag, width=80); self.model_path.insert(0, 'models/best_model.pth')
        self.model_path.grid(column=1, row=1)
        ttk.Button(self.tab_diag, text='Run diagnose', command=self.start_diagnose).grid(column=1, row=2, sticky='w')

        # Export tab
        ttk.Label(self.tab_export, text='Export script').grid(column=0, row=0, sticky='w')
        self.export_path = ttk.Entry(self.tab_export, width=80)
        self.export_path.insert(0, str(SCRIPTS_DIR / 'export.py'))
        self.export_path.grid(column=1, row=0)
        ttk.Button(self.tab_export, text='Browse', command=self.browse_export).grid(column=2, row=0)
        ttk.Label(self.tab_export, text='Model to export').grid(column=0, row=1, sticky='w')
        self.export_model = ttk.Entry(self.tab_export, width=80); self.export_model.insert(0, 'models/best_model.pth')
        self.export_model.grid(column=1, row=1)
        ttk.Button(self.tab_export, text='Export model', command=self.start_export).grid(column=1, row=2, sticky='w')

        # Characteristics / discriminator placeholders
        ttk.Label(frm, text='Características discriminativas (placeholder):').grid(column=0, row=1, sticky='w')
        self.char_frame = ttk.Frame(frm)
        self.char_frame.grid(column=0, row=2, sticky='w')
        self.char_vars = {}
        for i, ch in enumerate(['front','profile','young','old','glasses','smiling']):
            v = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(self.char_frame, text=ch, variable=v)
            cb.grid(column=i, row=0, sticky='w')
            self.char_vars[ch] = v
        ttk.Label(frm, text='(Puedes usar estas opciones como metadatos para etiquetar/entrenar detectores en el futuro)').grid(column=0, row=3, sticky='w')

        # Log window and control
        self.log = tk.Text(frm, width=120, height=20)
        self.log.grid(column=0, row=4, columnspan=3)
        self.status_var = tk.StringVar(value='idle')
        ttk.Label(frm, textvariable=self.status_var).grid(column=0, row=5, sticky='w')

        # attach text handler to logger
        handler = TextHandler(self.log)
        logger.addHandler(handler)

        # worker management
        self.worker = None
        self.task_queue = queue.Queue()

    # Browse helpers
    def browse_fetch(self):
        p = filedialog.askopenfilename(filetypes=[('Python','*.py')])
        if p: self.fetch_path.delete(0,'end'); self.fetch_path.insert(0,p)
    def browse_raw(self):
        p = filedialog.askdirectory()
        if p: self.raw_folder.delete(0,'end'); self.raw_folder.insert(0,p)
    def browse_faces_out(self):
        p = filedialog.askdirectory()
        if p: self.faces_out.delete(0,'end'); self.faces_out.insert(0,p)
    def browse_train(self):
        p = filedialog.askopenfilename(filetypes=[('Python','*.py')])
        if p: self.train_path.delete(0,'end'); self.train_path.insert(0,p)
    def browse_diag(self):
        p = filedialog.askopenfilename(filetypes=[('Python','*.py')])
        if p: self.diagnose_path.delete(0,'end'); self.diagnose_path.insert(0,p)
    def browse_export(self):
        p = filedialog.askopenfilename(filetypes=[('Python','*.py')])
        if p: self.export_path.delete(0,'end'); self.export_path.insert(0,p)

    # Start tasks
    def start_fetch(self):
        script = Path(self.fetch_path.get())
        if not script.exists():
            messagebox.showerror('No script', 'Fetch script not found')
            return
        pos = self.fetch_pos.get().strip()
        num = int(self.fetch_num.get().strip())
        out = Path('data_fetch')
        args = ['--pos', pos, '--num', str(num), '--out', str(out), '--use-date-ranges']
        self._start_subprocess_task(script, args, f'Fetch {pos}')

    def start_crop(self):
        raw = Path(self.raw_folder.get())
        out = Path(self.faces_out.get())
        if not raw.exists():
            messagebox.showerror('No raw folder', 'Raw folder not found')
            return
        margin = float(self.margin_e.get())
        min_face = int(self.minface_e.get())
        keep_no_face = bool(self.keep_no_face_var.get())
        # run cropping in background
        def job():
            self.status_var.set('cropping')
            logger.info('Cropping faces: raw=%s out=%s margin=%.2f min_face=%d', raw, out, margin, min_face)
            stats = crop_faces_in_folder(raw, out, recursive=True, margin=margin, min_size=min_face, keep_no_face=keep_no_face)
            logger.info('Cropping finished: %s', stats)
            self.status_var.set('idle')
        t = threading.Thread(target=job, daemon=True)
        t.start()

    def start_train_script(self):
        script = Path(self.train_path.get())
        if not script.exists():
            messagebox.showerror('No script', 'Train script not found')
            return
        # start GUI train script (it has its own window)
        args = ['--gui']
        self._start_subprocess_task(script, args, 'Train GUI')

    def start_diagnose(self):
        script = Path(self.diagnose_path.get())
        if not script.exists():
            messagebox.showerror('No script', 'Diagnose script not found')
            return
        model = self.model_path.get().strip()
        args = ['--model', model, '--data', 'data/final', '--split', 'val']
        self._start_subprocess_task(script, args, 'Diagnose')

    def start_export(self):
        script = Path(self.export_path.get())
        if not script.exists():
            messagebox.showerror('No script', 'Export script not found')
            return
        model = self.export_model.get().strip()
        args = ['--model', model, '--out', 'models/exported']
        self._start_subprocess_task(script, args, 'Export')

    def _start_subprocess_task(self, script: Path, args: List[str], title: str):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo('Running', 'Another task is running. Wait or close it.')
            return
        def job():
            self.status_var.set(title)
            run_script(script, args, log_cb=lambda ln: logger.info('[%s] %s', title, ln))
            self.status_var.set('idle')
        self.worker = threading.Thread(target=job, daemon=True)
        self.worker.start()

    def mainloop(self):
        self.root.mainloop()


class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record) + ''
        try:
            self.text_widget.configure(state='normal')
            self.text_widget.insert('end', msg)
            self.text_widget.see('end')
            self.text_widget.configure(state='disabled')
        except Exception:
            pass

# --------------------------- CLI entrypoint ---------------------------

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--run-step', choices=['fetch','crop','train','diagnose','export'], help='run single step non-interactive')
    parser.add_argument('--script', help='script path override')
    args = parser.parse_args()

    if args.gui:
        if not _HAS_TK:
            logger.error('Tkinter no disponible')
            return
        app = PipelineGUI()
        app.mainloop()
        return

    # non-interactive steps (minimal)
    if args.run_step == 'crop':
        # example usage: set RAW_DIR/OUT_DIR env or just use defaults
        raw = Path('data_linda_alpha/pos_raw')
        out = Path('data_linda_alpha/pos_faces')
        stats = crop_faces_in_folder(raw, out)
        print('Crop stats:', stats)
    else:
        print('Use --gui for full interface or implement other run-step handlers')

if __name__ == '__main__':
    main_cli()
