#!/usr/bin/env python3
"""
fetch_images_alpha.py (alpha -> beta)

Mejoras añadidas a la versión alpha solicitadas por el usuario:
 - Resume robusto: mantiene índice persistente de phash -> dest_filename y URLs procesadas para no re-descargar ni re-procesar.
 - Blacklist de dominios: detecta código HTTP 403 y guarda dominios a `blacklist_domains.txt`; evita futuras descargas desde esos dominios.
 - Excluir dominios problemáticos automáticamente (se consulta la blacklist en cada descarga programada).
 - Interfaz gráfica (Tkinter) ligera para lanzar la ejecución y visualizar logs/progreso. También puede ejecutarse en modo CLI.
 - Guardado periódico del estado (cada N eventos) para tolerancia a fallos.

Uso (CLI):
  python scripts/fetch_images_alpha.py --pos "Linda Hamilton" --num 2000 --engine google --out data_linda_alpha

Uso (GUI):
  python scripts/fetch_images_alpha.py --gui

Notas:
 - Para SerpAPI proporciona --serpapi_key
 - La implementación intenta capturar 403s/errores al descargar (serpapi y descargas internas). Icrawler interna puede devolver errores fuera de nuestro control — la blacklist se llenará principalmente cuando se use SerpAPI o cuando el propio downloader haga requests.
 - Revisa y adapta parámetros de `modifiers` por defecto según tu caso de uso para mejorar diversidad de resultados.

"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlsplit

import requests
from PIL import Image, UnidentifiedImageError
import imagehash
import pandas as pd
from tqdm import tqdm

# GUI
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    _HAS_TK = True
except Exception:
    _HAS_TK = False

# Optional face detection
try:
    import face_recognition  # type: ignore
    _HAS_FACE = True
except Exception:
    _HAS_FACE = False

# icrawler optional
try:
    from icrawler.builtin import GoogleImageCrawler, BingImageCrawler  # type: ignore
    _HAS_ICRAWLER = True
except Exception:
    _HAS_ICRAWLER = False

# serpapi optional
try:
    from serpapi import GoogleSearch  # type: ignore
    _HAS_SERPAPI = True
except Exception:
    _HAS_SERPAPI = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fetch_images")

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36"
}

STATE_FILENAME = ".fetch_state.json"
INDEX_FILENAME = ".fetch_index.json"
BLACKLIST_FILENAME = "blacklist_domains.txt"

SAVE_STATE_EVERY = 30  # save state after N downloads/processed events

# ----------------------------- Helpers ---------------------------------

def safe_ext_from_url(url: str) -> str:
    path = urlsplit(url).path
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    if ext in ("jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff", "tif"):
        return ext if ext != "jpeg" else "jpg"
    return "jpg"


def domain_of_url(url: str) -> Optional[str]:
    try:
        return urlsplit(url).hostname
    except Exception:
        return None


# ---------------------------- State management -------------------------

def load_state(out_root: Path) -> Tuple[Dict[str, str], Set[str], Set[str]]:
    """Load index (phash->dest) and set of processed_urls and blacklist.
    Returns (phash_index, processed_urls, blacklist_domains)
    """
    out_root = Path(out_root)
    phash_index: Dict[str, str] = {}
    processed_urls: Set[str] = set()
    blacklist: Set[str] = set()

    idxp = out_root / INDEX_FILENAME
    if idxp.exists():
        try:
            with open(idxp, 'r', encoding='utf-8') as f:
                phash_index = json.load(f)
        except Exception:
            logger.exception("Failed to load index file %s", idxp)

    statep = out_root / STATE_FILENAME
    if statep.exists():
        try:
            with open(statep, 'r', encoding='utf-8') as f:
                st = json.load(f)
                processed_urls = set(st.get('processed_urls', []))
        except Exception:
            logger.exception("Failed to load state file %s", statep)

    blp = out_root / BLACKLIST_FILENAME
    if blp.exists():
        try:
            with open(blp, 'r', encoding='utf-8') as f:
                for line in f:
                    dom = line.strip()
                    if dom:
                        blacklist.add(dom)
        except Exception:
            logger.exception("Failed to load blacklist file %s", blp)

    return phash_index, processed_urls, blacklist


def save_state(out_root: Path, phash_index: Dict[str, str], processed_urls: Set[str], blacklist: Set[str]):
    out_root = Path(out_root)
    idxp = out_root / INDEX_FILENAME
    statep = out_root / STATE_FILENAME
    blp = out_root / BLACKLIST_FILENAME
    try:
        with open(idxp, 'w', encoding='utf-8') as f:
            json.dump(phash_index, f, indent=2, ensure_ascii=False)
    except Exception:
        logger.exception("Failed to save index file %s", idxp)
    try:
        with open(statep, 'w', encoding='utf-8') as f:
            json.dump({'processed_urls': list(processed_urls)}, f, indent=2, ensure_ascii=False)
    except Exception:
        logger.exception("Failed to save state file %s", statep)
    try:
        with open(blp, 'w', encoding='utf-8') as f:
            for d in sorted(blacklist):
                f.write(d + '')
    except Exception:
        logger.exception("Failed to save blacklist file %s", blp)


# ---------------------------- Downloaders ------------------------------

def download_url_to(path: Path, url: str, headers: dict, timeout=10, max_bytes=None, verify=True, retries=3,
                    blacklist: Optional[Set[str]] = None, processed_urls: Optional[Set[str]] = None):
    """
    Descarga con requests y gestiona reintentos.
    - Si recibe status 403 añade el dominio a blacklist.
    - Si processed_urls contiene la url, no descarga.
    Devuelve (ok: bool, reason: Optional[str], domain_added: Optional[str])
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    url = url.strip()
    dom = domain_of_url(url)
    domain_added = None
    if blacklist and dom and dom in blacklist:
        return False, 'blacklisted_domain', None
    if processed_urls and url in processed_urls:
        return False, 'already_processed', None

    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, stream=True, verify=verify)
            if r.status_code == 403:
                # blacklisting domain
                if dom:
                    domain_added = dom
                return False, f'status_403', domain_added
            if r.status_code != 200:
                return False, f'status_{r.status_code}', None
            with open(path, 'wb') as f:
                total = 0
                for chunk in r.iter_content(1024 * 8):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
                        if max_bytes and total > max_bytes:
                            return False, 'too_big', None
            return True, None, None
        except requests.exceptions.SSLError as e:
            logger.debug('SSL error %s', e)
            if not verify:
                # if verify False didn't help, continue retry
                pass
        except Exception as e:
            logger.debug('download attempt %d failed for %s: %s', attempt + 1, url, e)
            time.sleep(0.3 + attempt * 0.6 + random.random() * 0.2)
            continue
    return False, 'download_failed', None


# SerpAPI download (parallel) with blacklist checks and state updates
def serpapi_images_for_query(query: str, out_dir: Path, max_num=300, serpapi_key: Optional[str] = None, headers=None, verify=True,
                             phash_index: Dict[str, str] = None, processed_urls: Set[str] = None, blacklist: Set[str] = None,
                             state_save_callback=None):
    if not _HAS_SERPAPI:
        raise RuntimeError('serpapi no instalado')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = {'engine': 'google_images', 'q': query, 'api_key': serpapi_key, 'ijn': '0'}
    count = 0
    page = 0
    max_pages = 100
    headers = headers or DEFAULT_HEADERS
    workers = min(12, max(4, max_num // 20))

    while count < max_num and page < max_pages:
        params['ijn'] = str(page)
        gs = GoogleSearch(params)
        res = gs.get_dict()
        imgs = res.get('images_results', [])
        if not imgs:
            break

        # parallel downloads
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {}
            for i, item in enumerate(imgs):
                if count + len(futures) >= max_num:
                    break
                url = item.get('original') or item.get('thumbnail') or item.get('src')
                if not url:
                    continue
                if processed_urls and url in processed_urls:
                    continue
                dom = domain_of_url(url)
                if blacklist and dom and dom in blacklist:
                    continue
                ext = safe_ext_from_url(url)
                fname = out_dir / f'serp_{page}_{i}.{ext}'
                fut = ex.submit(download_url_to, fname, url, headers, 10, 10_000_000, verify, 3, blacklist, processed_urls)
                futures[fut] = (url, fname)

            for fut in as_completed(futures):
                url, fname = futures[fut]
                ok, reason, domain_added = fut.result()
                if domain_added:
                    blacklist.add(domain_added)
                    logger.info('Blacklisting domain %s due to 403', domain_added)
                if not ok:
                    if fname.exists():
                        try:
                            fname.unlink()
                        except Exception:
                            pass
                    # mark processed_urls to avoid re-attempting same URL
                    if processed_urls is not None:
                        processed_urls.add(url)
                else:
                    # succeeded
                    count += 1
                    if processed_urls is not None:
                        processed_urls.add(url)
                # occasionally save state
                if state_save_callback and (len(processed_urls) % SAVE_STATE_EVERY == 0):
                    state_save_callback()
        page += 1
        if page > 50:
            break
        time.sleep(0.2 + random.random() * 0.4)
    return


# icrawler wrapper per query (best effort)
def icrawler_download_query(query: str, out_dir: Path, max_num=300, engine='google') -> Path:
    if not _HAS_ICRAWLER:
        raise RuntimeError('icrawler no instalado')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = ''.join(c if c.isalnum() or c in '-_' else '_' for c in query)[:200]
    sub = out_dir / slug
    sub.mkdir(parents=True, exist_ok=True)
    max_num_int = int(max_num)

    tries = 0
    while tries < 3:
        try:
            if engine == 'google':
                crawler = GoogleImageCrawler(storage={'root_dir': str(sub)}, feeder_threads=1, parser_threads=1, downloader_threads=4)
            else:
                crawler = BingImageCrawler(storage={'root_dir': str(sub)}, feeder_threads=1, parser_threads=1, downloader_threads=4)
            logger.info("icrawler: crawling '%s' -> %s (max %d)", query, sub, max_num_int)
            crawler.crawl(keyword=query, max_num=max_num_int, min_size=(100, 100))
            break
        except TypeError as e:
            logger.warning("icrawler parser TypeError for query '%s': %s; retrying...", query, e)
            tries += 1
            time.sleep(0.8 + tries * 0.5)
            continue
        except Exception as e:
            logger.exception("icrawler failed for query '%s' (attempt %d): %s", query, tries + 1, e)
            tries += 1
            time.sleep(0.8 + tries * 0.5)
            continue
    return sub


# -------------------------- Postprocess & dedupe -----------------------

def postprocess_folder(raw_dir: Path, final_dir: Path, phash_index: Dict[str, str], min_size=(64, 64), dedup_hamming=6,
                       keep_meta=True, face_filter=False, verbose=True, processed_urls: Set[str] = None):
    raw_dir = Path(raw_dir)
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in raw_dir.rglob('*') if p.is_file()])
    meta_list = []
    kept = 0

    # rebuild hash_index local view from phash_index
    local_hash_index = dict(phash_index)  # phash->filename

    for p in tqdm(files, desc=f"Postprocess {raw_dir.name}", disable=not verbose):
        try:
            with Image.open(p) as im:
                try:
                    if im.mode != 'RGB':
                        im = im.convert('RGB')
                except Exception:
                    pass
                w, h = im.size
                if w < min_size[0] or h < min_size[1]:
                    meta_list.append({'src': str(p), 'ok': False, 'reason': f'too_small {w}x{h}'})
                    continue
                if face_filter and _HAS_FACE:
                    import numpy as np
                    arr = np.array(im)
                    faces = face_recognition.face_locations(arr, model='hog')
                    if len(faces) == 0:
                        meta_list.append({'src': str(p), 'ok': False, 'reason': 'no_face'})
                        continue
                ph = imagehash.phash(im)
                phs = str(ph)
                # if exists in persistent index, mark as duplicate
                if phs in local_hash_index:
                    meta_list.append({'src': str(p), 'ok': False, 'reason': f'dupe_of:{local_hash_index[phs]}', 'phash': phs})
                    continue
                # check hamming against existing
                is_dup = False
                for existing_ph, fname in local_hash_index.items():
                    hd = imagehash.hex_to_hash(phs) - imagehash.hex_to_hash(existing_ph)
                    if hd <= dedup_hamming:
                        is_dup = True
                        meta_list.append({'src': str(p), 'ok': False, 'reason': f'dupe_of:{fname}', 'phash': phs})
                        break
                if is_dup:
                    continue
                out_name = f'img_{len(local_hash_index):06d}.jpg'
                out_path = final_dir / out_name
                try:
                    im.save(out_path, format='JPEG', quality=92)
                except Exception as e:
                    meta_list.append({'src': str(p), 'ok': False, 'reason': f'save_error:{repr(e)}'})
                    continue
                local_hash_index[phs] = out_name
                phash_index[phs] = out_name  # update persistent index
                meta_list.append({'src': str(p), 'ok': True, 'dest': str(out_path), 'phash': phs, 'width': w, 'height': h})
                kept += 1
                # occasionally save processed_urls if provided
                if processed_urls is not None and (len(processed_urls) % SAVE_STATE_EVERY == 0):
                    # leave saving to caller
                    pass
        except UnidentifiedImageError:
            meta_list.append({'src': str(p), 'ok': False, 'reason': 'unidentified'})
        except Exception as e:
            meta_list.append({'src': str(p), 'ok': False, 'reason': f'error:{repr(e)}'})

    # save metadata summary
    meta_path = final_dir / '_download_meta.json'
    try:
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_list, f, indent=2, ensure_ascii=False)
    except Exception:
        logger.exception('Failed saving meta file %s', meta_path)
    df = pd.DataFrame(meta_list)
    try:
        df.to_csv(final_dir / '_download_summary.csv', index=False)
    except Exception:
        logger.exception('Failed saving csv summary')

    logger.info('Postprocess finished: kept=%d total_files_checked=%d final_index_size=%d', kept, len(files), len(local_hash_index))
    return final_dir, meta_path


# -------------------------- Query builders -----------------------------

def generate_date_ranges(start_year: int, end_year: int, step: int) -> List[Tuple[int, int]]:
    ranges = []
    y = start_year
    while y <= end_year:
        y2 = min(y + step - 1, end_year)
        ranges.append((y, y2))
        y += step
    return ranges


def build_queries(base: str, modifiers: Iterable[str], use_date_ranges: bool, start_year: int, end_year: int, year_step: int) -> List[str]:
    queries = []
    mods = [m.strip() for m in modifiers if m and m.strip()]
    queries.append(base)
    for m in mods:
        queries.append(f"{base} {m}")
    queries = list(dict.fromkeys(queries))
    if use_date_ranges:
        drs = generate_date_ranges(start_year, end_year, year_step)
        ranged_queries = []
        for (a, b) in drs:
            for q in queries:
                # add textual range to diversify; icrawler/serpapi may treat it as extra tokens
                ranged_queries.append(f"{q} {a}..{b}")
        queries = ranged_queries
    return queries


# ---------------------------- Orquestador ------------------------------

def run_for_query(base_query: str, out_root: Path, tag: str, engine: str, num: int, serpapi_key: Optional[str] = None,
                  modifiers: List[str] = (), use_date_ranges=False, start_year=1980, end_year=2025, year_step=10,
                  min_size=(100, 100), dedup_hamming=6, face_filter=False, headers=None, verify=True,
                  resume=True):
    tmp_dir = out_root / f"{tag}_raw"
    final_dir = out_root / f"{tag}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    headers = {**DEFAULT_HEADERS, **(headers or {})}

    phash_index, processed_urls, blacklist = load_state(out_root)
    logger.info('Loaded state: index=%d processed_urls=%d blacklist=%d', len(phash_index), len(processed_urls), len(blacklist))

    queries = build_queries(base_query, modifiers, use_date_ranges, start_year, end_year, year_step)
    logger.info('Built %d queries for base "%s" (engine=%s)', len(queries), base_query, engine)

    per_query_target = max(50, int(math.ceil(num / max(1, len(queries)))))
    total_downloaded = 0

    def _save_cb():
        save_state(out_root, phash_index, processed_urls, blacklist)

    for q in queries:
        if total_downloaded >= num:
            break
        logger.info("Running query: '%s' (target ~%d)", q, per_query_target)
        if engine == 'serpapi':
            if not _HAS_SERPAPI:
                raise RuntimeError('SerpAPI no disponible; instala serpapi o usa icrawler.')
            subdir = tmp_dir / 'serpapi' / ''.join(c if c.isalnum() or c in '-_' else '_' for c in q)[:200]
            serpapi_images_for_query(q, subdir, max_num=per_query_target, serpapi_key=serpapi_key, headers=headers, verify=verify,
                                     phash_index=phash_index, processed_urls=processed_urls, blacklist=blacklist,
                                     state_save_callback=_save_cb)
            downloaded = sum(1 for _ in subdir.glob('**/*') if _.is_file())
            total_downloaded += downloaded
        else:
            try:
                sub = icrawler_download_query(q, tmp_dir, max_num=per_query_target, engine=engine)
                downloaded = sum(1 for _ in sub.glob('**/*') if _.is_file())
                total_downloaded += downloaded
            except Exception as e:
                logger.exception('icrawler failed for query "%s": %s', q, e)
        time.sleep(0.3 + random.random() * 0.7)
        # save state between queries
        _save_cb()

    logger.info('Finished crawling phase. Rough total files in raw_dir=%d', sum(1 for _ in tmp_dir.rglob('*') if _.is_file()))
    # postprocess (this updates the phash_index)
    final, meta = postprocess_folder(tmp_dir, final_dir, phash_index, min_size=min_size, dedup_hamming=dedup_hamming,
                                     keep_meta=True, face_filter=face_filter, verbose=True, processed_urls=processed_urls)
    # final save
    save_state(out_root, phash_index, processed_urls, blacklist)
    return final, meta


# ------------------------------- GUI ----------------------------------

class SimpleGUI:
    def __init__(self, root):
        self.root = root
        root.title('Fetch Images (alpha -> beta)')
        frm = ttk.Frame(root, padding=10)
        frm.grid()

        ttk.Label(frm, text='Positive query').grid(column=0, row=0, sticky='w')
        self.pos = ttk.Entry(frm, width=60)
        self.pos.insert(0, 'Linda Hamilton')
        self.pos.grid(column=1, row=0, sticky='w')

        ttk.Label(frm, text='Negative query').grid(column=0, row=1, sticky='w')
        self.neg = ttk.Entry(frm, width=60)
        self.neg.insert(0, 'middle age woman')
        self.neg.grid(column=1, row=1, sticky='w')

        ttk.Label(frm, text='Num images (total)').grid(column=0, row=2, sticky='w')
        self.num = ttk.Entry(frm, width=10)
        self.num.insert(0, '2000')
        self.num.grid(column=1, row=2, sticky='w')

        ttk.Label(frm, text='Engine').grid(column=0, row=3, sticky='w')
        self.engine = ttk.Combobox(frm, values=['google', 'bing', 'serpapi'], width=12)
        self.engine.set('google')
        self.engine.grid(column=1, row=3, sticky='w')

        ttk.Label(frm, text='Modifiers (comma)').grid(column=0, row=4, sticky='w')
        self.mods = ttk.Entry(frm, width=60)
        self.mods.insert(0, 'face,portrait,headshot,front,studio,close up,highres')
        self.mods.grid(column=1, row=4, sticky='w')

        ttk.Label(frm, text='Out folder').grid(column=0, row=5, sticky='w')
        self.out = ttk.Entry(frm, width=60)
        self.out.insert(0, 'data_linda_alpha')
        self.out.grid(column=1, row=5, sticky='w')
        ttk.Button(frm, text='Browse', command=self.browse_out).grid(column=2, row=5)

        self.use_dates_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text='Use date ranges', variable=self.use_dates_var).grid(column=1, row=6, sticky='w')

        ttk.Button(frm, text='Start', command=self.start).grid(column=1, row=7, sticky='w')
        ttk.Button(frm, text='Open output', command=self.open_out).grid(column=2, row=7)

        self.log = tk.Text(frm, width=100, height=20)
        self.log.grid(column=0, row=8, columnspan=3)

        # wire logger to text widget
        self._orig_logger_handlers = logger.handlers[:]
        handler = TextHandler(self.log)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        self._worker = None

    def browse_out(self):
        d = filedialog.askdirectory()
        if d:
            self.out.delete(0, tk.END)
            self.out.insert(0, d)

    def start(self):
        if self._worker and self._worker.is_alive():
            messagebox.showinfo('Running', 'Task is already running')
            return
        # gather params
        params = dict(
            pos=self.pos.get().strip(),
            neg=self.neg.get().strip(),
            num=int(self.num.get().strip()),
            engine=self.engine.get().strip(),
            out=Path(self.out.get().strip()),
            modifiers=[m.strip() for m in self.mods.get().split(',') if m.strip()],
            use_date_ranges=self.use_dates_var.get(),
        )
        # run in thread to keep UI responsive
        self._worker = threading.Thread(target=self._run_task, args=(params,), daemon=True)
        self._worker.start()

    def _run_task(self, params):
        try:
            logger.info('Starting run with params: %s', params)
            for query, tag in [(params['pos'], 'pos'), (params['neg'], 'neg')]:
                try:
                    final_dir, meta = run_for_query(query, Path(params['out']), tag, engine=params['engine'], num=params['num'],
                                                    serpapi_key=None, modifiers=params['modifiers'], use_date_ranges=params['use_date_ranges'],
                                                    start_year=1980, end_year=2025, year_step=10)
                    logger.info('Finished %s -> %s (meta=%s)', tag, final_dir, meta)
                except Exception as e:
                    logger.exception('Failed for %s: %s', tag, e)
            messagebox.showinfo('Done', 'Run finished (revisa logs)')
        except Exception as e:
            logger.exception('Error in worker: %s', e)

    def open_out(self):
        p = Path(self.out.get().strip())
        if not p.exists():
            messagebox.showerror('No folder', 'Output folder does not exist')
            return
        try:
            if os.name == 'nt':
                os.startfile(str(p))
            elif os.name == 'posix':
                import subprocess
                subprocess.Popen(['xdg-open', str(p)])
            else:
                messagebox.showinfo('Open', f'Open this folder: {p}')
        except Exception:
            messagebox.showinfo('Open', f'Open this folder: {p}')


class TextHandler(logging.Handler):
    # logging handler that writes to Tk Text widget
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


# ------------------------------ CLI -----------------------------------

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', default='Linda Hamilton')
    parser.add_argument('--neg', default='middle age woman')
    parser.add_argument('--num', type=int, default=6000)
    parser.add_argument('--engine', choices=['google', 'bing', 'serpapi'], default='google')
    parser.add_argument('--out', default='data_linda_alpha/raw')
    parser.add_argument('--serpapi_key', default=None)
    parser.add_argument('--min_size', type=int, nargs=2, default=(100, 100), help='min width height')
    parser.add_argument('--dedup_hamming', type=int, default=6, help='hamming distance for phash dedup')
    parser.add_argument('--face_filter', action='store_true', help='keep only images with faces (requires face_recognition)')
    parser.add_argument('--modifiers', default='face,portrait,headshot,front,studio,close up,highres', help='comma separated modifiers to append to the base query')
    parser.add_argument('--use-date-ranges', action='store_true', help='fragment searches across year ranges to bypass the ~1000-result limit')
    parser.add_argument('--start-year', type=int, default=1980)
    parser.add_argument('--end-year', type=int, default=2025)
    parser.add_argument('--year-step', type=int, default=10)
    parser.add_argument('--verify-ssl', action='store_true', help='verify SSL certificates when downloading (safer)')
    parser.add_argument('--referer', default=None, help='optional Referer header to include in requests')
    parser.add_argument('--gui', action='store_true', help='launch GUI (if available)')
    parser.add_argument('--resume', action='store_true', help='resume from previous run state')
    args = parser.parse_args()

    if args.gui:
        if not _HAS_TK:
            logger.error('Tkinter not available on this system')
            return
        root = tk.Tk()
        app = SimpleGUI(root)
        root.mainloop()
        return

    headers = {}
    if args.referer:
        headers['Referer'] = args.referer

    modifiers = [m.strip() for m in args.modifiers.split(',') if m.strip()]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for query, tag in [(args.pos, 'pos'), (args.neg, 'neg')]:
        try:
            final_dir, meta = run_for_query(query, out_root, tag, engine=args.engine, num=args.num,
                                            serpapi_key=args.serpapi_key, modifiers=modifiers,
                                            use_date_ranges=args.use_date_ranges, start_year=args.start_year,
                                            end_year=args.end_year, year_step=args.year_step,
                                            min_size=tuple(args.min_size), dedup_hamming=args.dedup_hamming,
                                            face_filter=args.face_filter, headers=headers, verify=args.verify_ssl,
                                            resume=args.resume)
            logger.info('Finished %s: final_dir=%s, meta=%s', tag, final_dir, meta)
        except Exception as e:
            logger.exception('Failed for %s: %s', tag, e)


if __name__ == '__main__':
    main_cli()
