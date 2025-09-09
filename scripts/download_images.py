#!/usr/bin/env python3
"""
Descarga imágenes desde Google/Bing usando icrawler o SerpAPI.
Uso:
  python scripts/download_images.py --pos "sarah connor" --neg "not sarah connor" --num 300 --engine google
"""
import argparse
import os
from pathlib import Path
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

try:
    from serpapi import GoogleSearch
    HAS_SERPAPI = True
except Exception:
    HAS_SERPAPI = False

import requests


def download_icrawler(query, out_dir, max_num=300, engine='google'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if engine == 'google':
        crawler = GoogleImageCrawler(storage={'root_dir': str(out_dir)})
    else:
        crawler = BingImageCrawler(storage={'root_dir': str(out_dir)})
    crawler.crawl(keyword=query, max_num=max_num, min_size=(100,100))


def download_serpapi(query, out_dir, max_num=300, api_key=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if api_key is None:
        raise ValueError("SerpAPI key required")
    params = {"engine":"google_images", "q": query, "api_key": api_key, "ijn":"0"}
    search = GoogleSearch(params)
    results = search.get_dict()
    images = results.get('images_results', [])
    count = 0
    for i, im in enumerate(images):
        if count >= max_num:
            break
        url = im.get('original') or im.get('thumbnail') or im.get('src')
        if not url:
            continue
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                ext = url.split('?')[0].split('.')[-1][:4]
                if len(ext) > 5 or '/' in ext:
                    ext = 'jpg'
                fname = out_dir / f"{i}.{ext}"
                with open(fname, 'wb') as f:
                    f.write(r.content)
                count += 1
        except Exception:
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', required=True, help='Positive topic')
    parser.add_argument('--neg', required=True, help='Negative topic')
    parser.add_argument('--num', type=int, default=300, help='Images per class')
    parser.add_argument('--engine', choices=['google','bing','serpapi'], default='google')
    parser.add_argument('--out', default='data/raw', help='Output root')
    parser.add_argument('--serpapi_key', default=None, help='SerpAPI key (if engine=serpapi)')
    args = parser.parse_args()

    for label in [(args.pos, 'pos'), (args.neg, 'neg')]:
        query, tag = label
        out_dir = os.path.join(args.out, tag)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Downloading {args.num} images for '{query}' into {out_dir} using {args.engine}")
        if args.engine == 'serpapi':
            if not HAS_SERPAPI:
                raise RuntimeError('serpapi library is not installed')
            download_serpapi(query, out_dir, max_num=args.num, api_key=args.serpapi_key)
        else:
            download_icrawler(query, out_dir, max_num=args.num, engine=args.engine)

if __name__ == '__main__':
    main()
