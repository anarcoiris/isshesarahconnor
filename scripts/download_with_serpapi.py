# scripts/download_with_serpapi.py (esquema)
from serpapi import GoogleSearch
import requests, os

def download(query, out_dir, max_num=300, api_key="YOUR_KEY"):
    os.makedirs(out_dir, exist_ok=True)
    params = {"engine":"google_images", "q":query, "api_key":api_key, "ijn":"0"}
    search = GoogleSearch(params)
    results = search.get_dict()
    images = results.get("images_results", [])
    for i, im in enumerate(images[:max_num]):
        url = im.get("original")
        if not url: url = im.get("thumbnail")
        ext = url.split("?")[0].split(".")[-1][:4]
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(os.path.join(out_dir, f"{i}.{ext}"), "wb") as f:
                    f.write(r.content)
        except Exception as e:
            print("skip", url, e)
