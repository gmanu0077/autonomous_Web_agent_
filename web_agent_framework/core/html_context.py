import os
import glob
import pickle
import re
from collections import Counter
from ..config import MAX_HTML_CONTEXT_CHARS, HTML_CHUNK_SIZE
from .utils import ensure_dir

def chunk_and_pickle_html(html: str, html_dir: str, chunk_size: int = HTML_CHUNK_SIZE) -> None:
    ensure_dir(html_dir)
    raw_path = os.path.join(html_dir, "raw.html")
    with open(raw_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(html)

    n = 0
    for i in range(0, len(html), chunk_size):
        chunk = html[i: i + chunk_size]
        fname = os.path.join(html_dir, f"html_chunk_{n:04d}.pkl")
        with open(fname, "wb") as f:
            pickle.dump(chunk, f)
        n += 1
    print(f"[BOOTSTRAP] Saved {n} HTML chunks into {html_dir}")

def _reconstruct_raw_html_from_pickles(html_dir: str) -> str:
    raw_path = os.path.join(html_dir, "raw.html")
    if os.path.exists(raw_path):
        try:
            with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_html = f.read()
            if raw_html.strip():
                return raw_html
        except Exception as e:
            print(f"[HTML CONTEXT] Failed to read raw.html ({e}), falling back to pickle chunks.")
    
    files = sorted(glob.glob(os.path.join(html_dir, "html_chunk_*.pkl")))
    if not files:
        return ""

    all_chunks: list[str] = []
    for path in files:
        try:
            with open(path, "rb") as f:
                chunk = pickle.load(f)
            if not isinstance(chunk, str):
                chunk = str(chunk)
            all_chunks.append(chunk)
        except Exception:
            continue

    return "".join(all_chunks)

def build_html_context_from_pickles(html_dir: str, max_chars: int = MAX_HTML_CONTEXT_CHARS) -> str:
    raw_html = _reconstruct_raw_html_from_pickles(html_dir)
    return raw_html if raw_html else "(No HTML available.)"

def build_html_windows_from_pickles(
    html_dir: str,
    window_size: int = MAX_HTML_CONTEXT_CHARS,
    overlap: int = 2000,
) -> list[str]:
    raw_html = _reconstruct_raw_html_from_pickles(html_dir)
    if not raw_html:
        return []

    window_size = max(2000, window_size)
    overlap = max(0, min(overlap, window_size - 1))

    windows: list[str] = []
    n = len(raw_html)
    start = 0

    while start < n:
        end = min(start + window_size, n)
        windows.append(raw_html[start:end])
        if end >= n:
            break
        start = end - overlap
    return windows

def iter_pickled_html_chunks(html_dir: str):
    files = sorted(glob.glob(os.path.join(html_dir, "html_chunk_*.pkl")))
    for idx, path in enumerate(files):
        try:
            with open(path, "rb") as f:
                chunk = pickle.load(f)
            if not isinstance(chunk, str):
                chunk = str(chunk)
            yield idx, path, chunk
        except Exception:
            continue

def build_dom_js_hints(html: str, max_items: int = 25) -> str:
    if not html:
        return "(No DOM / JS hints; HTML snapshot empty.)"

    class_counter = Counter()
    id_counter = Counter()

    for m in re.finditer(r'class="([^"]+)"', html):
        for c in m.group(1).strip().split():
            class_counter[c] += 1
    for m in re.finditer(r"class='([^']+)'", html):
        for c in m.group(1).strip().split():
            class_counter[c] += 1
    for m in re.finditer(r'id="([^"]+)"', html):
        id_counter[m.group(1).strip()] += 1
    for m in re.finditer(r"id='([^']+)'", html):
        id_counter[m.group(1).strip()] += 1

    interesting_keywords = ("product", "pdp", "gallery", "image", "media", "slider", "carousel", "zoom", "thumb", "thumbnail", "photo", "picture")

    def pick_interesting(counter: Counter) -> list[str]:
        items = []
        for token, count in counter.most_common():
            if any(k in token.lower() for k in interesting_keywords):
                items.append(f"{token} (x{count})")
            if len(items) >= max_items:
                break
        return items

    interesting_classes = pick_interesting(class_counter)
    interesting_ids = pick_interesting(id_counter)

    js_hints = []
    if "jquery" in html.lower() or "$(" in html: js_hints.append("jQuery detected")
    if "__NEXT_DATA__" in html: js_hints.append("__NEXT_DATA__ (Next.js) detected")
    if "__NUXT__" in html: js_hints.append("__NUXT__ (Nuxt/Vue) detected")
    if "application/ld+json" in html: js_hints.append("JSON-LD scripts present")
    if "window.__INITIAL_STATE__" in html: js_hints.append("window.__INITIAL_STATE__ detected")

    blocker_hints = []
    lh = html.lower()
    if "captcha" in lh or "opfcaptcha" in lh: blocker_hints.append("CAPTCHA / bot-detection present.")
    if "automated access" in lh: blocker_hints.append("Mentions of 'automated access'.")
    if "continue shopping" in lh: blocker_hints.append("Possible interstitial 'Continue shopping' text.")

    out = []
    if interesting_classes:
        out.append("LIKELY INTERESTING CLASS TOKENS:")
        out.extend(f"  - {c}" for c in interesting_classes)
    if interesting_ids:
        out.append("\nLIKELY INTERESTING ID TOKENS:")
        out.extend(f"  - {i}" for i in interesting_ids)
    if js_hints:
        out.append("\nJS / APP STATE HINTS:")
        out.extend(f"  - {h}" for h in js_hints)
    if blocker_hints:
        out.append("\nBLOCKER / BOT-DETECTION HINTS:")
        out.extend(f"  - {h}" for h in blocker_hints)

    return "\n".join(out) if out else "(No interesting hints found.)"
