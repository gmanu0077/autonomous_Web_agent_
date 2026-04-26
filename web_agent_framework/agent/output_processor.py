import json
from urllib.parse import urlparse, parse_qs, urlencode
from typing import Dict, Any, List, Optional
import re

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif")

class OutputProcessor:
    def __init__(self):
        pass

    def extract_product_image_urls(self, stdout: str) -> List[str]:
        urls = self._extract_json_after_marker(stdout, "PRODUCT_IMAGES:")
        if urls:
            return urls
        urls = self._extract_json_after_marker(stdout, "ALL_CANDIDATES:")
        return urls

    def _extract_json_after_marker(self, stdout: str, marker: str) -> List[str]:
        lines = stdout.splitlines()
        urls_json_lines: List[str] = []
        capture = False
        bracket_balance = 0

        for i, line in enumerate(lines):
            if marker in line:
                for j in range(i + 1, len(lines)):
                    stripped = lines[j].strip()
                    if not stripped and not capture:
                        continue
                    if not capture:
                        if stripped.startswith("[") or stripped.startswith("{"):
                            capture = True
                        else:
                            break
                    if capture:
                        urls_json_lines.append(lines[j])
                        bracket_balance += lines[j].count("[") + lines[j].count("{")
                        bracket_balance -= lines[j].count("]") + lines[j].count("}")
                        if bracket_balance <= 0:
                            break
                break

        if not urls_json_lines:
            return []

        try:
            data = json.loads("\n".join(urls_json_lines))
        except Exception:
            return []

        urls: List[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    urls.append(item)
                elif isinstance(item, dict):
                    for key in ("url", "image", "src"):
                        if key in item and isinstance(item[key], str):
                            urls.append(item[key])
                            break
        return urls

    def is_real_image_url(self, url: str) -> bool:
        if not isinstance(url, str):
            return False
        lu = url.strip().lower()
        if not lu.startswith(("http://", "https://")):
            return False

        junk_tokens = [
            "sprite", "favicon", "logo", "icon", "tracking", "analytics",
            "pixel", "loader", "loading", "placeholder", "dummy", "spinner",
            "badge", "arrow", "play", "pause", "share", "facebook", "instagram",
            "twitter", "linkedin", "pinterest",
            "/ads/", "/advert", "/advertisement",
        ]
        if any(tok in lu for tok in junk_tokens):
            return False

        parsed = urlparse(lu)
        path = parsed.path or ""
        if any(path.endswith(ext) for ext in IMAGE_EXTS):
            return True

        qs = parsed.query or ""
        q = parse_qs(qs)
        fmt_vals = q.get("fmt") or q.get("format") or q.get("fm") or q.get("ext") or []
        if fmt_vals:
            if fmt_vals[0].lower() in {"jpeg", "jpg", "png", "webp", "avif", "gif"}:
                return True

        if "/is/image/" in lu:
            return True

        return False

    def clean_product_image_urls(self, raw_urls: List[str], max_images: Optional[int] = None) -> List[str]:
        cleaned: List[str] = []
        seen: set[str] = set()

        for u in raw_urls:
            if not isinstance(u, str): continue
            s = u.strip()
            if not s or not self.is_real_image_url(s) or s in seen:
                continue
            seen.add(s)
            cleaned.append(s)

        if max_images is not None:
            cleaned = cleaned[:max_images]
        return cleaned

    def detect_suspect_step(self, stdout: str, plan: Dict[str, Any], returncode: int) -> Optional[str]:
        if not isinstance(plan, dict): return None
        steps = plan.get("steps", [])
        step_ids = [s.get("id") for s in steps if s.get("id")]
        if not step_ids: return None

        last_ok_idx = -1
        for idx, sid in enumerate(step_ids):
            if f"STEP {sid} OK" in stdout:
                last_ok_idx = idx

        if returncode != 0:
            if last_ok_idx + 1 < len(step_ids):
                return step_ids[last_ok_idx + 1]
            return step_ids[last_ok_idx] if last_ok_idx >= 0 else step_ids[-1]

        return step_ids[-1]
