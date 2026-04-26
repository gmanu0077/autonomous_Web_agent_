import os
import re
from typing import Dict, Any

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def extract_url_from_task(task: str) -> str | None:
    """
    Extract first http(s) URL from the user task string.
    """
    m = re.search(r"https?://[^\s\"')]+", task)
    if m:
        return m.group(0)
    return None

def infer_scrape_intent(task: str) -> Dict[str, bool]:
    """
    Infer whether the user wants product images, product details, or both
    from the free-form task.
    """
    lower = task.lower()

    image_keywords = [
        "image", "images", "photo", "photos", "picture", "pictures",
        "gallery", "thumbnails", "product images", "hero image", "lookbook",
    ]
    detail_keywords = [
        "detail", "details", "product details", "title", "name",
        "price", "pricing", "description", "bullet", "bullets",
        "spec", "specs", "specification", "specifications",
        "size", "colour", "color", "material", "fabric",
        "sku", "product id", "product information", "product info",
    ]

    want_images = any(kw in lower for kw in image_keywords)
    want_details = any(kw in lower for kw in detail_keywords)

    if not want_images and not want_details:
        want_images = True
        want_details = True

    return {
        "want_images": want_images,
        "want_details": want_details,
    }

def build_annotated_task(task: str, intent: Dict[str, bool]) -> str:
    """
    Append a SCRAPE_INTENT block to the original task so the LLM
    has explicit, structured info.
    """
    want_images = bool(intent.get("want_images", True))
    want_details = bool(intent.get("want_details", False))

    lines = [
        task,
        "",
        "SCRAPE_INTENT:",
        f"- want_product_images: {want_images}",
        f"- want_product_details: {want_details}",
        f"- want_both: {want_images and want_details}",
    ]
    return "\n".join(lines)
