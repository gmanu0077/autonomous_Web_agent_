import uuid
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
from bs4 import BeautifulSoup, Tag
from .url_utils import normalize_url, get_image_family_key, get_url_res_score
from .image_promotion import strip_sizing_modifiers


def plot_dom_graph(G: nx.DiGraph, output_path: str = "graph.png", max_nodes: int = 150) -> None:
    """Save DOM graph visualization to PNG. Requires matplotlib."""
    if len(G.nodes) == 0:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    nodes_to_plot = sorted(G.nodes(data=True), key=lambda x: x[1].get("depth", 0))[:max_nodes]
    node_ids = [n[0] for n in nodes_to_plot]
    subG = G.subgraph(node_ids)
    plt.figure(figsize=(20, 12))
    pos = {}
    depth_counts = {}
    for n_id, data in nodes_to_plot:
        d = data.get("depth", 0)
        depth_counts[d] = depth_counts.get(d, 0) + 1
        pos[n_id] = (depth_counts[d], -d)
    labels = {n_id: f"{G.nodes[n_id].get('tag', '??')}" for n_id in subG.nodes()}
    nx.draw(subG, pos, labels=labels, with_labels=True, node_size=1500, node_color="skyblue", font_size=7, arrows=True)
    plt.title(f"DOM Graph ({len(subG.nodes)} nodes)")
    plt.savefig(output_path)
    plt.close()

def _build_node_searchable(tag: str, attrs: dict, text: str) -> str:
    """Build a searchable string for node retrieval (Phase 2)."""
    parts = [tag]
    if attrs:
        for k, v in attrs.items():
            if v and isinstance(v, str):
                parts.append(f"{k}={v}")
            elif v and isinstance(v, list):
                parts.append(f"{k}={' '.join(str(x) for x in v)}")
    if text:
        parts.append(text[:300].replace('\n', ' '))
    return " ".join(parts).lower()


def _build_node_selector(tag: str, attrs: dict) -> Optional[str]:
    """Build a CSS selector for the node when possible."""
    if not attrs:
        return None
    elem_id = attrs.get('id') if isinstance(attrs.get('id'), str) else None
    if elem_id and all(c.isalnum() or c in '-_' for c in elem_id):
        return f"#{elem_id}"
    cls = attrs.get('class')
    if cls:
        cls_str = " ".join(cls) if isinstance(cls, list) else str(cls)
        first_cls = (cls_str.split()[0] if cls_str.split() else None)
        if first_cls and all(c.isalnum() or c in '-_' for c in first_cls):
            return f"{tag}.{first_cls}"
    itemprop = attrs.get('itemprop')
    if itemprop:
        return f"{tag}[itemprop=\"{itemprop}\"]"
    return None


def _infer_semantic_hint(tag: str, attrs: dict, text: str) -> Optional[str]:
    """Infer semantic label for easier retrieval (comments, video-title, related, etc.)."""
    combined = _build_node_searchable(tag, attrs, text)
    hints = [
        ("comment", "comments"), ("video-title", "video-title"), ("related", "related-videos"),
        ("upnext", "related-videos"), ("watch7-content", "video-content"), ("primary-info", "video-info"),
        ("secondary-info", "video-info"), ("title", "title"), ("description", "description"),
        ("price", "price"), ("add-to-cart", "add-to-cart"), ("gallery", "gallery"),
        ("channel", "channel"), ("owner", "channel"), ("subscribe", "subscribe"),
        ("count", "view-count"), ("menu", "menu"), ("info-container", "info"),
        ("product", "product"), ("plp", "product"), ("prod-", "product"), ("item", "product"),
        ("product-card", "product"), ("product-grid", "product"), ("product-list", "product"),
        ("href", "link"), ("url", "link"), ("/p/", "product-link"),
        ("listing", "product"), ("offer", "product"), ("carousel", "product"),
        ("card__link", "link"), ("item-card", "product"),
        ("vl-video-card__img", "gallery"), ("img", "gallery"), ("product-img", "gallery"),
    ]
    for keyword, hint in hints:
        if keyword in combined:
            return hint
    return None


class DOMGraphBuilder:
    def __init__(self, max_depth: int = 40):
        self.max_depth = max_depth
        self.G = nx.DiGraph()

    def build(self, html: str) -> nx.DiGraph:
        if not html: return self.G
        soup = BeautifulSoup(html, 'lxml')
        for s in soup(['script', 'style', 'noscript', 'svg', 'path', 'iframe']): s.decompose()
        body = soup.find('body') or soup
        self._traverse(body, None, 0)
        return self.G

    def _traverse(self, element: Any, parent_id: Optional[str], depth: int):
        if depth > self.max_depth or not isinstance(element, Tag): return
        tag_name = element.name.lower()
        node_id = str(uuid.uuid4())
        attrs = {k: v for k, v in element.attrs.items() if k in ['id', 'class', 'data-testid', 'role', 'itemprop', 'aria-label']}
        if 'class' in attrs and isinstance(attrs['class'], list): attrs['class'] = " ".join(attrs['class'])
        direct_text = "".join([t for t in element.find_all(string=True, recursive=False)]).strip()
        
        node_props = {'tag': tag_name, 'attrs': attrs, 'text': direct_text[:1000], 'depth': depth}
        node_props['searchable'] = _build_node_searchable(tag_name, attrs, direct_text)
        sel = _build_node_selector(tag_name, attrs)
        if sel: node_props['selector'] = sel
        hint = _infer_semantic_hint(tag_name, attrs, direct_text)
        if hint: node_props['semantic_hint'] = hint
        if tag_name == 'img':
            node_props['src'] = element.get('src')
            node_props['data_src'] = element.get('data-src') or element.get('data-lazy-src') or element.get('data-zoom-image')
            node_props['srcset'] = element.get('srcset') or element.get('data-srcset')
            node_props['alt'] = element.get('alt', '')
        elif tag_name == 'a':
            node_props['href'] = element.get('href', '')

        self.G.add_node(node_id, **node_props)
        if parent_id: self.G.add_edge(parent_id, node_id)
        for child in element.children:
            if isinstance(child, Tag): self._traverse(child, node_id, depth + 1)

def perform_smart_chunking(G: nx.DiGraph, url: str) -> List[Dict[str, Any]]:
    chunks = []
    visited = set()

    def get_subtree_data(node_id):
        descendants = nx.descendants(G, node_id)
        descendants.add(node_id)
        total_text = []
        imgs = []
        total_area = 0
        harvested_in_subtree = set()
        
        for d_id in descendants:
            node = G.nodes[d_id]
            txt = (node.get('text') or '').strip()
            if txt: total_text.append(txt)
            rect = node.get('rect', {})
            total_area += rect.get('area', 0)

            tag = node.get('tag')
            if tag == 'img':
                src = node.get('src') or node.get('data_src')
                if src:
                    imgs.append({
                        'src': src,
                        'data_src': node.get('data_src'),
                        'srcset': node.get('srcset'),
                        'alt': node.get('alt', ''),
                        'rect': node.get('rect', {}),
                        'node_id': d_id,
                        'attrs': node.get('attrs', {})
                    })
            elif node.get('bgUrl'):
                imgs.append({
                    'src': node['bgUrl'],
                    'rect': node.get('rect', {}),
                    'node_id': d_id,
                    'is_bg': True
                })
        return " ".join(total_text), imgs, total_area, list(harvested_in_subtree)

    def chunk_walker(node_id):
        if node_id in visited: return
        node = G.nodes[node_id]
        tag = node.get('tag', '').lower()
        
        if tag in ['html', 'body', 'main', 'article', 'section', 'div'] and (node_id == "node_0" or node.get('rect', {}).get('area', 0) > 1500000):
            for child in G.successors(node_id): 
                chunk_walker(child)
            return

        full_text, imgs, subtree_area, harvested_urls = get_subtree_data(node_id)
        text_len = len(full_text)
        img_count = len(imgs) + len(harvested_urls)
        
        attrs_str = str(node.get('attrs', {})).lower()
        full_text_lower = full_text.lower()
        
        is_gallery_container = any(k in attrs_str for k in ['gallery', 'carousel', 'pdp-images', 'main-image', 'imgtagwrapper', 'image-block', 'ppd', 'leftcol', 'centercol', 'rightcol', 'product-images', 'pimg'])
        noise_keywords = ['related', 'similar', 'bought', 'bought together', 'customers also', 'viewed together', 'frequently', 'suggested', 'you may like', 'also bought', 'people also', 'upsell', 'cross-sell', 'recommendation', 'others also', 'best seller', 'trending']
        is_noise_section = any(k in attrs_str for k in noise_keywords) or any(k in full_text_lower for k in noise_keywords)
        is_grid = any(k in attrs_str for k in ['grid', 'product-grid', 'product-items', 'product-list'])
        
        rect = node.get('rect', {})
        node_area = rect.get('area', 0)
        node_top = rect.get('top', 0)
        
        is_good_chunk = False
        if is_gallery_container and 0 < img_count < 40:
            is_good_chunk = True
        elif is_noise_section or is_grid:
            is_good_chunk = False 
        elif 40000 < node_area < 1200000 and (text_len > 100 or 0 < img_count < 25):
            is_good_chunk = True
        elif 200 < text_len < 4000:
            is_good_chunk = True
        
        if node_area > 1500000 or img_count > 60:
            is_good_chunk = False

        if is_good_chunk:
            gallery_boost = 2000000 if is_gallery_container else 0
            spatial_score = 0
            if 0 <= node_top <= 1500:
                spatial_score = 1000000 - (node_top * 500)
            else:
                spatial_score = -abs(node_top) * 100
            
            importance = (img_count * 100000) + (node_area / 5) + gallery_boost + spatial_score
            if is_noise_section or is_grid:
                importance -= 10000000
            
            chunk_searchable = (node.get('searchable') or '') + ' ' + full_text[:500].lower()
            chunks.append({
                'id': node_id, 
                'tag': node['tag'], 
                'attrs': node['attrs'],
                'rect': rect,
                'text': full_text, 
                'images': imgs,
                'harvested_urls': harvested_urls,
                'visual_importance': importance,
                'jqueryData': node.get('jqueryData'),
                'searchable': chunk_searchable.strip(),
                'selector': node.get('selector'),
                'semantic_hint': node.get('semantic_hint'),
            })
            visited.add(node_id)
            for d in nx.descendants(G, node_id): visited.add(d)
            return
        
        for child in G.successors(node_id): 
            chunk_walker(child)

    roots = [n for n, d in G.in_degree() if d == 0]
    for root in roots: chunk_walker(root)

    # Fallback when no rect data (e.g. from DOMGraphBuilder/HTML): chunk by text
    if not chunks and len(G.nodes) > 0:
        chunks = _fallback_text_chunking(G, url)
    return sorted(chunks, key=lambda x: x.get('visual_importance', 0), reverse=True)


def _fallback_text_chunking(G: nx.DiGraph, url: str) -> List[Dict[str, Any]]:
    """Chunk graph when nodes lack rect (e.g. built from HTML). Uses text and structure."""
    chunks = []
    visited = set()

    def get_subtree_text_imgs(node_id):
        descendants = nx.descendants(G, node_id)
        descendants.add(node_id)
        total_text, imgs = [], []
        for d_id in descendants:
            node = G.nodes[d_id]
            txt = (node.get("text") or "").strip()
            if txt:
                total_text.append(txt)
            if node.get("tag") == "img":
                src = node.get("src") or node.get("data_src")
                if src:
                    imgs.append({"src": src, "node_id": d_id, "attrs": node.get("attrs", {})})
        return " ".join(total_text), imgs

    def walk(node_id, depth=0):
        if node_id in visited or depth > 25:
            return
        node = G.nodes[node_id]
        tag = (node.get("tag") or "").lower()
        if tag in ("html", "head", "script", "style", "meta", "link"):
            for c in G.successors(node_id):
                walk(c, depth + 1)
            return
        full_text, imgs = get_subtree_text_imgs(node_id)
        text_len = len(full_text)
        if 50 < text_len < 15000:
            chunk_searchable = (node.get("searchable") or "") + " " + full_text[:500].lower()
            chunks.append({
                "id": node_id,
                "tag": node.get("tag", "?"),
                "attrs": node.get("attrs", {}),
                "rect": {},
                "text": full_text[:8000],
                "images": imgs,
                "harvested_urls": [],
                "visual_importance": text_len + len(imgs) * 100,
                "searchable": chunk_searchable.strip(),
                "selector": node.get("selector"),
                "semantic_hint": node.get("semantic_hint"),
            })
            visited.add(node_id)
            for d in nx.descendants(G, node_id):
                visited.add(d)
            return
        for c in G.successors(node_id):
            walk(c, depth + 1)

    roots = [n for n, d in G.in_degree() if d == 0]
    for root in roots:
        walk(root)
    return chunks


def get_chunks_for_step(chunks: List[Dict[str, Any]], step_id: str, goal: str, user_demand: str = "") -> List[Dict[str, Any]]:
    """Filter chunks relevant to a specific step. Step-specific keywords improve selector relevance."""
    step_keywords = {
        "scroll": ["scroll", "lazy", "load", "content", "container", "section"],
        "scroll_page": ["scroll", "lazy", "load", "content", "container"],
        "scroll_to_load": ["scroll", "lazy", "load"],
        "get_product_links": ["product", "link", "href", "url", "item", "plp", "card", "grid"],
        "extract_product": ["product", "price", "title", "item", "card"],
        "get_comments": ["comment", "comments"],
        "get_related": ["related", "video", "sidebar", "secondary"],
        "get_video": ["video", "title", "metadata", "description"],
    }
    q = f"{step_id} {goal} {user_demand}".lower()
    keywords = step_keywords.get(step_id.lower(), []) + q.split()
    scored = []
    for c in chunks:
        searchable = (c.get("searchable") or "").lower()
        hint = (c.get("semantic_hint") or "").lower()
        text = (c.get("text") or "").lower()
        attrs_str = str(c.get("attrs") or {}).lower()
        combined = f"{searchable} {hint} {text} {attrs_str}"
        score = 0
        for kw in keywords:
            if len(kw) > 2 and kw in combined:
                score += 10
        if hint and any(kw in hint for kw in keywords if len(kw) > 2):
            score += 50
        if score > 0:
            scored.append((score, c))
    if scored:
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:25]]
    return chunks[:25]


def get_chunks_for_demand(chunks: List[Dict[str, Any]], demand: str) -> List[Dict[str, Any]]:
    """Filter chunks by user demand for Phase 2 retrieval. Returns chunks whose searchable or semantic_hint matches."""
    if not demand or not chunks:
        return chunks
    q = demand.lower().strip()
    scored = []
    for c in chunks:
        searchable = (c.get("searchable") or "").lower()
        hint = (c.get("semantic_hint") or "").lower()
        text = (c.get("text") or "").lower()
        score = 0
        if q in hint:
            score += 100
        if q in searchable:
            score += 50
        if q in text:
            score += 10
        for word in q.split():
            if len(word) > 2 and (word in hint or word in searchable or word in text):
                score += 5
        if score > 0:
            scored.append((score, c))
    if not scored:
        return chunks
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:20]]
