"""
Requirement-driven graph query for step code generation.

Flow: LLM decides what's needed → Query graph (generic search) → Code (MCP + retrieved data + LLM).

No hardcoding: the LLM interprets the user demand and step goal, returns requirements (element_types,
attrs, keywords, etc.), and we use those to search the graph. On debug/failure, LLM plans what else
might be needed → re-query graph → generate new code.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

REQUIREMENTS_SCHEMA = {
    "element_types": [],
    "attrs": [],
    "keywords": [],
    "semantic_hints": [],
    "data_schema": "",
    "selectors_for": "",
    "child_selector_hint": "",
}


async def extract_requirements_via_llm(
    step_id: str,
    goal: str,
    user_demand: str,
    graph_summary: str,
    llm_adapter: Any,
) -> Dict[str, Any]:
    """
    Ask the LLM what DOM elements, attributes, and data this step needs.
    Returns a requirements dict used to query the graph. No hardcoded heuristics.
    """
    prompt = f"""You are analyzing a web page to fulfill a user demand. For this step, determine what DOM elements and structure the code will need.

USER DEMAND: {user_demand}
STEP_ID: {step_id}
STEP GOAL: {goal}

GRAPH SUMMARY (available page structure — tags, selectors, hints):
{graph_summary[:3000]}

Return ONLY valid JSON (no markdown, no explanation) with these keys:
- element_types: list of HTML tag names (e.g. ["a", "img", "div", "span"])
- attrs: list of attributes to look for (e.g. ["href", "src", "data-src", "alt", "class"])
- keywords: list of terms that might appear in class names, ids, or text (e.g. ["product", "image", "card", "link"])
- semantic_hints: list of semantic labels (e.g. ["product", "link", "gallery", "comments"])
- data_schema: string describing expected output shape (e.g. "list of {{url: str, title: str}}")
- selectors_for: short label for what we're selecting (e.g. "product_images", "product_links", "scroll_target")
- child_selector_hint: if we need children inside containers, what tag/selector (e.g. "a", "img", or "")

Be generic and universal — no site-specific selectors. Infer from the demand and goal what elements to target."""

    try:
        resp = await llm_adapter.generate(prompt, system_prompt="You output only valid JSON. No markdown, no code blocks.")
        raw = resp.strip()
        if "```" in raw:
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            if m:
                raw = m.group(1).strip()
        data = json.loads(raw)
        out = dict(REQUIREMENTS_SCHEMA)
        for k in out:
            if k in data and data[k] is not None:
                out[k] = data[k] if isinstance(data[k], type(out[k])) else out[k]
        return out
    except Exception:
        return _minimal_fallback_requirements(step_id, goal, user_demand)


async def plan_debug_requirements(
    step_id: str,
    goal: str,
    error: str,
    previous_requirements: Dict[str, Any],
    graph_summary: str,
    llm_adapter: Any,
) -> Dict[str, Any]:
    """
    When a step fails: ask LLM what else might be needed. Returns updated/expanded requirements
    to re-query the graph and generate new code.
    """
    prompt = f"""A web automation step failed. We need to expand or adjust what we're querying from the page graph.

STEP_ID: {step_id}
STEP GOAL: {goal}
ERROR: {error}

PREVIOUS REQUIREMENTS (what we asked for):
{json.dumps(previous_requirements, indent=2)}

GRAPH SUMMARY:
{graph_summary[:2500]}

Return ONLY valid JSON with the SAME keys as requirements. Expand or change them based on the error:
- element_types, attrs, keywords, semantic_hints, data_schema, selectors_for, child_selector_hint

Consider: wrong element types? missing attributes? need different keywords? need containers vs leaf nodes?
Be generic — no site-specific selectors."""

    try:
        resp = await llm_adapter.generate(prompt, system_prompt="You output only valid JSON. No markdown.")
        raw = resp.strip()
        if "```" in raw:
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            if m:
                raw = m.group(1).strip()
        data = json.loads(raw)
        out = dict(previous_requirements)
        for k in REQUIREMENTS_SCHEMA:
            if k in data and data[k] is not None:
                out[k] = data[k]
        return out
    except Exception:
        return previous_requirements  # Keep previous on parse failure


def _minimal_fallback_requirements(step_id: str, goal: str, user_demand: str) -> Dict[str, Any]:
    """Fallback when LLM is unavailable: use demand words as keywords only."""
    combined = f"{(step_id or '')} {(goal or '')} {(user_demand or '')}".lower()
    words = [w for w in combined.split() if len(w) > 2][:8]
    return {
        "element_types": ["div", "a", "span", "img", "button"],
        "attrs": ["class", "id", "href", "src"],
        "keywords": words or ["content", "item", "link"],
        "semantic_hints": [],
        "data_schema": "extract relevant data as JSON",
        "selectors_for": "generic",
        "child_selector_hint": "",
    }


def _build_graph_summary(graph_chunks: List[Dict[str, Any]], graph_nodes: List[Dict[str, Any]], max_chars: int = 2500) -> str:
    """Build a brief summary of graph structure for the LLM."""
    lines = []
    seen_tags = set()
    for c in (graph_chunks or [])[:20]:
        tag = c.get("tag", "")
        sel = c.get("selector")
        hint = c.get("semantic_hint")
        attrs = c.get("attrs") or {}
        has_imgs = bool(c.get("images"))
        parts = [f"tag={tag}"]
        if sel:
            parts.append(f"selector={sel}")
        if hint:
            parts.append(f"hint={hint}")
        if attrs.get("id"):
            parts.append(f"id={attrs['id']}")
        if has_imgs:
            parts.append("has_images=true")
        lines.append(" ".join(parts))
        seen_tags.add(tag)
    for n in (graph_nodes or [])[:50]:
        tag = n.get("tag", "")
        if tag and tag not in seen_tags:
            seen_tags.add(tag)
            lines.append(f"tag={tag} (node)")
        if len("\n".join(lines)) > max_chars:
            break
    return "\n".join(lines) if lines else "No graph data available."


def query_graph_for_requirements(
    graph_chunks: List[Dict[str, Any]],
    requirements: Dict[str, Any],
    max_selectors: int = 15,
) -> Dict[str, Any]:
    """
    Query graph chunks for nodes matching the requirements. Generic scoring — no hardcoded branches.
    """
    if not graph_chunks:
        return {
            "selectors": [],
            "data_schema": requirements.get("data_schema", ""),
            "sample_nodes": [],
            "message": "No graph chunks available.",
        }

    element_types = set((requirements.get("element_types") or []) + ["div", "a", "span", "button", "img"])
    keywords = set((k.lower() for k in requirements.get("keywords") or []))
    semantic_hints = set((h.lower() for h in requirements.get("semantic_hints") or []))
    attrs_needed = set(requirements.get("attrs") or [])

    scored: List[tuple] = []
    seen_selectors: set = set()

    for c in graph_chunks:
        tag = (c.get("tag") or "").lower()
        hint = (c.get("semantic_hint") or "").lower()
        searchable = (c.get("searchable") or "").lower()
        text = (c.get("text") or "").lower()
        attrs = c.get("attrs") or {}
        attrs_str = str(attrs).lower()

        score = 0
        if tag in element_types:
            score += 20
        if hint and hint in semantic_hints:
            score += 80
        for kw in keywords:
            if kw in searchable or kw in text or kw in attrs_str:
                score += 15
        if attrs_needed and any(a in attrs for a in attrs_needed):
            score += 25
        if c.get("href"):
            score += 30
        if c.get("selector"):
            score += 10
        if c.get("images"):
            score += 25

        if score > 0:
            sel = c.get("selector")
            if not sel and attrs:
                aid = attrs.get("id")
                acls = attrs.get("class")
                if isinstance(acls, list):
                    acls = " ".join(acls) if acls else ""
                if aid and isinstance(aid, str) and all(x.isalnum() or x in "-_" for x in aid):
                    sel = f"#{aid}"
                elif acls:
                    first = (acls.split()[0] if isinstance(acls, str) else (acls[0] if acls else None))
                    if first and all(x.isalnum() or x in "-_" for x in str(first)):
                        sel = f"{tag}.{first}"
            if sel and sel not in seen_selectors:
                seen_selectors.add(sel)
                scored.append((score, c))

    scored.sort(key=lambda x: -x[0])
    result_chunks = [c for _, c in scored[:max_selectors]]

    selectors = []
    sample_nodes = []
    for c in result_chunks:
        sel = c.get("selector")
        if not sel:
            attrs = c.get("attrs") or {}
            aid = attrs.get("id")
            acls = attrs.get("class")
            if isinstance(acls, list):
                acls = " ".join(acls) if acls else ""
            if aid and isinstance(aid, str) and all(x.isalnum() or x in "-_" for x in aid):
                sel = f"#{aid}"
            elif acls:
                first = (acls.split()[0] if isinstance(acls, str) else (acls[0] if acls else None))
                if first:
                    tag = c.get("tag", "div")
                    sel = f"{tag}.{first}"
        if sel and sel not in selectors:
            selectors.append(sel)
        sample_nodes.append({
            "tag": c.get("tag"),
            "selector": c.get("selector"),
            "hint": c.get("semantic_hint"),
            "class": (c.get("attrs") or {}).get("class"),
            "href": c.get("href"),
        })

    # Add img selectors from chunks that have images (when img in element_types)
    if "img" in element_types:
        for c in result_chunks:
            for img in (c.get("images") or [])[:2]:
                attrs = img.get("attrs") or {}
                acls = attrs.get("class")
                if acls:
                    cls_str = acls if isinstance(acls, str) else " ".join(acls) if acls else ""
                    first_cls = (cls_str.split()[0] if cls_str.split() else None)
                    if first_cls and all(x.isalnum() or x in "-_" for x in str(first_cls)[:50]):
                        sel = f"img.{first_cls}"
                        if sel not in selectors:
                            selectors.append(sel)

    child_hint = requirements.get("child_selector_hint") or ""

    return {
        "selectors": selectors[:max_selectors],
        "data_schema": requirements.get("data_schema", ""),
        "sample_nodes": sample_nodes[:10],
        "selectors_for": requirements.get("selectors_for", ""),
        "child_selector_hint": child_hint or None,
        "message": f"Found {len(selectors)} selectors.",
    }


def _load_graph_nodes(job_dir: str) -> List[Dict[str, Any]]:
    """Load all nodes from graph.json (full graph). Returns list of node dicts."""
    path = os.path.join(job_dir, "graph.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("nodes", [])
    except Exception:
        return []


def _nodes_to_selectors(nodes: List[Dict[str, Any]], requirements: Dict[str, Any], max_n: int = 20) -> List[str]:
    """Extract selectors from graph nodes matching requirements. Generic scoring."""
    element_types = set(requirements.get("element_types") or [])
    keywords = set((k.lower() for k in requirements.get("keywords") or []))
    semantic_hints = set((h.lower() for h in requirements.get("semantic_hints") or []))
    attrs_needed = set(requirements.get("attrs") or [])

    scored: List[tuple] = []
    seen: set = set()

    for n in nodes:
        tag = (n.get("tag") or "").lower()
        hint = (n.get("semantic_hint") or "").lower()
        searchable = (n.get("searchable") or "").lower()
        text = (n.get("text") or "").lower()
        attrs = n.get("attrs") or {}
        attrs_str = str(attrs).lower()
        href = n.get("href", "")
        src = n.get("src") or n.get("data_src")

        score = 0
        if tag in element_types:
            score += 30
        if tag == "a" and href and "javascript:" not in str(href).lower()[:20]:
            score += 40
        if tag == "img" and (src or n.get("data_src")):
            score += 40
        if hint and hint in semantic_hints:
            score += 80
        for kw in keywords:
            if kw in searchable or kw in text or kw in attrs_str or (href and kw in str(href).lower()):
                score += 15
        if attrs_needed and any(a in attrs for a in attrs_needed):
            score += 20
        if n.get("selector"):
            score += 10

        if score > 0:
            sel = n.get("selector")
            if not sel and attrs:
                aid = attrs.get("id")
                acls = attrs.get("class")
                if isinstance(acls, list):
                    acls = " ".join(acls) if acls else ""
                if aid and isinstance(aid, str) and all(x.isalnum() or x in "-_" for x in aid):
                    sel = f"#{aid}"
                elif acls:
                    first = (acls.split()[0] if isinstance(acls, str) else (acls[0] if acls else None))
                    if first and all(x.isalnum() or x in "-_" for x in str(first)):
                        sel = f"{tag}.{first}"
            if sel and sel not in seen:
                seen.add(sel)
                scored.append((score, n))

    scored.sort(key=lambda x: -x[0])
    out = []
    for _, n in scored[:max_n]:
        s = n.get("selector") or _build_selector(n)
        if s and s not in out:
            out.append(s)
    return out


def _build_selector(n: Dict[str, Any]) -> Optional[str]:
    """Build CSS selector from node attrs. Tries id, class, data-testid, itemprop, role."""
    tag = n.get("tag", "div")
    attrs = n.get("attrs") or {}
    aid = attrs.get("id")
    acls = attrs.get("class")
    if isinstance(acls, list):
        acls = " ".join(acls) if acls else ""
    if aid and isinstance(aid, str) and all(x.isalnum() or x in "-_" for x in aid):
        return f"#{aid}"
    if acls:
        for part in (acls.split() if isinstance(acls, str) else acls):
            if part and all(x.isalnum() or x in "-_" for x in str(part)[:50]):
                return f"{tag}.{part}"
    for attr, fmt in [("data-testid", '[data-testid="{}"]'), ("itemprop", '[itemprop="{}"]'), ("role", '[role="{}"]')]:
        val = attrs.get(attr)
        if val and isinstance(val, str) and len(val) < 80:
            return f"{tag}{fmt.format(val)}"
    return None


async def query_graph_for_step(
    job_dir: str,
    step_id: str,
    goal: str,
    user_demand: str = "",
    graph_chunks: Optional[List[Dict[str, Any]]] = None,
    llm_adapter: Optional[Any] = None,
    previous_requirements: Optional[Dict[str, Any]] = None,
    debug_error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entry: LLM decides requirements → query graph (generic) → return structured result.

    When debug_error is set, uses plan_debug_requirements to get expanded requirements.
    """
    if graph_chunks is None and job_dir:
        chunks_path = os.path.join(job_dir, "graph_chunks.json")
        if os.path.isfile(chunks_path):
            try:
                with open(chunks_path, "r", encoding="utf-8") as f:
                    graph_chunks = json.load(f)
            except Exception:
                graph_chunks = []
        else:
            graph_chunks = []
    graph_chunks = graph_chunks or []
    graph_nodes = _load_graph_nodes(job_dir) if job_dir else []

    graph_summary = _build_graph_summary(graph_chunks, graph_nodes)

    # 1. Get requirements: LLM (or debug plan) or minimal fallback
    if debug_error and previous_requirements and llm_adapter:
        requirements = await plan_debug_requirements(
            step_id, goal, debug_error, previous_requirements, graph_summary, llm_adapter
        )
    elif llm_adapter:
        requirements = await extract_requirements_via_llm(
            step_id, goal, user_demand, graph_summary, llm_adapter
        )
    else:
        requirements = _minimal_fallback_requirements(step_id, goal, user_demand)

    result = query_graph_for_requirements(graph_chunks, requirements)
    result["requirements"] = requirements

    # 2. Query full graph nodes
    node_selectors = _nodes_to_selectors(graph_nodes, requirements, max_n=15)
    existing = set(result.get("selectors", []))
    for s in node_selectors:
        if s and s not in existing:
            result["selectors"] = result.get("selectors", []) + [s]
            existing.add(s)

    # 3. Empty fallback: top chunks by visual_importance + any nodes with href/src
    if not result.get("selectors"):
        result["used_empty_fallback"] = True
        seen_fb = set()
        for c in sorted(graph_chunks, key=lambda x: -x.get("visual_importance", 0))[:10]:
            sel = c.get("selector")
            if not sel and c.get("attrs"):
                sel = _build_selector({"tag": c.get("tag", "div"), "attrs": c.get("attrs", {})})
            if sel and sel not in seen_fb:
                result["selectors"] = result.get("selectors", []) + [sel]
                seen_fb.add(sel)
        for n in graph_nodes:
            if len(result.get("selectors", [])) >= 12:
                break
            if (n.get("tag") == "a" and n.get("href")) or (n.get("tag") == "img" and (n.get("src") or n.get("data_src"))):
                sel = n.get("selector") or _build_selector(n)
                if sel and sel not in seen_fb:
                    result["selectors"] = result.get("selectors", []) + [sel]
                    seen_fb.add(sel)
        if "img" not in seen_fb and any("img" in str(t).lower() for t in (requirements.get("element_types") or [])):
            result["selectors"] = result.get("selectors", []) + ["img"]

    return result
