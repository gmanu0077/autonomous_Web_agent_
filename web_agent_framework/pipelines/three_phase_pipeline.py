"""
3-Phase Autonomous Web Agent Pipeline.

Phase 1: Autonomous Page Preparation (fetch, blocker loop, re-fetch, expansion + redirect guard, DOM graph)
Phase 2: Signal readiness, listen for user demand (handled by chat_cli)
Phase 3: Targeted Operation Execution (demand-aware plan, execute, verify, debug loop, result delivery)
"""

import asyncio
import json
import os
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional
from urllib.parse import urlsplit, urlunsplit

from langgraph.graph import StateGraph, END
import networkx as nx

from ..config import TEMP_ROOT, MAX_HTML_CONTEXT_CHARS
from ..core.utils import ensure_dir, write_text
from ..core.html_context import chunk_and_pickle_html, build_html_context_from_pickles, build_dom_js_hints
from ..core.browser_js_snippets import js_get_smart_dom_tree, js_click_cookieish, js_expand_sections_universal, js_universal_scroll
from ..core.dom_graph import DOMGraphBuilder, perform_smart_chunking, plot_dom_graph, get_chunks_for_demand
from ..core.graph_query import query_graph_for_step
from ..adapters.pydoll_adapter import PydollBrowserAdapter
from ..adapters.mcp import get_browser_adapter
from ..adapters.llm_factory import get_llm_adapter
from ..adapters.node_rag_factory import get_node_rag_adapter
from ..agent.demand_planner import DemandPlanner
from ..agent.step_generator import StepGenerator
from ..agent.runner import Runner

def _get_rag():
    """Return RAG adapter for Pydoll code context, or None if unavailable."""
    try:
        from ..adapters.rag import ChromaOllamaRAG
        return ChromaOllamaRAG()
    except Exception:
        return None


def _load_from_job_dir(job_dir: str) -> tuple:
    """Load graph_chunks, html_context, dom_js_hints from job_dir. Returns (graph_chunks, html_context, dom_js_hints)."""
    graph_chunks: List[Dict[str, Any]] = []
    html_context = ""
    dom_js_hints = ""

    if not job_dir or not os.path.isdir(job_dir):
        return graph_chunks, html_context, dom_js_hints

    chunks_path = os.path.join(job_dir, "graph_chunks.json")
    if os.path.isfile(chunks_path):
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                graph_chunks = json.load(f)
        except Exception:
            pass

    ctx_path = os.path.join(job_dir, "html_context.txt")
    if os.path.isfile(ctx_path):
        try:
            with open(ctx_path, "r", encoding="utf-8", errors="ignore") as f:
                html_context = f.read()
        except Exception:
            pass

    hints_path = os.path.join(job_dir, "dom_js_hints.txt")
    if os.path.isfile(hints_path):
        try:
            with open(hints_path, "r", encoding="utf-8", errors="ignore") as f:
                dom_js_hints = f.read()
        except Exception:
            pass

    if not html_context:
        html_dir = os.path.join(job_dir, "html")
        if os.path.isdir(html_dir):
            html_context = build_html_context_from_pickles(html_dir, MAX_HTML_CONTEXT_CHARS)
        if not dom_js_hints and html_context:
            dom_js_hints = build_dom_js_hints(html_context)

    return graph_chunks, html_context, dom_js_hints


def _normalize_url_for_compare(url: str) -> str:
    """Normalize URL for redirect comparison (strip fragment, lowercase scheme)."""
    if not url:
        return ""
    try:
        parsed = urlsplit(url)
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, ""))
    except Exception:
        return url


def _extract_cdp_value(raw):
    """Extract actual value from CDP response (Pydoll may return nested result/value)."""
    if raw is None or not isinstance(raw, dict):
        return raw
    if "result" in raw:
        raw = raw["result"]
    if isinstance(raw, dict) and "result" in raw:
        raw = raw["result"]
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"]
    return raw


async def _execute_script_json_async(page, script: str):
    """Execute JS and parse JSON result (async). Handles CDP-wrapped responses."""
    result = await page.execute_script(script)
    result = _extract_cdp_value(result) if isinstance(result, dict) else result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    return result


# --- State ---

class ThreePhaseState(TypedDict):
    url: str
    user_demand: Optional[str]
    status: str
    html: Optional[str]
    html_dir: str
    html_context: str
    dom_js_hints: str
    graph: Optional[Any]  # nx.DiGraph
    graph_chunks: List[Dict[str, Any]]
    job_dir: str
    plan: Dict[str, Any]
    shared: Dict[str, Any]
    result: str
    error: Optional[str]
    prepared_state: Optional[Dict[str, Any]]  # Pass-through when Phase 3 only
    # Phase 3 routing from CLI: "action" = Pydoll plan/execute only; "chat" = Node RAG only; None = LLM intent
    phase3_mode: Optional[str]


# --- Phase 1 Nodes ---

def _get_browser():
    """Use MCP if MCP_SERVER_URL is set, else Pydoll."""
    return get_browser_adapter()


async def fetch_initial_html_node(state: ThreePhaseState) -> ThreePhaseState:
    """Initial HTML fetch via MCP or Pydoll."""
    print("[Phase 1] Fetching initial HTML...")
    browser = _get_browser()
    page = await browser.start()
    await page.navigate(state["url"])
    await asyncio.sleep(5)
    html = await page.get_html()
    await browser.close()
    return {**state, "html": html, "status": "fetched"}


async def analyze_blockers_node(state: ThreePhaseState) -> ThreePhaseState:
    """LLM analyzes HTML for blockers (cookie banners, consent overlays)."""
    print("[Phase 1] LLM analyzing blockers...")
    html_snippet = (state.get("html") or "")[:8000]
    llm = get_llm_adapter()
    prompt = f"""
Analyze this HTML snippet for page blockers:
- Cookie banners, consent overlays, newsletter modals
- Elements that might block interaction (position:fixed, high z-index, covering viewport)

HTML (truncated):
{html_snippet}

Return JSON:
{{"has_blockers": true/false, "blocker_descriptions": ["..."], "suggested_selectors": ["button.accept", "[aria-label='Accept']"]}}
"""
    try:
        resp = await llm.generate(prompt, system_prompt="You output only valid JSON.")
        clean = resp.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        data = json.loads(clean.strip())
        state["shared"] = state.get("shared") or {}
        state["shared"]["blocker_analysis"] = data
        return {**state}
    except Exception as e:
        print(f"[Phase 1] Blocker analysis failed: {e}, assuming no blockers")
        state["shared"] = state.get("shared") or {}
        state["shared"]["blocker_analysis"] = {"has_blockers": False, "blocker_descriptions": [], "suggested_selectors": []}
        return {**state}


async def execute_blocker_scripts_node(state: ThreePhaseState) -> ThreePhaseState:
    """Execute blocker dismissal scripts via browser. Uses js_click_cookieish as fallback."""
    print("[Phase 1] Executing blocker scripts...")
    browser = _get_browser()
    page = await browser.start()
    await page.navigate(state["url"])
    await asyncio.sleep(3)

    # Run cookieish click (proven fallback)
    try:
        await page.execute_script(js_click_cookieish())
        await asyncio.sleep(1)
    except Exception as e:
        print(f"[Phase 1] Blocker click warning: {e}")

    html = await page.get_html()
    await browser.close()
    return {**state, "html": html, "status": "blockers_executed"}


async def verify_blockers_gone_node(state: ThreePhaseState) -> ThreePhaseState:
    """LLM verifies if blockers are gone from HTML."""
    print("[Phase 1] Verifying blockers gone...")
    html_snippet = (state.get("html") or "")[:6000]
    llm = get_llm_adapter()
    prompt = f"""
Does this HTML still show obvious cookie banners, consent overlays, or full-screen modals that block the main content?
Look for: cookie, consent, accept, agree, newsletter, modal, overlay with high z-index.

HTML (truncated):
{html_snippet}

Return JSON: {{"blockers_still_present": true/false}}
"""
    try:
        resp = await llm.generate(prompt, system_prompt="You output only valid JSON.")
        clean = resp.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        data = json.loads(clean.strip())
        blockers_still = data.get("blockers_still_present", False)
        state["shared"] = state.get("shared") or {}
        state["shared"]["blockers_verified_gone"] = not blockers_still
        return {**state}
    except Exception:
        state["shared"] = state.get("shared") or {}
        state["shared"]["blockers_verified_gone"] = True  # Assume ok on parse error
        return {**state}


async def clean_html_refetch_node(state: ThreePhaseState) -> ThreePhaseState:
    """Clean HTML re-fetch after blocker removal."""
    print("[Phase 1] Clean HTML re-fetch...")
    browser = _get_browser()
    page = await browser.start()
    await page.navigate(state["url"])
    await asyncio.sleep(2)
    html = await page.get_html()
    await browser.close()
    return {**state, "html": html, "status": "clean_html"}


async def content_expansion_redirect_guard_node(state: ThreePhaseState) -> ThreePhaseState:
    """Universal scroll + expansion (like graph_scraper) to load all DOM nodes and lazy content."""
    print("[Phase 1] Content expansion with redirect guard...")
    browser = _get_browser()
    page = await browser.start()
    await page.navigate(state["url"])
    await asyncio.sleep(2)

    original_url = state["url"]
    original_norm = _normalize_url_for_compare(original_url)

    for _ in range(3):
        try:
            # Initial scroll to reveal content (graph_scraper pattern)
            print("[Phase 1] Scrolling to reveal content...")
            await page.execute_script("window.scrollTo(0, 500);")
            await asyncio.sleep(1)

            # Universal step-scroll to load lazy content (comments, related, infinite scroll)
            print("[Phase 1] Step-scrolling to load lazy content...")
            await page.execute_script(js_universal_scroll())
            await asyncio.sleep(1)

            # Universal section expansion (accordions, show more, comments, related)
            print("[Phase 1] Expanding hidden sections...")
            await page.execute_script(js_expand_sections_universal())
            await asyncio.sleep(2)

            # Redirect guard
            current = await page.execute_script("return location.href;")
            current_str = str(current) if current else ""
            current_norm = _normalize_url_for_compare(current_str)
            if current_norm != original_norm:
                print(f"[Phase 1] Redirect guard: URL changed, navigating back")
                await page.navigate(original_url)
                await asyncio.sleep(1)
        except Exception as e:
            print(f"[Phase 1] Expansion warning: {e}")
            break

    html = await page.get_html()
    await browser.close()
    return {**state, "html": html, "status": "expanded"}


async def build_dom_graph_node(state: ThreePhaseState) -> ThreePhaseState:
    """Build full page DOM graph (NetworkX) using smart DOM tree. Uses expanded HTML from state."""
    print("[Phase 1] Building DOM graph...")
    # DOM graph and chunks use state["html"] from content_expansion (already scrolled+expanded)
    html = state.get("html") or ""

    browser = _get_browser()
    page = await browser.start()
    await page.navigate(state["url"])
    await asyncio.sleep(2)

    # Scroll + expand before capturing DOM (same as graph_scraper — get all nodes)
    try:
        await page.execute_script("window.scrollTo(0, 500);")
        await asyncio.sleep(1)
        await page.execute_script(js_universal_scroll())
        await asyncio.sleep(1)
        await page.execute_script(js_expand_sections_universal())
        await asyncio.sleep(2)
    except Exception as e:
        print(f"[Phase 1] Pre-capture scroll/expand warning: {e}")

    dom_nodes_raw = await _execute_script_json_async(page, js_get_smart_dom_tree())
    await browser.close()

    if isinstance(dom_nodes_raw, list) and dom_nodes_raw:
        G = nx.DiGraph()
        for node in dom_nodes_raw:
            node_id = node.get("id")
            parent_id = node.get("parentId")
            props = {k: v for k, v in node.items() if k not in ("id", "parentId")}
            if node_id:
                G.add_node(node_id, **props)
                if parent_id:
                    G.add_edge(parent_id, node_id)
        state["graph"] = G
        state["graph_chunks"] = perform_smart_chunking(G, state["url"])
    else:
        # Fallback: build from HTML string (from content_expansion)
        builder = DOMGraphBuilder(max_depth=40)
        G = builder.build(html)
        state["graph"] = G
        state["graph_chunks"] = perform_smart_chunking(G, state["url"])

    # Persist HTML and build context
    job_dir = state.get("job_dir") or ensure_dir(os.path.join(TEMP_ROOT, f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    html_dir = ensure_dir(os.path.join(job_dir, "html"))
    ensure_dir(html_dir)
    html = state.get("html") or ""
    chunk_and_pickle_html(html, html_dir)
    html_context = build_html_context_from_pickles(html_dir, MAX_HTML_CONTEXT_CHARS)
    dom_js_hints = build_dom_js_hints(html_context)

    # Persist to disk for Phase 3 (avoid in-memory bloat)
    graph_chunks = state.get("graph_chunks") or []
    write_text(os.path.join(job_dir, "graph_chunks.json"), json.dumps(graph_chunks, indent=2, default=str))
    write_text(os.path.join(job_dir, "html_context.txt"), html_context[:MAX_HTML_CONTEXT_CHARS])
    write_text(os.path.join(job_dir, "dom_js_hints.txt"), dom_js_hints)
    try:
        graph_data = nx.node_link_data(G, edges="links")
        write_text(os.path.join(job_dir, "graph.json"), json.dumps(graph_data, indent=2, default=str))
    except Exception:
        pass

    # Save graph visualization (graph.png)
    try:
        plot_dom_graph(G, output_path=os.path.join(job_dir, "graph.png"), max_nodes=200)
    except Exception as e:
        print(f"[Phase 1] Could not save graph.png: {e}")

    return {
        **state,
        "html_dir": html_dir,
        "html_context": html_context,
        "dom_js_hints": dom_js_hints,
        "job_dir": job_dir,
        "status": "ready_for_demand",
    }


async def index_nodes_rag_node(state: ThreePhaseState) -> ThreePhaseState:
    """Index graph nodes into NodeRAG for user queries."""
    print("[Phase 1] Indexing nodes into NodeRAG...")
    job_dir = state.get("job_dir")
    if not job_dir:
        return state
        
    rag_dir = os.path.join(job_dir, "node_rag")
    rag = get_node_rag_adapter(rag_dir)
    
    # Use all nodes from graph
    G = state.get("graph")
    if G and G.nodes:
        nodes = []
        for n_id, data in G.nodes(data=True):
            node = dict(data)
            node["id"] = n_id
            nodes.append(node)
            
        llm = get_llm_adapter()
        await rag.index_nodes(nodes, llm)
        
    return state


# --- Phase 1 conditional routing ---

MAX_BLOCKER_RETRIES = 3


def route_after_verify(state: ThreePhaseState) -> str:
    """After verify: retry execute or proceed to clean refetch."""
    shared = state.get("shared") or {}
    verified = shared.get("blockers_verified_gone", True)
    retries = shared.get("blocker_retry_count", 0)
    if verified or retries >= MAX_BLOCKER_RETRIES:
        return "clean_refetch"
    shared["blocker_retry_count"] = retries + 1
    return "execute_blockers"


# --- Phase 3 Nodes ---

async def query_node_rag_node(state: ThreePhaseState) -> ThreePhaseState:
    """Query NodeRAG with user demand and return answer."""
    print("[Phase 3] Querying NodeRAG (chat mode)...")
    user_demand = state.get("user_demand") or ""
    prep = state.get("prepared_state") or state
    job_dir = prep.get("job_dir") or state.get("job_dir") or ""

    if not job_dir or not user_demand:
        return state

    rag_dir = os.path.join(job_dir, "node_rag")
    nodes_index = os.path.join(rag_dir, "nodes_by_id.json")
    if not os.path.isfile(nodes_index):
        return {
            **state,
            "result": (
                "No node index found for this job. Re-load the URL to re-run page analysis and indexing."
            ),
            "status": "completed",
        }

    rag = get_node_rag_adapter(rag_dir)
    llm = get_llm_adapter()

    # Query top-k nodes
    top_nodes = await rag.query(user_demand, llm, k=10)

    if not top_nodes:
        return {**state, "result": "I couldn't find any relevant nodes for that question."}
        
    # Build prompt for LLM with full node data
    nodes_str = json.dumps(top_nodes, indent=2)
    prompt = f"""You are an AI assistant that answers questions about a web page's structure and content.
You have been provided with the top 10 most relevant DOM nodes from the page based on the user's query.

USER QUERY: {user_demand}

RELEVANT NODES (JSON):
{nodes_str}

Based ONLY on the node data above, answer the user's query. If the data doesn't contain the answer, say so.
Be specific and mention relevant tags, attributes, or text content when applicable.
"""
    try:
        answer = await llm.generate(prompt)
        return {**state, "result": answer, "status": "completed"}
    except Exception as e:
        print(f"[NodeRAG] Query failed: {e}")
        return {**state, "result": f"Error querying NodeRAG: {e}"}


async def plan_from_demand_node(state: ThreePhaseState) -> ThreePhaseState:
    """Demand-aware planning: LLM uses DOM graph + HTML to plan steps for user demand."""
    print("[Phase 3] Planning steps from user demand...")
    user_demand = state.get("user_demand") or ""
    if not user_demand:
        return {**state, "error": "No user demand provided"}

    prep = state.get("prepared_state") or state
    job_dir = prep.get("job_dir") or state.get("job_dir") or ""
    graph_chunks, html_context, dom_js_hints = _load_from_job_dir(job_dir)
    html_context = (html_context or prep.get("html_context") or "")[:12000]
    dom_js_hints = (dom_js_hints or prep.get("dom_js_hints") or "")[:2000]

    relevant_chunks = get_chunks_for_demand(graph_chunks, user_demand) or graph_chunks[:20]
    chunks_text = ""
    for i, c in enumerate(relevant_chunks[:15]):
        text = (c.get("text") or "")[:500]
        sel = c.get("selector")
        hint = c.get("semantic_hint")
        extra = []
        if sel:
            extra.append(f"selector={sel}")
        if hint:
            extra.append(f"hint={hint}")
        extra_str = " " + ", ".join(extra) if extra else ""
        if text or extra_str:
            chunks_text += f"\nChunk {i}{extra_str}: {text}\n"

    llm = get_llm_adapter()
    planner = DemandPlanner(llm)
    plan = await planner.plan_for_demand(user_demand, html_context, dom_js_hints, chunks_text)

    return {**state, "plan": plan, "status": "planned"}


async def execute_step_with_verify_node(state: ThreePhaseState) -> ThreePhaseState:
    """Execute one step, verify outcome (LLM), debug loop if needed."""
    print("[Phase 3] Executing steps with verify/debug loop...")
    plan = state.get("plan") or {}
    steps = plan.get("steps", [])
    if not steps:
        return {**state, "result": "No steps to execute.", "status": "completed"}

    prep = state.get("prepared_state") or state
    url = prep.get("url") or state.get("url")
    job_dir = prep.get("job_dir") or state.get("job_dir") or ensure_dir(os.path.join(TEMP_ROOT, "phase3"))
    graph_chunks, html_context, _ = _load_from_job_dir(job_dir)
    if not html_context:
        html_context = prep.get("html_context") or state.get("html_context") or ""
    shared = prep.get("shared") or state.get("shared") or {}
    user_demand = state.get("user_demand") or ""
    relevant_chunks = get_chunks_for_demand(graph_chunks, user_demand) if user_demand else (graph_chunks[:25] if graph_chunks else [])

    llm = get_llm_adapter()
    rag = _get_rag()
    step_gen = StepGenerator(llm, rag_adapter=rag)
    runner = Runner(job_dir)
    max_debug_retries = 2

    browser = _get_browser()
    page = await browser.start()
    await page.navigate(url)
    await asyncio.sleep(4)

    # Re-apply scroll/expand so lazy content (comments, related) loads before extraction steps
    try:
        await page.execute_script("window.scrollTo(0, 500);")
        await asyncio.sleep(1)
        await page.execute_script(js_universal_scroll())
        await asyncio.sleep(2)
        await page.execute_script(js_expand_sections_universal())
        await asyncio.sleep(2)
    except Exception as e:
        print(f"[Phase 3] Pre-step scroll/expand warning: {e}")

    result_text = ""
    for step in steps:
        sid = step.get("id", "step")
        goal = step.get("goal", "")
        for debug_round in range(max_debug_retries + 1):
            try:
                # LLM decides what's needed → graph query → code
                graph_query_result = await query_graph_for_step(
                    job_dir, sid, goal, user_demand, graph_chunks=graph_chunks, llm_adapter=llm
                )
                code = await step_gen.generate_step(
                    sid, goal, html_context[:4000], shared,
                    graph_query_result=graph_query_result,
                    dom_chunks=None,  # Prefer graph query over raw chunks
                    user_demand=user_demand,
                )
                runner.save_step_module(sid, code)
                run_result = await runner.run_in_process(page, [sid], shared)
                run_ok = run_result.get("ok", False)
                if run_ok:
                    # Verify with LLM
                    current_html = await page.get_html()
                    verify_prompt = f"""
Step goal: {goal}
Step ID: {sid}
Current page HTML (truncated): {str(current_html)[:3000]}

Did this step achieve its goal? Return JSON: {{"achieved": true/false, "evidence": "..."}}
"""
                    verify_resp = await llm.generate(verify_prompt, system_prompt="You output only valid JSON.")
                    try:
                        v_clean = verify_resp.strip()
                        if "```" in v_clean:
                            v_clean = v_clean.split("```")[1]
                            if v_clean.startswith("json"):
                                v_clean = v_clean[4:]
                        v_data = json.loads(v_clean.strip())
                        if v_data.get("achieved", True):
                            break
                    except Exception:
                        break
                # Not achieved or run failed — debug: LLM plans what else needed → re-query graph → new code
                if debug_round < max_debug_retries:
                    errs = run_result.get("errors") or {}
                    error_hint = errs.get(sid, "Step did not achieve goal") if isinstance(errs, dict) else "Step did not achieve goal"
                    prev_req = graph_query_result.get("requirements") if graph_query_result else None
                    graph_query_result = await query_graph_for_step(
                        job_dir, sid, goal, user_demand, graph_chunks=graph_chunks, llm_adapter=llm,
                        previous_requirements=prev_req, debug_error=error_hint
                    )
                    code = await step_gen.generate_step(
                        sid, goal, html_context[:4000], shared,
                        repair_mode=True, previous_error=error_hint,
                        graph_query_result=graph_query_result,
                        user_demand=user_demand,
                    )
                    runner.save_step_module(sid, code)
                else:
                    result_text = f"Step '{sid}' failed after retries."
                    break
            except Exception as e:
                result_text = str(e)
                if debug_round >= max_debug_retries:
                    break

    await browser.close()

    # Save step outputs to temp job dir (alongside step code)
    step_results = shared.get("step_results", {})
    outputs_path = os.path.join(job_dir, "steps", "step_outputs.json")
    try:
        write_text(outputs_path, json.dumps(step_results, indent=2))
    except Exception:
        pass

    # Build result: always include partial results; on failure, show both failure msg and what we got
    if step_results:
        partial_json = json.dumps(step_results, indent=2)
        if result_text:
            result_text = f"{result_text}\n\n--- Partial results from completed steps ---\n{partial_json}"
        else:
            result_text = partial_json
    if not result_text:
        result_text = "Execution completed."

    return {**state, "result": result_text, "status": "completed"}


# --- Graph construction ---

def create_phase1_graph():
    """Phase 1: Autonomous Page Preparation."""
    workflow = StateGraph(ThreePhaseState)
    workflow.add_node("fetch", fetch_initial_html_node)
    workflow.add_node("analyze_blockers", analyze_blockers_node)
    workflow.add_node("execute_blockers", execute_blocker_scripts_node)
    workflow.add_node("verify_blockers", verify_blockers_gone_node)
    workflow.add_node("clean_refetch", clean_html_refetch_node)
    workflow.add_node("expansion", content_expansion_redirect_guard_node)
    workflow.add_node("build_graph", build_dom_graph_node)
    workflow.add_node("index_nodes", index_nodes_rag_node)

    workflow.set_entry_point("fetch")
    workflow.add_edge("fetch", "analyze_blockers")
    workflow.add_edge("analyze_blockers", "execute_blockers")
    workflow.add_edge("execute_blockers", "verify_blockers")
    workflow.add_conditional_edges("verify_blockers", route_after_verify)
    workflow.add_edge("clean_refetch", "expansion")
    workflow.add_edge("expansion", "build_graph")
    workflow.add_edge("build_graph", "index_nodes")
    workflow.add_edge("index_nodes", END)

    return workflow.compile()


async def detect_intent(state: ThreePhaseState) -> str:
    """Route Phase 3: explicit CLI mode overrides LLM intent detection."""
    mode = (state.get("phase3_mode") or "").strip().lower()
    if mode == "chat":
        return "query_rag"
    if mode == "action":
        return "plan"

    user_demand = state.get("user_demand") or ""
    if not user_demand:
        return "plan"

    llm = get_llm_adapter()
    prompt = f"""Analyze the user's intent from their demand for a web agent.
Is this a question seeking information *about* the current page content/structure (e.g. "what is the price?", "how many items are there?", "what are the node IDs for X?")
OR is it a request for *action* (e.g. "click the buy button", "scroll to the bottom", "search for shoes").

USER DEMAND: {user_demand}

Return ONLY valid JSON (no markdown):
{{"intent": "query" or "action"}}
"""
    try:
        data = await llm.generate_json(prompt, system_prompt="You output only valid JSON.")
        intent = data.get("intent", "action")
        return "query_rag" if intent == "query" else "plan"
    except Exception:
        # Default to action if intent detection fails
        return "plan"


async def detect_intent_node(state: ThreePhaseState) -> ThreePhaseState:
    """Node to update state with intent (for conditional edge)."""
    # This node doesn't strictly need to do anything if we use a pure function for conditional edge,
    # but some versions of langgraph prefer nodes.
    return state


def create_phase3_graph():
    """Phase 3: Targeted Operation Execution (with intent detection and NodeRAG)."""
    workflow = StateGraph(ThreePhaseState)
    workflow.add_node("detect_intent", detect_intent_node)
    workflow.add_node("query_rag", query_node_rag_node)
    workflow.add_node("plan", plan_from_demand_node)
    workflow.add_node("execute", execute_step_with_verify_node)

    workflow.set_entry_point("detect_intent")
    
    # Conditional edge from detect_intent to either query_rag or plan
    workflow.add_conditional_edges(
        "detect_intent",
        detect_intent, # Pure function
        {
            "query_rag": "query_rag",
            "plan": "plan"
        }
    )
    
    workflow.add_edge("query_rag", END)
    workflow.add_edge("plan", "execute")
    workflow.add_edge("execute", END)
    return workflow.compile()


# --- Main entry ---

async def run_three_phase_agent(
    url: str,
    user_demand: Optional[str] = None,
    prepared_state: Optional[Dict[str, Any]] = None,
    phase3_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the 3-phase agent.

    - If user_demand is None: run Phase 1 only, return prepared state.
    - If user_demand and prepared_state: run Phase 3 only, return result.

    phase3_mode: "action" (Pydoll plan/execute), "chat" (Node RAG only), or None (LLM intent routing).
    """
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Phase 3 must reuse Phase 1 job_dir (graph, node_rag, steps).
    if user_demand and prepared_state and prepared_state.get("job_dir"):
        job_dir = str(prepared_state["job_dir"])
    else:
        job_dir = ensure_dir(os.path.join(TEMP_ROOT, f"job_{job_id}"))

    initial_state: ThreePhaseState = {
        "url": url,
        "user_demand": user_demand,
        "status": "init",
        "html": None,
        "html_dir": "",
        "html_context": "",
        "dom_js_hints": "",
        "graph": None,
        "graph_chunks": [],
        "job_dir": job_dir,
        "plan": {},
        "shared": {},
        "result": "",
        "error": None,
        "prepared_state": prepared_state,
        "phase3_mode": phase3_mode,
    }

    if user_demand and prepared_state:
        # Phase 3 only
        phase3 = create_phase3_graph()
        final = await phase3.ainvoke(initial_state)
        return dict(final)
    else:
        # Phase 1 only
        phase1 = create_phase1_graph()
        final = await phase1.ainvoke(initial_state)
        return dict(final)
