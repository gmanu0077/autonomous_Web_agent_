# Web Agent Framework (WAF)

> A 3-phase autonomous web agent that prepares any web page, then either **executes actions** on it (Pydoll-driven automation) or **answers questions** about it (semantic search over an indexed DOM graph).

The framework is built around a **LangGraph** state machine, **Pydoll** (or Selenium) for real Chrome control via **CDP**, an **LLM** (Gemini or Ollama) for reasoning/planning/code generation, and **ChromaDB** (or FAISS) for two distinct RAG layers — code RAG (Pydoll/Selenium snippets) and per-job DOM-node RAG (live page nodes).

---

## Table of Contents

1. [Why this framework exists](#why-this-framework-exists)
2. [High-level architecture](#high-level-architecture)
3. [Repository layout](#repository-layout)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Building the code RAG (one-time)](#building-the-code-rag-one-time)
8. [Running the agent (entry point)](#running-the-agent-entry-point)
9. [How a request flows through the code](#how-a-request-flows-through-the-code)
   - [Phase 1 — Page preparation](#phase-1--page-preparation)
   - [Phase 2 — Mode selection](#phase-2--mode-selection)
   - [Phase 3a — Action mode](#phase-3a--action-mode-pydoll-execution)
   - [Phase 3b — Chat mode](#phase-3b--chat-mode-node-rag)
10. [Per-job output (what gets written to disk)](#per-job-output-what-gets-written-to-disk)
11. [Module reference](#module-reference)
12. [MCP servers (optional)](#mcp-servers-optional)
13. [Step generation contract](#step-generation-contract)
14. [Extending the framework](#extending-the-framework)
15. [Troubleshooting](#troubleshooting)
16. [License](#license)

---

## Why this framework exists

Most "browser agent" projects either:

- generate a giant prompt with the raw HTML and hope the LLM produces a working snippet, or
- hard-code site-specific selectors/CSS heuristics.

This project takes a different approach:

1. **Prepare** the page once: dismiss blockers, fully expand and scroll, then build a graph of every DOM node (`networkx.DiGraph`) and a smart-chunk view of the page.
2. **Index** every node into a per-job vector store, so the page can be queried later in plain language.
3. **For actions**: ask the LLM what *requirements* (tags, attrs, semantic hints, child-selector hints, schema) the step needs; **query the DOM graph** for matching selectors; pass *those concrete selectors* into a sandboxed Python step that runs JavaScript through Pydoll.
4. **Verify** every step with the LLM (look at fresh HTML, decide if the goal was achieved); if not, **debug-loop**: re-plan requirements, re-query the graph, regenerate code.

The result is generic — no site-specific code, no hard-coded selectors — but always grounded in real, observed page structure.

---

## High-level architecture

```
                  ┌────────────────────────────────────────────┐
  user (CLI) ──►  │              run_agent.py                  │
                  │              chat_cli.py                   │
                  └───────────────┬────────────────────────────┘
                                  │
                       ┌──────────▼──────────┐
                       │ three_phase_pipeline│  (LangGraph StateGraph)
                       └──┬──────────────────┘
                          │
   ┌──────────────────────┴──────────────────────────────────┐
   │                                                         │
   │  PHASE 1  (page preparation)                            │
   │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
   │  │ fetch   │─►│ analyze  │─►│ execute  │─►│ verify   │  │
   │  │ HTML    │  │ blockers │  │ blockers │  │ blockers │  │
   │  └─────────┘  └──────────┘  └──────────┘  └────┬─────┘  │
   │                                          (loop)│        │
   │  ┌──────────┐  ┌─────────────────┐  ┌──────────▼─────┐  │
   │  │ index    │◄─│ build DOM graph │◄─│ scroll + expand│  │
   │  │ nodes    │  │  + chunks       │  │ + redirect grd │  │
   │  └────┬─────┘  └─────────────────┘  └────────────────┘  │
   │       │                                                 │
   └───────┼─────────────────────────────────────────────────┘
           │
           │ status = "ready_for_demand"  →  user picks mode  │
           │
   ┌───────┼─────────────────────────────────────────────────┐
   │       │                                                 │
   │  PHASE 3 (intent routing)                               │
   │       │                                                 │
   │  ┌────▼─────────┐                                       │
   │  │ detect_intent│                                       │
   │  └──┬──────┬────┘                                       │
   │     │      │                                            │
   │     │ chat │ action                                     │
   │     ▼      ▼                                            │
   │  ┌──────┐ ┌──────────┐  ┌──────────────────┐            │
   │  │ Node │ │ Demand-  │  │ Step exec loop:  │            │
   │  │ RAG  │ │ Planner  │─►│  1. graph_query  │            │
   │  │ Q→A  │ │ (steps)  │  │  2. step gen     │            │
   │  └──────┘ └──────────┘  │  3. run module   │            │
   │                         │  4. verify (LLM) │            │
   │                         │  5. debug loop   │            │
   │                         └──────────────────┘            │
   └─────────────────────────────────────────────────────────┘
```

---

## Repository layout

```
waf/
├── run_agent.py                       # entry point shim
├── LICENSE                            # MIT
├── .gitignore
└── web_agent_framework/
    ├── chat_cli.py                    # interactive REPL (URL → Phase 1 → mode → Phase 3)
    ├── config.py                      # all env-tunable knobs
    ├── build_embeddings.py            # builds ChromaDB code-RAG (run once)
    │
    ├── pipelines/
    │   ├── three_phase_pipeline.py    # MAIN pipeline (LangGraph)
    │   ├── autonomous_agent_pipeline.py  # legacy 3-step bootstrap/plan/execute pipeline
    │   └── ecommerce_scraper_graph.py # legacy e-commerce-specific pipeline
    │
    ├── agent/
    │   ├── demand_planner.py          # Phase 3 LLM step planner ("plan_for_demand")
    │   ├── planner.py                 # legacy fixed-ID planner (autonomous pipeline)
    │   ├── step_generator.py          # turns step goals into Pydoll Python modules
    │   ├── step_system_prompt.md      # system prompt that constrains generated code
    │   ├── runner.py                  # imports and runs generated step modules
    │   └── output_processor.py        # post-processing helpers (image URL cleanup)
    │
    ├── core/
    │   ├── dom_graph.py               # NetworkX DOM graph + smart chunking
    │   ├── graph_query.py             # requirement-driven graph→selector mapping
    │   ├── browser_js_snippets.py     # JS snippets injected into the live page
    │   ├── html_context.py            # HTML pickling, windowing, DOM/JS hint extraction
    │   ├── image_promotion.py         # high-res image URL heuristics
    │   ├── url_utils.py               # URL normalisation, family keys, scoring
    │   └── utils.py                   # ensure_dir, write_text, intent inference
    │
    ├── adapters/
    │   ├── browser.py                 # BrowserAdapter / PageAdapter Protocols
    │   ├── pydoll_adapter.py          # Pydoll → PageAdapter implementation
    │   ├── mcp.py                     # MCP-over-stdio browser adapter
    │   ├── llm.py                     # LLMAdapter Protocol
    │   ├── llm_factory.py             # picks Gemini or Ollama from config
    │   ├── gemini_adapter.py          # Google Gemini implementation
    │   ├── ollama_adapter.py          # local Ollama implementation
    │   ├── rag.py                     # ChromaOllamaRAG (code RAG)
    │   ├── node_rag.py                # ChromaNodeRAGAdapter / FaissNodeRAGAdapter
    │   └── node_rag_factory.py        # picks chroma or faiss
    │
    └── mcp_servers/
        ├── mcp_server_pydoll.py       # MCP stdio server wrapping Pydoll
        └── mcp_server_selenium.py     # MCP stdio server wrapping Selenium
```

---

## Requirements

### System

- **Python 3.10+** (uses PEP 604 union types and `TypedDict` features available there).
- **Google Chrome** installed (Pydoll launches a real Chrome instance via CDP; Selenium server uses Chrome too).
- **OS:** Windows / macOS / Linux. Default `DEFAULT_DB_DIR` is a Windows path (`D:\autonomous web agent\chroma_db_pydoll`) — override `DB_DIR` env var on other OSes.

### Python packages

A `requirements.txt` is **not** committed yet. The package set the codebase imports is:

```text
# Core orchestration
langgraph
networkx
beautifulsoup4
lxml

# Vector DB
chromadb

# Browser automation
pydoll-python        # imported as `pydoll` (provides pydoll.browser.Chrome)
selenium             # only needed if MCP_BROWSER=selenium

# LLM providers (install at least one)
google-generativeai  # for LLM_PROVIDER=gemini
ollama               # for LLM_PROVIDER=ollama

# MCP (optional, only if you spawn or talk to an MCP server)
mcp
anyio

# Optional/feature flags
faiss-cpu            # only if NODE_RAG_BACKEND=faiss
numpy
matplotlib           # to render graph.png; silently skipped if missing
```

If you need a starter file, create `requirements.txt` with the lines above. The only mandatory ones for the default path (Gemini + Pydoll + Chroma) are `langgraph`, `networkx`, `beautifulsoup4`, `lxml`, `chromadb`, `pydoll-python`, `google-generativeai`, `numpy`.

### External services

- **Gemini API key** at https://aistudio.google.com/apikey — required if `LLM_PROVIDER=gemini` (the default).
- **Local Ollama** at http://localhost:11434 — required if `LLM_PROVIDER=ollama`. Pull at least one chat model and one embedding model:
  ```bash
  ollama pull qwen3-coder:30b      # or llama3.2 / mistral
  ollama pull nomic-embed-text     # embedding model
  ```

---

## Installation

```bash
git clone <this repo>
cd waf

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -U pip
pip install langgraph networkx beautifulsoup4 lxml chromadb \
            pydoll-python google-generativeai numpy
# optional extras
pip install ollama selenium mcp anyio faiss-cpu matplotlib
```

Create a `.env` file (or export real environment variables):

```ini
# Pick a provider
LLM_PROVIDER=gemini
LLM_MODEL=gemini-flash-latest
GEMINI_API_KEY=YOUR_KEY_HERE

# Or, all-local:
# LLM_PROVIDER=ollama
# LLM_MODEL=qwen3-coder:30b
# EMBEDDING_MODEL=nomic-embed-text

# Where the per-job working directories go
TEMP_ROOT=./temp_pydoll

# Where the *code* RAG is stored (see "Building the code RAG")
DB_DIR=./chroma_db_pydoll
COLLECTION_NAME=pydoll_code
```

The framework reads env vars in `web_agent_framework/config.py`. Anything you do not set falls back to a default in that file.

---

## Configuration

All knobs live in `web_agent_framework/config.py`. Every value is overridable via an environment variable of the same name.

| Variable                       | Default                                  | Purpose |
| ------------------------------ | ---------------------------------------- | ------- |
| `LLM_PROVIDER`                 | `gemini`                                 | `gemini` or `ollama`. Used by `llm_factory.get_llm_adapter()`. |
| `LLM_MODEL`                    | `gemini-flash-latest` / `qwen3-coder:30b` | Chat model name for the chosen provider. |
| `GEMINI_API_KEY`               | _(empty)_                                | Required when `LLM_PROVIDER=gemini`. Also accepted as `GOOGLE_API_KEY`. |
| `EMBEDDING_MODEL`              | `models/gemini-embedding-001` / `nomic-embed-text` | Embedding model used by both code RAG and Node RAG. |
| `DB_DIR`                       | `D:\autonomous web agent\chroma_db_pydoll` | Persist directory for the **code** Chroma collection. |
| `COLLECTION_NAME`              | `pydoll_code`                            | Code RAG collection name. |
| `MCP_BROWSER`                  | `pydoll`                                 | Which internal MCP server to spawn (`pydoll` or `selenium`). |
| `MCP_SERVER_URL`               | _(empty)_                                | If set (e.g. `stdio:npx -y @modelcontextprotocol/server-browser`) the agent connects to that external MCP server instead of the internal one. |
| `TEMP_ROOT`                    | `<workspace>/temp_pydoll`                | Root for all per-job working directories. |
| `MAX_CONTEXT_CHARS`            | `425`                                    | Char budget for code-RAG context per planner step. |
| `MAX_HTML_CONTEXT_CHARS`       | `108000`                                 | How much HTML text is materialised per job. |
| `TOP_K`                        | `5`                                      | Top-k chunks pulled from the code RAG. |
| `MAX_ATTEMPTS`                 | `3`                                      | Used by the legacy autonomous pipeline. |
| `NODE_RAG_BACKEND`             | `chroma`                                 | `chroma` or `faiss` for the per-job node store. |
| `NODE_RAG_CHROMA_COLLECTION`   | `dom_nodes`                              | Per-job node-RAG collection name. |
| `HTML_CHUNK_SIZE`              | `1024`                                   | Pickle chunk size for the saved HTML. |
| `HTML_CHUNK_OVERLAP`           | `128`                                    | Overlap for code-RAG chunking. |

> Note on `MAX_CONTEXT_CHARS=425`: it is intentionally tiny — the per-step context is a small slice of the code RAG. The much larger budget for prompts is `MAX_HTML_CONTEXT_CHARS`.

---

## Building the code RAG (one-time)

`adapters/rag.py` reads from a Chroma collection populated by `build_embeddings.py`. It is used only by the **step generator** (so generated Pydoll code follows working patterns) and by the legacy planner.

```bash
python -m web_agent_framework.build_embeddings \
    --source-dir ./web_agent_framework \
    --chunk-size 1024 \
    --overlap 128 \
    --db-dir ./chroma_db_pydoll \
    --collection pydoll_code
```

What it does:

1. Walks each `--source-dir` and reads every `.py`, `.md`, `.txt` file.
2. Splits them into overlapping chunks of `--chunk-size` chars.
3. Embeds each chunk with the configured `EMBEDDING_MODEL` (via the LLM adapter — Gemini or Ollama).
4. Drops the existing collection of that name and writes a fresh one to `--db-dir`.

If you skip this step the agent still runs — `_get_rag_context()` simply returns `"(No RAG context — run build_embeddings to populate Chroma with Pydoll code)"` and the LLM has to generate Pydoll code from scratch.

---

## Running the agent (entry point)

```bash
python run_agent.py
```

This is a one-line shim:

```9:9:run_agent.py
from web_agent_framework.chat_cli import main
```

`chat_cli.main()` calls `asyncio.run(run_chat_cli())` and drops you into a REPL:

```
============================================================
  Autonomous Web Agent — 3-Phase Chat CLI
============================================================

Commands:
  <URL>           — Load and prepare a page (Phase 1)
  menu / m        — After page is ready: pick Actions vs Chat again
  exit / quit     — Exit the agent

URL>
```

Typical interaction:

```
URL> https://example.com/products/123

[Phase 1] Preparing page: https://example.com/products/123
  Fetching HTML, handling blockers, expanding content, building DOM graph...
[Phase 1] Fetching initial HTML...
[Phase 1] LLM analyzing blockers...
[Phase 1] Executing blocker scripts...
[Phase 1] Verifying blockers gone...
[Phase 1] Clean HTML re-fetch...
[Phase 1] Content expansion with redirect guard...
[Phase 1] Building DOM graph...
[Phase 1] Indexing nodes into NodeRAG...

============================================================
  Page analyzed and ready.
  Job id: job_20260507_120145
============================================================

  Choose how to continue (same job / saved nodes):
    [1] Actions — automate the page with Pydoll (plan → generated steps)
    [2] Chat    — ask about this page (semantic search over indexed DOM nodes)

Mode [1/2]> 1

  Action mode — describe what to do on the page (Pydoll).

Action> get the price and add to cart
[Phase 3] Actions (Pydoll): get the price and add to cart
[Phase 3] Planning steps from user demand...
[Phase 3] Executing steps with verify/debug loop...
...
----------------------------------------
Agent: { ...JSON of step outputs... }
----------------------------------------

Action> menu
  Switched to Chat mode.

Chat> what is the brand name on this page?
[Phase 3] Querying NodeRAG (chat mode)...
...
```

`chat_cli.run_chat_cli()` keeps the prepared `agent_state` in memory between turns; both Action and Chat reuse the same `job_dir`, so the DOM graph and node index from Phase 1 are shared.

---

## How a request flows through the code

Source of truth: `web_agent_framework/pipelines/three_phase_pipeline.py`. The CLI calls one function:

```735:740:web_agent_framework/pipelines/three_phase_pipeline.py
async def run_three_phase_agent(
    url: str,
    user_demand: Optional[str] = None,
    prepared_state: Optional[Dict[str, Any]] = None,
    phase3_mode: Optional[str] = None,
) -> Dict[str, Any]:
```

- `user_demand=None`  → run **Phase 1 only**, return prepared state.
- `user_demand` and `prepared_state` set → run **Phase 3 only**, return result.

Both branches are LangGraph `StateGraph(ThreePhaseState)` graphs compiled in `create_phase1_graph()` and `create_phase3_graph()`. The shared state schema:

```123:141:web_agent_framework/pipelines/three_phase_pipeline.py
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
```

### Phase 1 — Page preparation

LangGraph wiring:

```656:664:web_agent_framework/pipelines/three_phase_pipeline.py
    workflow.set_entry_point("fetch")
    workflow.add_edge("fetch", "analyze_blockers")
    workflow.add_edge("analyze_blockers", "execute_blockers")
    workflow.add_edge("execute_blockers", "verify_blockers")
    workflow.add_conditional_edges("verify_blockers", route_after_verify)
    workflow.add_edge("clean_refetch", "expansion")
    workflow.add_edge("expansion", "build_graph")
    workflow.add_edge("build_graph", "index_nodes")
    workflow.add_edge("index_nodes", END)
```

| Node | What it does |
| ---- | ------------ |
| **`fetch_initial_html_node`** | Spawns the configured browser adapter (`get_browser_adapter()` → `MCPBrowserAdapter`), navigates to `url`, waits 5 s, snapshots `document.documentElement.outerHTML`. |
| **`analyze_blockers_node`** | Sends the first 8000 chars of the HTML to the LLM with a prompt that returns `{has_blockers, blocker_descriptions, suggested_selectors}`. Stored in `state["shared"]["blocker_analysis"]`. |
| **`execute_blocker_scripts_node`** | Re-navigates and runs `js_click_cookieish()` (a heuristic that clicks any visible button whose text matches accept/agree/close/etc.). |
| **`verify_blockers_gone_node`** | Asks the LLM `{blockers_still_present: bool}` over 6000 chars of fresh HTML. Stores the result in `state["shared"]["blockers_verified_gone"]`. |
| **`route_after_verify`** | Conditional edge. If blockers are gone or `blocker_retry_count >= 3` → goes to `clean_refetch`. Otherwise increments the retry counter and re-runs `execute_blockers`. |
| **`clean_html_refetch_node`** | A final clean re-navigate + HTML capture after the blocker loop converged. |
| **`content_expansion_redirect_guard_node`** | Up to 3 passes of: `scrollTo(0,500)` → `js_universal_scroll()` (15 step scrolls of 600 px each, 400 ms apart) → `js_expand_sections_universal()` (clicks anything labelled "show more / load more / view comments / read more / expand"). After each pass it reads `location.href`; if it changed (some sites redirect on interaction) it navigates back to the original URL. |
| **`build_dom_graph_node`** | Re-navigates, repeats scroll/expand, then runs `js_get_smart_dom_tree()` — a JS walker that emits `{id, parentId, tag, depth, attrs, text, bgUrl, jqueryData, rect, computed, [src/data_src/srcset/alt for img], [href for a]}` for every visible element (skipping `style/noscript/svg/path/iframe/head/meta/link`). It builds a `networkx.DiGraph` from the `(id, parentId)` pairs. If the JS payload comes back empty, it falls back to BeautifulSoup parsing via `core.dom_graph.DOMGraphBuilder`. The graph is fed to `perform_smart_chunking(G, url)` which scores each subtree by area, image count, gallery hints and noise patterns, returning a list of "chunks" — each is a self-contained semantic region of the page. The function then writes `graph_chunks.json`, `html_context.txt`, `dom_js_hints.txt`, `graph.json`, `graph.png` (rendered via matplotlib if available) and the pickled HTML chunks under `html/`. |
| **`index_nodes_rag_node`** | Walks every node in the DiGraph, builds an embed-text per node (`Tag/Text/Attrs/Hint`), embeds via the LLM adapter, and stores in the per-job vector store under `<job_dir>/node_rag/`. Backend is selectable (`NODE_RAG_BACKEND`). For Chroma the full payloads are persisted in `nodes_by_id.json`; for FAISS in `node_data.json` plus `faiss_index.bin`. |

When the graph reaches `END`, the state has `status="ready_for_demand"` and the CLI moves on.

### Phase 2 — Mode selection

There is no LangGraph node for Phase 2: it lives in `chat_cli.prompt_mode_choice()`. The user picks Action (`1`) or Chat (`2`). That choice is stored as `session_mode` and passed to subsequent calls as `phase3_mode`.

### Phase 3 — Targeted operation execution

Wiring:

```715:730:web_agent_framework/pipelines/three_phase_pipeline.py
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
```

`detect_intent()` honours the explicit `phase3_mode` first:

- `phase3_mode="chat"` → `query_rag`
- `phase3_mode="action"` → `plan` → `execute`
- `None` → LLM classifies the demand as `"query"` (info about the page) or `"action"` (do something).

#### Phase 3a — Action mode (Pydoll execution)

1. **`plan_from_demand_node` → `DemandPlanner.plan_for_demand(...)`** — Loads the saved `graph_chunks.json`, `html_context.txt`, `dom_js_hints.txt` from the job dir. Filters chunks by demand using `core.dom_graph.get_chunks_for_demand()`. Sends a JSON-only prompt to the LLM that returns `{"steps":[{id, goal}, …]}` of 1–5 steps.

2. **`execute_step_with_verify_node`** — for each step:

   1. **`graph_query.query_graph_for_step(...)`** — asks the LLM "what tags / attrs / keywords / semantic_hints / data_schema / child_selector_hint do I need?" (`extract_requirements_via_llm`), then runs a **generic scoring search** over the chunks **and** the full graph nodes (`query_graph_for_requirements`, `_nodes_to_selectors`). If nothing matches, falls back to top chunks by `visual_importance` (`used_empty_fallback=True`).
   2. **`StepGenerator.generate_step(...)`** — builds a prompt containing:
      - the graph-query result (selectors, data schema, child-selector hint, sample nodes),
      - top-K Pydoll/Selenium code chunks pulled from the **code RAG** (Chroma collection `pydoll_code`),
      - up to 3500 chars of HTML,
      - relevant fields from `shared`,
      and asks the LLM to emit `async def run_step(tab, shared): ...`. The system prompt is the verbatim contents of `agent/step_system_prompt.md` (forbids navigation, requires `JSON.stringify` IIFEs, demands type guards, etc.). Generated code is validated against a forbidden-token list (`pydoll.commands`, `query_selector`, `playwright`, `selenium`, …).
   3. **`runner.save_step_module(sid, code)`** — writes the code as `<job_dir>/steps/step_<sid>.py`.
   4. **`runner.run_in_process(page, [sid], shared)`** — adds the job dir to `sys.path`, imports `steps.step_<sid>` (force-reloads if it was already imported), and `await`s its `run_step(tab, shared)` against the live Pydoll page.
   5. **Verification** — captures fresh HTML and asks the LLM `{achieved: bool, evidence: str}` against the step's goal.
   6. **Debug loop** — if `achieved=False` or the run threw, the loop calls `query_graph_for_step()` again with `previous_requirements` and `debug_error` set (`plan_debug_requirements()` asks the LLM what to expand/change), then regenerates the step in `repair_mode=True`. Up to `max_debug_retries=2` extra attempts.

   When all steps are done (or one fails terminally), `shared["step_results"]` is dumped to `<job_dir>/steps/step_outputs.json`. The CLI prints the partial JSON.

#### Phase 3b — Chat mode (Node RAG)

`query_node_rag_node` in `three_phase_pipeline.py`:

1. Reads `<job_dir>/node_rag/nodes_by_id.json` (returns a friendly error if missing).
2. Calls `rag.query(user_demand, llm, k=10)` — embeds the user demand (using `task_type="retrieval_query"` when supported), runs a vector search against the Chroma/FAISS store, and returns the top 10 node payloads.
3. Builds a prompt that includes the full JSON for those nodes and the user's question, and asks the LLM to answer **only** from the data provided. The answer is stored as `state["result"]` and printed by the CLI.

No browser is opened in chat mode.

---

## Per-job output (what gets written to disk)

Every Phase 1 run creates a fresh directory under `TEMP_ROOT`:

```
temp_pydoll/job_20260507_120145/
├── html/
│   ├── raw.html                    # full HTML after expansion
│   ├── html_chunk_0000.pkl         # 1024-char pickled chunks
│   ├── html_chunk_0001.pkl
│   └── ...
├── graph.json                      # node_link_data dump of the DOM DiGraph
├── graph.png                       # matplotlib visualisation (top 200 nodes)
├── graph_chunks.json               # smart-chunk view (one entry per semantic region)
├── html_context.txt                # truncated raw HTML for prompts
├── dom_js_hints.txt                # interesting class/id tokens, JS framework hints
├── node_rag/                       # per-job node vector store
│   ├── nodes_by_id.json            # full node payloads keyed by id
│   ├── chroma.sqlite3              # (Chroma backend)
│   └── ...                         # or faiss_index.bin + node_data.json
└── steps/                          # only created when Action mode runs
    ├── __init__.py
    ├── step_<id1>.py               # generated Pydoll modules
    ├── step_<id2>.py
    └── step_outputs.json           # accumulated shared["step_results"]
```

The `chat_cli` shows the basename of `job_dir` as the "Job id" so you can find it on disk later.

---

## Module reference

This section maps each file to the single thing it owns.

### Top-level
- **`run_agent.py`** — `if __name__ == "__main__": main()`. Imports from `web_agent_framework.chat_cli`.

### `web_agent_framework/chat_cli.py`
- `extract_url(text)` — find the first `https?://...` in a string.
- `prompt_mode_choice()` — read 1/2 from stdin.
- `run_chat_cli()` — async REPL.
- `main()` — `asyncio.run(run_chat_cli())`.

### `web_agent_framework/config.py`
A flat module of constants (see [Configuration](#configuration)).

### `web_agent_framework/pipelines/`
- **`three_phase_pipeline.py`** — primary pipeline; defines `ThreePhaseState`, all node functions, `create_phase1_graph()`, `create_phase3_graph()`, and the public `run_three_phase_agent()`.
- **`autonomous_agent_pipeline.py`** — a smaller legacy 3-step pipeline (`bootstrap → plan → execution`) that uses the legacy `Planner` and a hard-coded list of `ALLOWED_STEP_IDS`. Not used by the CLI.
- **`ecommerce_scraper_graph.py`** — legacy e-commerce scraper (`fetch → build → chunk → details → images`) wired around `EcommerceGraphScraper`. Useful as an example of plugging custom nodes into the framework.

### `web_agent_framework/agent/`
- **`demand_planner.py`** — `DemandPlanner.plan_for_demand(user_demand, html_context, dom_js_hints, graph_chunks_text)` returns `{"steps":[{id, goal}]}` from an LLM prompt. Used in Phase 3a.
- **`planner.py`** — legacy `Planner.plan_steps(...)` constrained to `ALLOWED_STEP_IDS`, plus `build_step_contexts(...)` that fills per-step prompts from the code RAG.
- **`step_generator.py`** — `StepGenerator.generate_step(...)`: (a) builds the prompt (graph-query result *or* DOM chunks → RAG context → HTML excerpt → shared excerpt), (b) calls the LLM with the system prompt from `step_system_prompt.md`, (c) extracts code from a `python` fenced block, (d) validates against forbidden tokens.
- **`step_system_prompt.md`** — the verbatim system prompt used when generating step modules. Read it before extending; it defines the public `run_step(tab, shared)` contract, the JS-IIFE+`JSON.stringify` rule, what is forbidden, and the page-analysis-first architecture.
- **`runner.py`** — `Runner.save_step_module(...)` writes a `step_<id>.py`; `Runner.run_in_process(...)` imports and runs it. Errors abort the loop unless the step ID is `handle_blockers`.
- **`output_processor.py`** — `extract_product_image_urls(stdout)`, `is_real_image_url(url)`, `clean_product_image_urls(...)`, `detect_suspect_step(...)` — heuristics for cleaning/parsing legacy stdout markers (`PRODUCT_IMAGES:`, `STEP <id> OK`).

### `web_agent_framework/core/`
- **`dom_graph.py`**
  - `DOMGraphBuilder` — BeautifulSoup-based fallback when no live JS DOM is available.
  - `perform_smart_chunking(G, url)` — multi-criteria scoring (gallery containers, area windows, text length, noise/grid penalties, spatial scoring).
  - `_fallback_text_chunking(...)` — used when the graph has no rect data.
  - `get_chunks_for_step(...)` / `get_chunks_for_demand(...)` — keyword/semantic-hint scoring filters.
  - `plot_dom_graph(...)` — saves a `graph.png` (silently no-op if matplotlib is missing).
- **`graph_query.py`** — the heart of the requirement-driven flow:
  - `extract_requirements_via_llm(...)` — first LLM call ("what does the step need?")
  - `plan_debug_requirements(...)` — second LLM call after a step fails ("what should we ask for now?")
  - `query_graph_for_requirements(...)` — generic scoring over chunks (no hardcoded selectors)
  - `_nodes_to_selectors(...)` — same scoring over the full nodelist from `graph.json`
  - `query_graph_for_step(...)` — orchestrator the pipeline calls
- **`browser_js_snippets.py`** — small JS programs that get `tab.execute_script(...)`-ed:
  - `js_get_smart_dom_tree()` — flat list of every visible element (with rects, jQuery data, computed style, bgUrl, etc.)
  - `js_click_cookieish()` — best-effort blocker dismiss
  - `js_expand_sections_universal()` — clicks "show/load/view more" things
  - `js_universal_scroll()` — 15-step scroll for lazy content
  - `js_probe_image_meta()` — preload images and report natural width/height
- **`html_context.py`** — pickle/restore HTML, build sliding windows, and extract DOM/JS hints (interesting class tokens, framework markers, blocker signals).
- **`url_utils.py`** — `normalize_url`, `get_image_family_key`, `get_url_res_score`.
- **`image_promotion.py`** — generate higher-resolution variants from sized image URLs (Amazon-style `_SL`, generic width/height tokens).
- **`utils.py`** — `ensure_dir`, `write_text`, `extract_url_from_task`, `infer_scrape_intent`, `build_annotated_task`.

### `web_agent_framework/adapters/`
- **`browser.py`** — `BrowserAdapter` and `PageAdapter` `Protocol`s. `PageAdapter` defines: `navigate, execute_script, get_html, screenshot, click, fill`.
- **`pydoll_adapter.py`** — wraps `pydoll.browser.Chrome` to fit those protocols. `click`/`fill` are JS-based (works around APIs missing on `pydoll.tab`).
- **`mcp.py`** — `MCPBrowserAdapter` spawns the internal MCP server (or connects to `MCP_SERVER_URL`) over stdio using `mcp.ClientSession`. Returns an `MCPPageAdapter` whose `execute_script` / `navigate` / `get_html` are tool calls. **`get_browser_adapter()`** is the factory the pipeline uses — always returns an `MCPBrowserAdapter`.
- **`llm.py`** — `LLMAdapter` Protocol: `generate, chat, generate_json, embed`.
- **`llm_factory.py`** — `get_llm_adapter()` lazily imports either Gemini or Ollama based on `LLM_PROVIDER`.
- **`gemini_adapter.py`** — `google.generativeai` wrapper. `generate_json` uses `response_mime_type="application/json"`. `embed` uses `genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type=...)`.
- **`ollama_adapter.py`** — `ollama.chat`/`ollama.embeddings` wrapper, with simple JSON cleanup in `generate_json`.
- **`rag.py`** — `ChromaOllamaRAG`: code-RAG client (Chroma collection populated by `build_embeddings.py`). Embeds the query through whichever LLM adapter is configured.
- **`node_rag.py`** — two backends:
  - `ChromaNodeRAGAdapter` (default) — per-job Chroma collection plus `nodes_by_id.json` for full payloads.
  - `FaissNodeRAGAdapter` — `faiss.IndexFlatL2` plus `node_data.json`.
  Both share `_node_to_embed_text(node)` and stop indexing on auth errors (`_embedding_error_is_fatal`).
- **`node_rag_factory.py`** — `get_node_rag_adapter(storage_dir)` picks the backend based on `NODE_RAG_BACKEND`.

### `web_agent_framework/mcp_servers/`
See [MCP servers](#mcp-servers-optional).

---

## MCP servers (optional)

The framework can drive a browser through the **Model Context Protocol** rather than calling Pydoll/Selenium in-process. This is useful for:

- Running the browser in a separate process (so a crash in Chrome doesn't take down the agent).
- Letting other MCP-aware tools share the same browser session.

Two stdio servers are bundled:

- `web_agent_framework/mcp_servers/mcp_server_pydoll.py` — wraps Pydoll. Notably, it auto-wraps every `browser_execute_script` call in `(function(){ ... })()` and `JSON.stringify(...)` so the CDP response is always a JSON string (a workaround for the dreaded "non-serializable script result" CDP error).
- `web_agent_framework/mcp_servers/mcp_server_selenium.py` — wraps Selenium with `--headless=new`.

Both expose the same four tools:

| Tool                       | Args                              | Returns |
| -------------------------- | --------------------------------- | ------- |
| `browser_navigate`         | `{url: str}`                       | `{ok: true}` |
| `browser_execute_script`   | `{script: str, args?: list}`       | string (JSON or HTML or value) |
| `browser_get_html`         | `{}`                              | full `document.documentElement.outerHTML` |
| `browser_close`            | `{}`                              | `{ok: true}` |

When `MCP_SERVER_URL` is empty (the default), `MCPBrowserAdapter._ensure_session()` calls `_get_internal_mcp_server_command()` and spawns one of those modules with `python -u -m web_agent_framework.mcp_servers.mcp_server_<MCP_BROWSER>`. If you set `MCP_SERVER_URL=stdio:npx -y @modelcontextprotocol/server-browser`, it connects to that external server instead.

You can run the servers stand-alone for testing:

```bash
python -m web_agent_framework.mcp_servers.mcp_server_pydoll
# or
python -m web_agent_framework.mcp_servers.mcp_server_selenium
```

---

## Step generation contract

Generated step modules are **plain Python files** that live under `<job_dir>/steps/step_<id>.py`. The single rule (enforced by `StepGenerator._validate_generated_step_code`) is:

```python
async def run_step(tab, shared):
    ...
```

Inside that function the LLM may **only** use `await tab.execute_script(js)` to interact with the page. The full set of forbidden tokens, JS-IIFE rules, type-guard rules, and the "page-analysis-first" architecture are documented in `web_agent_framework/agent/step_system_prompt.md` — that file is the contract. Modify it carefully: the prompt is loaded verbatim and is the most important knob in the whole framework.

A skeletal valid step:

```python
import json

async def run_step(tab, shared):
    js = r"""
    return JSON.stringify((function(){
        const out = {};
        const el = document.querySelector('h1');
        out.title = el ? el.textContent.trim() : null;
        return out;
    })());
    """
    raw = await tab.execute_script(js)
    data = json.loads(raw) if isinstance(raw, str) else raw

    step_results = shared.get("step_results") or {}
    step_results["my_step"] = {"ok": True, "data": data}
    shared["step_results"] = step_results
```

---

## Extending the framework

### Add a new LLM provider

1. Add a class in `adapters/` that implements `LLMAdapter` (`generate`, `chat`, `generate_json`, `embed`).
2. Wire it into `adapters/llm_factory.py` behind a new value of `LLM_PROVIDER`.
3. Add the embedding-model default in `config._DEFAULT_EMBED`.

### Add a new browser

1. Implement `BrowserAdapter`/`PageAdapter` in `adapters/`.
2. Either:
   - Make a new MCP server in `mcp_servers/` (preferred — gives you process isolation), and add it to `_get_internal_mcp_server_command` via the `MCP_BROWSER` env var, **or**
   - Bypass MCP and import your adapter directly in `pipelines/three_phase_pipeline._get_browser`.

### Add a new pipeline

The 3-phase pipeline is the canonical example. Build a new `StateGraph(MyState)` in `pipelines/`, wire your nodes, and call it from a new entry point or from `chat_cli`. Any LangGraph node has access to the LLM via `get_llm_adapter()` and to the browser via `get_browser_adapter()`.

### Change Node RAG embedding text

Edit `_node_to_embed_text(node)` in `adapters/node_rag.py`. The function is shared by both backends.

---

## Troubleshooting

| Symptom | Likely cause / fix |
| ------- | ------------------ |
| `GEMINI_API_KEY or GOOGLE_API_KEY required for LLM_PROVIDER=gemini` | Set `GEMINI_API_KEY` in `.env` or environment. |
| Embedding spam stops with `403`/`leaked`/`API key not valid` | The key was revoked. Create a new one and update `.env`. The framework intentionally aborts bulk indexing on these errors (see `_embedding_error_is_fatal`). |
| Step keeps failing with "non-serializable" | Generated code returned a raw object from JS. The `mcp_server_pydoll.py` wrapper guards against this, but if you bypass MCP make sure your JS uses `return JSON.stringify((function(){…})())`. |
| `[NodeRAG/Chroma] nodes_by_id.json not found` in chat mode | The Phase 1 indexing step was skipped or failed. Reload the URL — `chat_cli` prints "Re-load the URL to re-run page analysis and indexing." |
| `[StepGenerator] Validation failed: Contains forbidden token/API: query_selector` | The LLM returned code that uses banned APIs. Re-run; the step prompt forbids these. If it persists, harden the system prompt or reduce the model's freedom. |
| `(No RAG context — run build_embeddings to populate Chroma with Pydoll code)` | The code-RAG collection is empty. Run `python -m web_agent_framework.build_embeddings`. |
| `[Phase 1] Could not save graph.png` | `matplotlib` is not installed. Optional, ignore. |
| MCP server doesn't start | Missing `mcp` / `anyio`. `pip install mcp anyio`. Check `MCP_BROWSER`. The error is printed once: `[MCP] mcp package not installed.` |
| Job dir piles up under `temp_pydoll/` | Expected. The framework never auto-deletes jobs. Add a cron job or `rm -rf temp_pydoll/job_*` periodically. The path is configurable with `TEMP_ROOT`. |

---

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 manu goel.
